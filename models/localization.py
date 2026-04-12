"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class LocalizationHead(nn.Module):
    """Lightweight regression head: 512x7x7 -> [cx, cy, w, h].

    Design rationale -- why a small head (not 4096x4096)
    -----------------------------------------------------
    The Oxford-IIIT Pets dataset has ~3,680 training images.
    The standard VGG FC head (25088->4096->4096->4) has ~119M parameters --
    32,490 parameters per training sample -- which memorises training bboxes
    perfectly (train_mIoU=0.73) while generalising poorly (val_mIoU=0.21).

    A compact head (25088->512->4) has ~12.8M parameters (3,491/sample):
      - Still enough representational capacity to learn bbox regression
      - Too small to memorise individual samples, forced to generalise
      - 9x fewer parameters -> significantly reduced overfitting
      - Faster to train, reducing wall-clock time

    Dropout placement: after BN, before the next linear layer (same
    reasoning as the classifier: BN statistics computed on full distribution).
    Dropout p=0.5 provides additional regularisation on top of the smaller
    head size.

    Final activation: Softplus (smooth, strictly positive, non-zero gradient
    everywhere). Avoids degenerate zero-size boxes that ReLU can produce.
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
       
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(512, 4),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class VGG11Localizer(nn.Module):
    """VGG11-based object localizer.

    The encoder is initialised from classifier.pth (pre-trained on the 37-class
    pet classification task).  The rich visual features learned for classification
    transfer directly to localization -- the encoder already knows what a pet
    looks like and where its boundaries are.

    The encoder is kept FROZEN throughout training (not just phase 1).
    Rationale: fine-tuning 9.2M encoder params on only 3,680 bbox annotations
    causes the encoder to overfit, destroying the generalisation learned from
    classification.  A frozen encoder + lightweight trainable head is the
    standard approach for regression on small datasets.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = LocalizationHead(dropout_p=dropout_p)

    def load_encoder_from_classifier(self, classifier_ckpt: str,
                                      device: torch.device = None):
        """Load encoder weights from a saved VGG11Classifier checkpoint."""
        import os
        if not os.path.exists(classifier_ckpt):
            print(f"  [warn] Classifier ckpt not found at {classifier_ckpt}, "
                  f"encoder randomly initialised.")
            return
        map_dev = device if device else torch.device('cpu')
        full_state = torch.load(classifier_ckpt, map_location=map_dev)
        enc_state = {k[len('encoder.'):]: v
                     for k, v in full_state.items()
                     if k.startswith('encoder.')}
        self.encoder.load_state_dict(enc_state, strict=True)
        print(f"  [init] Encoder loaded from {classifier_ckpt}")

    def freeze_encoder(self):
        """Freeze all encoder parameters -- they should not be updated."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Returns:
            Bounding box [B, 4] as (x_center, y_center, width, height),
            values in pixel space (not normalised), all positive.
        """
        
        features = self.encoder(x, return_features=False)
        return self.head(features)