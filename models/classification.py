"""Classification components
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class ClassificationHead(nn.Module):
    """Fully-connected classification head that mirrors the VGG paper's
    classifier (two hidden FC-4096 layers + output FC layer).

    Design rationale for regularisation placement
    ---------------------------------------------
    1. BatchNorm1d after each hidden FC layer:
       • Stabilises training of the deep FC stack by normalising pre-activations.
       • Allows the use of higher learning rates in the dense layers.
       • Empirically reduces overfitting on moderately-sized datasets like
         Oxford-IIIT Pets (7,349 images across 37 classes).

    2. CustomDropout after BN (and before the next FC):
       • Placed *after* BN so that BN statistics are computed on the full
         (un-dropped) distribution, keeping batch statistics meaningful.
       • A drop probability of 0.5 (VGG default) is used; this can be tuned.
       • Dropout before the final linear layer adds a second regularisation
         checkpoint that significantly reduces over-fitting on the last
         representation before the class scores.
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.6):
        super().__init__()
     
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
   
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    This is the end-to-end model for Task 1.  It:
      • Builds a VGG11Encoder backbone (conv blocks + adaptive avg pool).
      • Attaches a ClassificationHead that maps the 512×7×7 bottleneck to
        class logits.
      • Exposes a save_checkpoint / load_checkpoint interface so that the
        trained backbone weights can be re-used in multitask.py.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        """Kaiming-normal for conv layers, Xavier-uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.

        Args:
            x: Input image tensor [B, in_channels, H, W].  Expects 224×224
               normalised images (ImageNet mean/std) per the VGG paper.

        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x, return_features=False) 
        logits = self.head(features)          
        return logits