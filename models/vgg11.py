"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout



VGG11_CFG = [
    64, 'M',
    128, 'M',
    256, 256, 'M',
    512, 512, 'M',
    512, 512, 'M',
]


def _make_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv → BN → ReLU block.

    Design rationale
    ----------------
    BatchNorm is placed *between* Conv and ReLU (pre-activation style for BN
    is debated, but post-conv / pre-ReLU is the most common convention and
    produces stable gradients). Inserting BN after every conv layer:
      • Reduces internal covariate shift, allowing higher learning rates.
      • Acts as a mild regulariser, reducing the need for weight decay alone.
      • Makes the network much less sensitive to weight initialisation.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), 
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional encoder with BatchNorm after every conv.

    Implements the VGG-11 architecture from Simonyan & Zisserman (2015),
    Table 1, column A.  The only permitted departures are:
      • BatchNorm2d after every Conv2d (before ReLU).
      • The classifier head is left to downstream modules; this encoder
        outputs the 512-d spatially-pooled bottleneck (after AdaptiveAvgPool).

    The ``return_features`` flag exposes intermediate skip-connection tensors
    needed by the U-Net decoder (Task 3).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()


        layers = []
        in_ch = in_channels
        self._pool_indices = []  
        block_idx = 0
        for v in VGG11_CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(_make_conv_block(in_ch, v))
                in_ch = v
            block_idx += 1

        self.features = nn.Sequential(*layers)

  
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

  
    _SKIP_LAYERS = {
        'pool1': 1,   
        'pool2': 3,  
        'pool3': 6,  
        'pool4': 9,  
        'pool5': 12,  
    }

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through VGG11 convolutional blocks.

        Args:
            x: Input image tensor [B, in_channels, H, W].
                Per the VGG paper H=W=224 is expected, but the adaptive pool
                handles other sizes gracefully.
            return_features: If True, also return a dict of intermediate
                feature maps keyed by 'pool1' … 'pool5' for skip connections.

        Returns:
            bottleneck: [B, 512, 7, 7] after AdaptiveAvgPool.
            feature_dict (only when return_features=True):
                {'pool1': ..., 'pool2': ..., 'pool3': ..., 'pool4': ..., 'pool5': ...}
        """
        if not return_features:
            x = self.features(x)
            x = self.avgpool(x)
            return x

        feature_dict: Dict[str, torch.Tensor] = {}
        skip_at = {v: k for k, v in self._SKIP_LAYERS.items()}

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in skip_at:
                feature_dict[skip_at[idx]] = x

        x = self.avgpool(x)
        return x, feature_dict