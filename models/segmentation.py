"""Segmentation model -- VGG11 encoder + U-Net decoder
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class DecoderBlock(nn.Module):
    """One stage of the U-Net expansive path.

    Structure:
        ConvTranspose2d (learnable upsampling, stride 2) ->
        channel-wise concatenation with encoder skip ->
        Conv3x3 -> BN -> ReLU ->
        Conv3x3 -> BN -> ReLU

    Args:
        in_ch:   channels coming into this block (from the previous decoder stage)
        skip_ch: channels of the corresponding encoder skip connection
        out_ch:  output channels after the two conv layers

    Upsampling:
        ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2) halves channels
        and exactly doubles spatial dimensions (7->14, 14->28, ..., 112->224).
        Because all VGG11 pool outputs are even-sized (7, 14, 28, 56, 112) and
        ConvTranspose2d with k=2, s=2 produces exactly 2x the input size,
        no padding or interpolation is ever required for size alignment.

    No bilinear/bicubic interpolation is used anywhere -- upsampling is
    exclusively via transposed convolutions, as required by the assignment.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
    
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                     kernel_size=2, stride=2, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                    
        x = torch.cat([x, skip], dim=1) 
        return self.conv(x)            


class FinalUpsample(nn.Module):
    """Final decoder stage: 112x112 -> 224x224, no skip connection.

    The encoder has 5 MaxPool layers (strides 2), so the bottleneck is
    7x7 (224 / 2^5 = 7).  Four DecoderBlocks recover 7->112.  This block
    performs the fifth and final doubling (112->224) using a transposed
    convolution, followed by two conv layers to refine the feature maps.
    There is no corresponding encoder skip at this resolution because the
    VGG11 encoder does not expose pre-pool1 features (the input image itself).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch,
                                     kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)    
        return self.conv(x)  


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder.

    Architecture
    ------------
    Encoder (contracting path): VGG11 convolutional backbone.
      Captures skip connections at each of the 5 MaxPool outputs:
        pool1: [B,  64, 112, 112]
        pool2: [B, 128,  56,  56]
        pool3: [B, 256,  28,  28]
        pool4: [B, 512,  14,  14]
        pool5: [B, 512,   7,   7]

    Bottleneck: 512 -> 1024 channels at 7x7 (deepens representation).

    Decoder (expansive path): symmetric to encoder, 5 upsampling stages.
      Each of the first 4 stages uses a DecoderBlock (ConvTranspose2d +
      skip concatenation + 2x Conv-BN-ReLU).  The 5th stage uses
      FinalUpsample (ConvTranspose2d + Conv-BN-ReLU, no skip).

    Channel flow:
        bottleneck    1024 @ 7x7
        dec5  1024 -> 512 up -> concat pool4(512) -> 512 @ 14x14
        dec4   512 -> 256 up -> concat pool3(256) -> 256 @ 28x28
        dec3   256 -> 128 up -> concat pool2(128) -> 128 @ 56x56
        dec2   128 ->  64 up -> concat pool1(64)  ->  64 @ 112x112
        dec1    64 ->  32 up (no skip)             ->  32 @ 224x224
        output: 1x1 conv -> num_classes @ 224x224

    All upsampling is done exclusively via ConvTranspose2d.
    No bilinear interpolation is used anywhere in the forward pass.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

       
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.dec5 = DecoderBlock(1024, 512, 512)   
        self.dec4 = DecoderBlock(512,  256, 256) 
        self.dec3 = DecoderBlock(256,  128, 128)  
        self.dec2 = DecoderBlock(128,   64,  64)   

     
        self.dec1 = FinalUpsample(64, 32)         

    
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv/transposed-conv, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_encoder_from_classifier(self, classifier_ckpt: str,
                                      device: torch.device = None):
        """Initialise encoder from a saved VGG11Classifier checkpoint."""
        import os
        if not os.path.exists(classifier_ckpt):
            print(f"  [warn] Classifier ckpt not found at {classifier_ckpt}")
            return
        map_dev = device or torch.device('cpu')
        state = torch.load(classifier_ckpt, map_location=map_dev)
        enc_state = {k[len('encoder.'):]: v
                     for k, v in state.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(enc_state, strict=True)
        print(f"  [init] Encoder loaded from {classifier_ckpt}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, in_channels, H, W]  (H=W=224 expected)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
     
        _, skips = self.encoder(x, return_features=True)
     
        x = self.bottleneck(skips['pool5'])  
        x = self.dec5(x, skips['pool4'])     
        x = self.dec4(x, skips['pool3'])    
        x = self.dec3(x, skips['pool2'])  
        x = self.dec2(x, skips['pool1'])   
        x = self.dec1(x)                   

        return self.final_conv(x)           