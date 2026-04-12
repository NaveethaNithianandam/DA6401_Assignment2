"""Unified multi-task model
"""

import torch
import torch.nn as nn
from pathlib import Path

from models.vgg11 import VGG11Encoder
from models.classification import ClassificationHead
from models.localization import LocalizationHead
from models.segmentation import DecoderBlock, FinalUpsample


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task perception model.

    A single forward pass through the shared VGG11 encoder produces three
    simultaneous outputs:
        - classification  : [B, num_breeds]        breed logits
        - localization    : [B, 4]                 (cx, cy, w, h) pixel coords
        - segmentation    : [B, seg_classes, H, W] pixel-wise class logits

    Architecture
    ------------
    Shared encoder: VGG11Encoder (frozen; weights from classifier.pth)
      Exposes skip connections at pool1-pool5 for the segmentation decoder.
      The avgpool output (512 @ 7x7) feeds classification and localization.

    Classification head : ClassificationHead  (weights from classifier.pth)
    Localization head   : LocalizationHead    (weights from localizer.pth)
    Segmentation decoder: bottleneck + dec5..dec1 + 1x1 conv (from unet.pth)

    All three tasks branch from the same encoder forward pass -- no image is
    processed twice.

    Checkpoint paths (relative to project root, as required):
        checkpoints/classifier.pth
        checkpoints/localizer.pth
        checkpoints/unet.pth
    """

    CLASSIFIER_CKPT = "checkpoints/classifier.pth"
    LOCALIZER_CKPT  = "checkpoints/localizer.pth"
    UNET_CKPT       = "checkpoints/unet.pth"

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3,
                 in_channels: int = 3):
        import gdown
        gdown.download(id="1D1KxzKCOwRsJywKaJFxXqHWCsCatSiyj", output=self.CLASSIFIER_CKPT, quiet=False)
        gdown.download(id="13X1m-0cOFMo8rkF5Qor92I7TNEDZpnwx", output=self.LOCALIZER_CKPT, quiet=False)
        gdown.download(id="1NqsTV_knzhL2a5A5WVF18U3GWSdqGVRs", output=self.UNET_CKPT, quiet=False)


        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.cls_head = ClassificationHead(num_classes=num_breeds)

        
        self.loc_head = LocalizationHead()

      
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

        self.final_conv = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load_checkpoints()

    def _load_checkpoints(self):
        """Load trained weights from all three task checkpoints.

        Strategy:
          - classifier.pth  -> encoder weights + cls_head weights
          - localizer.pth   -> loc_head weights only
                              (encoder already loaded from classifier)
          - unet.pth        -> bottleneck + dec5..dec1 + final_conv
                              (encoder NOT re-loaded; classifier weights used)

        The encoder is loaded from classifier.pth because that is where it
        was fine-tuned most extensively.  The UNet encoder weights would give
        the same features but with slight segmentation-specific fine-tuning;
        using the classifier version is the safer default for a shared backbone.
        """
        if Path(self.CLASSIFIER_CKPT).exists():
            state = torch.load(self.CLASSIFIER_CKPT, map_location='cpu',
                               weights_only=True)
            enc_state = {k[len('encoder.'):]: v
                         for k, v in state.items() if k.startswith('encoder.')}
            self.encoder.load_state_dict(enc_state, strict=True)
            cls_state = {k[len('head.'):]: v
                         for k, v in state.items() if k.startswith('head.')}
            self.cls_head.load_state_dict(cls_state, strict=True)
            print(f"  [multitask] encoder + cls_head loaded from {self.CLASSIFIER_CKPT}")
        else:
            print(f"  [multitask] WARNING: {self.CLASSIFIER_CKPT} not found")

        if Path(self.LOCALIZER_CKPT).exists():
            state = torch.load(self.LOCALIZER_CKPT, map_location='cpu',
                               weights_only=True)
            loc_state = {k[len('head.'):]: v
                         for k, v in state.items() if k.startswith('head.')}
            self.loc_head.load_state_dict(loc_state, strict=True)
            print(f"  [multitask] loc_head loaded from {self.LOCALIZER_CKPT}")
        else:
            print(f"  [multitask] WARNING: {self.LOCALIZER_CKPT} not found")

        if Path(self.UNET_CKPT).exists():
            state = torch.load(self.UNET_CKPT, map_location='cpu',
                               weights_only=True)

            def _extract(prefix):
                return {k[len(prefix):]: v
                        for k, v in state.items() if k.startswith(prefix)}

            self.bottleneck.load_state_dict(_extract('bottleneck.'), strict=True)
            self.dec5.load_state_dict(_extract('dec5.'),       strict=True)
            self.dec4.load_state_dict(_extract('dec4.'),       strict=True)
            self.dec3.load_state_dict(_extract('dec3.'),       strict=True)
            self.dec2.load_state_dict(_extract('dec2.'),       strict=True)
            self.dec1.load_state_dict(_extract('dec1.'),       strict=True)
            self.final_conv.load_state_dict(_extract('final_conv.'), strict=True)
            print(f"  [multitask] seg decoder loaded from {self.UNET_CKPT}")
        else:
            print(f"  [multitask] WARNING: {self.UNET_CKPT} not found")

    def forward(self, x: torch.Tensor):
        """Single forward pass producing all three task outputs simultaneously.

        Args:
            x: [B, in_channels, H, W]  normalised input image (H=W=224).

        Returns:
            dict with keys:
                'classification': [B, num_breeds]        logits
                'localization'  : [B, 4]                 (cx, cy, w, h) pixels
                'segmentation'  : [B, seg_classes, H, W] logits
        """
  
        bottleneck_feat, skips = self.encoder(x, return_features=True)
    
        cls_out = self.cls_head(bottleneck_feat)      

        loc_out = self.loc_head(bottleneck_feat)        

        seg = self.bottleneck(skips['pool5'])            # [B, 1024, 7, 7]
        seg = self.dec5(seg, skips['pool4'])             # [B,  512, 14, 14]
        seg = self.dec4(seg, skips['pool3'])             # [B,  256, 28, 28]
        seg = self.dec3(seg, skips['pool2'])             # [B,  128, 56, 56]
        seg = self.dec2(seg, skips['pool1'])             # [B,   64, 112, 112]
        seg = self.dec1(seg)                             # [B,   32, 224, 224]
        seg_out = self.final_conv(seg)                   # [B, seg_classes, 224, 224]

        return {
            'classification': cls_out,
            'localization':   loc_out,
            'segmentation':   seg_out,
        }