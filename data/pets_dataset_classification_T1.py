"""Oxford-IIIT Pet dataset loader for all three tasks.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── ImageNet normalisation constants (used by VGG paper) ──────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224   # fixed per VGG11 paper


def get_train_transforms() -> A.Compose:
    """Aggressive augmentation to combat overfitting on the small Pets dataset.

    Design rationale
    ----------------
    • RandomResizedCrop: simulates different viewpoints and scales — the single
      most effective augmentation for recognition tasks.
    • HorizontalFlip: free symmetry augmentation, pets look the same both ways.
    • ColorJitter (stronger): pets are photographed under wildly different
      lighting; heavy colour augmentation prevents the conv features latching
      onto illumination artefacts.
    • Rotate ±15°: pets are rarely perfectly upright; small rotations add
      pose diversity without distorting anatomy.
    • GaussianBlur: simulates out-of-focus shots and forces the model to rely
      on shape/texture rather than high-frequency detail.
    • CoarseDropout (CutOut): randomly blanks rectangular patches, acting as
      a structural regulariser that forces attention to distributed features.
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.6),
        A.Rotate(limit=15, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels'], min_visibility=0.1))


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels'], min_visibility=0.1))


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet Dataset supporting classification, localisation,
    and segmentation tasks.

    Directory layout expected
    -------------------------
    root/
      images/          ← JPEG images  (<breed>_<idx>.jpg)
      annotations/
        list.txt       ← CLASS_ID (1-indexed), SPECIES, BREED_ID
        trainval.txt   ← split file
        test.txt       ← split file
        trimaps/       ← PNG segmentation masks  (1=foreground,2=background,3=not_classified)
        xmls/          ← Pascal VOC bounding-box XML files
    """

    # Map 1-indexed class id (from list.txt) → 0-indexed label
    def __init__(
        self,
        root: str,
        split: str = "trainval",           # "trainval" or "test"
        task: str = "classification",      # "classification" | "localization" | "segmentation"
        transforms: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.task = task
        self.transforms = transforms

        self.samples = self._load_split()

    # ------------------------------------------------------------------
    def _load_split(self) -> List[dict]:
        split_file = self.root / "annotations" / f"{self.split}.txt"
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                img_name   = parts[0]                  # e.g. Abyssinian_1
                class_id   = int(parts[1]) - 1         # 0-indexed
                samples.append({'img_name': img_name, 'label': class_id})
        return samples

    # ------------------------------------------------------------------
    def _load_bbox(self, img_name: str, img_w: int, img_h: int) -> Optional[np.ndarray]:
        """Parse Pascal VOC XML and return [x_center, y_center, w, h] in pixel space."""
        import xml.etree.ElementTree as ET
        xml_path = self.root / "annotations" / "xmls" / f"{img_name}.xml"
        if not xml_path.exists():
            return None
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is None:
            return None
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        # Convert to [x_center, y_center, width, height] in pixel space
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width    = xmax - xmin
        height   = ymax - ymin
        return np.array([x_center, y_center, width, height], dtype=np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample  = self.samples[idx]
        img_name = sample['img_name']
        label    = sample['label']

        # Load image
        img_path = self.root / "images" / f"{img_name}.jpg"
        image = np.array(Image.open(img_path).convert('RGB'))
        orig_h, orig_w = image.shape[:2]

        # Load bbox (for localization / passthrough for others)
        bbox_xyxy = self._load_bbox(img_name, orig_w, orig_h)

        # Default bbox if missing (full image centre)
        if bbox_xyxy is None:
            bbox_xyxy = np.array([
                orig_w / 2, orig_h / 2, orig_w, orig_h
            ], dtype=np.float32)

        # Convert cx,cy,w,h → x_min,y_min,w,h (COCO format for albumentations)
        cx, cy, bw, bh = bbox_xyxy
        x_min = cx - bw / 2
        y_min = cy - bh / 2
        # Clamp
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        bw = min(bw, orig_w - x_min)
        bh = min(bh, orig_h - y_min)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=[[x_min, y_min, bw, bh]],
                bbox_labels=[label],
            )
            image = transformed['image']               # Tensor [3, H, W]
            bboxes = transformed['bboxes']
            if len(bboxes) > 0:
                xm, ym, tw, th = bboxes[0]
                # Convert back to cx,cy,w,h in the *transformed* pixel space
                bbox_out = torch.tensor(
                    [xm + tw / 2, ym + th / 2, tw, th], dtype=torch.float32
                )
            else:
                bbox_out = torch.tensor([IMAGE_SIZE/2, IMAGE_SIZE/2, IMAGE_SIZE, IMAGE_SIZE],
                                        dtype=torch.float32)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            bbox_out = torch.from_numpy(bbox_xyxy)

        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.task == "classification":
            return image, label_tensor

        elif self.task == "localization":
            return image, bbox_out

        elif self.task == "segmentation":
            mask_path = self.root / "annotations" / "trimaps" / f"{img_name}.png"
            mask = np.array(Image.open(mask_path))
            # Trimap: 1=foreground, 2=background, 3=not_classified → 0-indexed
            mask = (mask - 1).astype(np.int64)
            mask = torch.from_numpy(mask).long()
            # Resize mask to IMAGE_SIZE
            import torch.nn.functional as F
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(IMAGE_SIZE, IMAGE_SIZE), mode='nearest'
            ).squeeze().long()
            return image, mask

        else:
            raise ValueError(f"Unknown task: {self.task}")


# ── Convenience factory functions ────────────────────────────────────────────

def build_classification_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = OxfordPetsDataset(root, split='trainval', task='classification',
                                 transforms=get_train_transforms())
    val_ds   = OxfordPetsDataset(root, split='test',     task='classification',
                                 transforms=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


