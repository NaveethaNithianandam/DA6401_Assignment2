"""Oxford-IIIT Pet dataset loader for all three tasks.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224   # fixed per VGG11 paper


# =============================================================================
#  Classification transforms  (spatial augmentation is fine here)
# =============================================================================

def get_cls_train_transforms() -> A.Compose:
    """Aggressive spatial + colour augmentation for classification.

    RandomResizedCrop is safe for classification because the label does not
    depend on where in the image the crop is taken.
    """
    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE,
                            scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.6),
        A.Rotate(limit=15, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_cls_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =============================================================================
#  Localization transforms  (NO spatial crops / rotations)
#
#  Root cause of the original train/val mismatch:
#  RandomResizedCrop shifts bbox coordinates relative to the random crop
#  window.  At val time (simple Resize) the coordinates follow a completely
#  different distribution, causing val_MSE >> train_MSE (~4800 vs ~200).
#
#  Fix: use only colour augmentation + HorizontalFlip (which albumentations
#  tracks correctly through bbox_params) during localization training.
#  Spatial crops/rotates are removed entirely.
# =============================================================================

def get_loc_train_transforms() -> A.Compose:
    """Colour-only augmentation for localization training.

    Spatial augmentation is intentionally omitted:
      - RandomResizedCrop shifts bbox coords relative to the random window,
        creating a train/val distribution mismatch of ~656x in MSE magnitude.
      - Rotate requires bbox re-projection that is brittle with small bboxes.
      - HorizontalFlip is safe because albumentations tracks it correctly via
        bbox_params (it mirrors cx -> IMAGE_SIZE - cx).
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels'],
                                min_visibility=0.3))


def get_loc_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['bbox_labels'],
                                min_visibility=0.3))


# =============================================================================
#  Segmentation transforms
# =============================================================================

def get_seg_train_transforms() -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE,
                            scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_seg_val_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =============================================================================
#  Dataset
# =============================================================================

class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet Dataset.

    root/
      images/           JPEG images (<breed>_<idx>.jpg)
      annotations/
        trainval.txt    split file
        test.txt        split file
        trimaps/        PNG segmentation masks (1=fg, 2=bg, 3=uncertain)
        xmls/           Pascal VOC bounding-box XML files

    For localization, bbox coordinates are returned in pixel space
    (x_center, y_center, width, height) relative to the 224x224 image,
    matching the assignment requirement exactly.
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        task: str = "classification",
        transforms=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.task = task
        self.transforms = transforms
        self.samples = self._load_split()

    def _load_split(self) -> List[dict]:
        split_file = self.root / "annotations" / f"{self.split}.txt"
        samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                samples.append({'img_name': parts[0], 'label': int(parts[1]) - 1})
        return samples

    def _load_bbox_xyxy(self, img_name: str):
        """Return (xmin, ymin, xmax, ymax) in original pixel coords, or None."""
        import xml.etree.ElementTree as ET
        xml_path = self.root / "annotations" / "xmls" / f"{img_name}.xml"
        if not xml_path.exists():
            return None
        root = ET.parse(xml_path).getroot()
        obj = root.find('object')
        if obj is None:
            return None
        bb = obj.find('bndbox')
        return (float(bb.find('xmin').text), float(bb.find('ymin').text),
                float(bb.find('xmax').text), float(bb.find('ymax').text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_name = s['img_name']
        label    = s['label']

        image = np.array(Image.open(
            self.root / "images" / f"{img_name}.jpg").convert('RGB'))
        orig_h, orig_w = image.shape[:2]

        # ── Classification ───────────────────────────────────────────────────
        if self.task == "classification":
            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return image, torch.tensor(label, dtype=torch.long)

        # ── Localization ─────────────────────────────────────────────────────
        elif self.task == "localization":
            bbox = self._load_bbox_xyxy(img_name)
            if bbox is None:
                # Fallback: full image as bbox
                xmin, ymin, xmax, ymax = 0.0, 0.0, float(orig_w), float(orig_h)
            else:
                xmin, ymin, xmax, ymax = bbox

            # Clamp to image bounds
            xmin = max(0.0, min(xmin, orig_w - 1))
            ymin = max(0.0, min(ymin, orig_h - 1))
            xmax = max(xmin + 1, min(xmax, orig_w))
            ymax = max(ymin + 1, min(ymax, orig_h))
            bw = xmax - xmin
            bh = ymax - ymin

            if self.transforms is not None:
                out = self.transforms(
                    image=image,
                    bboxes=[[xmin, ymin, bw, bh]],  # COCO: x_min, y_min, w, h
                    bbox_labels=[label],
                )
                image = out['image']
                bboxes = out['bboxes']
                if len(bboxes) > 0:
                    xm, ym, tw, th = bboxes[0]
                    # Convert COCO back to cx, cy, w, h (pixel space, 224x224)
                    bbox_out = torch.tensor(
                        [xm + tw / 2, ym + th / 2, tw, th], dtype=torch.float32)
                else:
                    # bbox was cropped out -- use image centre as fallback
                    bbox_out = torch.tensor(
                        [IMAGE_SIZE / 2, IMAGE_SIZE / 2,
                         float(IMAGE_SIZE), float(IMAGE_SIZE)], dtype=torch.float32)
            else:
                # Scale bbox to 224x224
                sx = IMAGE_SIZE / orig_w
                sy = IMAGE_SIZE / orig_h
                cx = (xmin + xmax) / 2 * sx
                cy = (ymin + ymax) / 2 * sy
                tw = bw * sx
                th = bh * sy
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                bbox_out = torch.tensor([cx, cy, tw, th], dtype=torch.float32)

            return image, bbox_out

        # ── Segmentation ─────────────────────────────────────────────────────
        elif self.task == "segmentation":
            mask_path = self.root / "annotations" / "trimaps" / f"{img_name}.png"
            mask = np.array(Image.open(mask_path))
            mask = (mask - 1).astype(np.int64)  # 0=fg, 1=bg, 2=uncertain

            if self.transforms is not None:
                image = self.transforms(image=image)['image']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            import torch.nn.functional as F
            mask = torch.from_numpy(mask).long()
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(IMAGE_SIZE, IMAGE_SIZE), mode='nearest').squeeze().long()
            return image, mask

        else:
            raise ValueError(f"Unknown task: {self.task}")


# =============================================================================
#  DataLoader factories
# =============================================================================

def build_classification_loaders(
    root: str, batch_size: int = 32,
    num_workers: int = 4, pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = OxfordPetsDataset(root, 'trainval', 'classification',
                                 get_cls_train_transforms())
    val_ds   = OxfordPetsDataset(root, 'test',     'classification',
                                 get_cls_val_transforms())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )


def build_localization_loaders(
    root: str, batch_size: int = 32,
    num_workers: int = 4, pin_memory: bool = True,
    val_fraction: float = 0.15, seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build localization loaders by splitting trainval internally.

    The official Oxford Pets test split has NO bounding box XML annotations
    (XMLs are only provided for trainval).  Using test as val produces
    val_mIoU=0.22 because every val sample falls back to the full-image box
    (cx=112, cy=112, w=224, h=224) -- a meaningless target.

    Fix: split the 3680 trainval samples (99.8% XML coverage) into
    train (85%) and val (15%) using a fixed random seed for reproducibility.
    Both subsets have real tight bounding boxes -> meaningful IoU evaluation.
    """
    import random
    full_ds = OxfordPetsDataset(root, 'trainval', 'localization', transforms=None)
    all_samples = full_ds.samples[:]

    # Filter to only samples that have XML annotations
    xml_samples = [s for s in all_samples
                   if full_ds._load_bbox_xyxy(s['img_name']) is not None]
    print(f"  Localization: {len(xml_samples)}/{len(all_samples)} samples have XML annotations")

    # Deterministic shuffle + split
    rng = random.Random(seed)
    rng.shuffle(xml_samples)
    n_val   = max(1, int(len(xml_samples) * val_fraction))
    n_train = len(xml_samples) - n_val

    train_samples = xml_samples[:n_train]
    val_samples   = xml_samples[n_train:]
    print(f"  Split: {len(train_samples)} train / {len(val_samples)} val")

    # Build datasets with correct transforms, injecting pre-filtered sample lists
    train_ds = OxfordPetsDataset(root, 'trainval', 'localization',
                                 get_loc_train_transforms())
    val_ds   = OxfordPetsDataset(root, 'trainval', 'localization',
                                 get_loc_val_transforms())
    train_ds.samples = train_samples
    val_ds.samples   = val_samples

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )


def build_segmentation_loaders(
    root: str, batch_size: int = 16,
    num_workers: int = 4, pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = OxfordPetsDataset(root, 'trainval', 'segmentation',
                                 get_seg_train_transforms())
    val_ds   = OxfordPetsDataset(root, 'test',     'segmentation',
                                 get_seg_val_transforms())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )