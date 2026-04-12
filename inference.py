"""Inference and evaluation for the multi-task perception model.

Usage:
    python inference.py --image /path/to/image.jpg
    python inference.py --image /path/to/image.jpg --visualize
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from multitask import MultiTaskPerceptionModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224


BREED_NAMES = [
    'Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
    'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
    'Siamese','Sphynx','american_bulldog','american_pit_bull_terrier',
    'basset_hound','beagle','boxer','chihuahua','english_cocker_spaniel',
    'english_setter','german_shorthaired','great_pyrenees','havanese',
    'japanese_chin','keeshond','leonberger','miniature_pinscher',
    'newfoundland','pomeranian','pug','saint_bernard','samoyed',
    'scottish_terrier','shiba_inu','staffordshire_bull_terrier',
    'wheaten_terrier','yorkshire_terrier',
]

SEG_CLASSES = {0: 'foreground', 1: 'background', 2: 'boundary'}


def preprocess(image_path: str) -> torch.Tensor:
    """Load and normalise an image to [1, 3, 224, 224]."""
    image = np.array(Image.open(image_path).convert('RGB'))
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    tensor = transform(image=image)['image']   
    return tensor.unsqueeze(0)                


def run_inference(image_path: str, device: torch.device = None):
    """Run the multi-task model on a single image.

    Returns:
        dict with keys:
            'breed'       : predicted breed name (str)
            'breed_idx'   : predicted class index (int)
            'confidence'  : softmax probability of top class (float)
            'bbox'        : [cx, cy, w, h] in pixel space (list of 4 floats)
            'seg_mask'    : [H, W] numpy array of class indices (0/1/2)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskPerceptionModel().to(device)
    model.eval()

    image_tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    logits     = outputs['classification'][0]         
    probs      = torch.softmax(logits, dim=0)
    breed_idx  = probs.argmax().item()
    confidence = probs[breed_idx].item()
    breed_name = BREED_NAMES[breed_idx] if breed_idx < len(BREED_NAMES) else str(breed_idx)

   
    bbox = outputs['localization'][0].cpu().tolist()  

    seg_logits = outputs['segmentation'][0]            
    seg_mask   = seg_logits.argmax(dim=0).cpu().numpy() 

    return {
        'breed':      breed_name,
        'breed_idx':  breed_idx,
        'confidence': confidence,
        'bbox':       bbox,
        'seg_mask':   seg_mask,
    }


def visualize(image_path: str, results: dict):
    """Draw predictions on the image and display with matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available -- skipping visualization")
        return

    image = np.array(Image.open(image_path).convert('RGB'))
    orig_h, orig_w = image.shape[:2]

    sx = orig_w / IMAGE_SIZE
    sy = orig_h / IMAGE_SIZE
    cx, cy, w, h = results['bbox']
    cx *= sx; cy *= sy; w *= sx; h *= sy
    x1, y1 = cx - w / 2, cy - h / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    rect = patches.Rectangle((x1, y1), w, h,
                               linewidth=2, edgecolor='lime', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f"{results['breed']}  ({results['confidence']*100:.1f}%)",
                      fontsize=11)
    axes[0].axis('off')

    seg = results['seg_mask']
    seg_rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
    seg_rgb[seg == 0] = [255, 150,  50]   
    seg_rgb[seg == 1] = [ 50, 100, 200]  
    seg_rgb[seg == 2] = [200, 200, 200]   
    axes[1].imshow(seg_rgb)
    axes[1].set_title('segmentation mask', fontsize=11)
    axes[1].axis('off')

    seg_resized = Image.fromarray(seg_rgb).resize((orig_w, orig_h))
    blended = (np.array(image) * 0.55 + np.array(seg_resized) * 0.45).astype(np.uint8)
    axes[2].imshow(blended)
    axes[2].set_title('overlay', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('inference_output.png', dpi=120, bbox_inches='tight')
    print("Saved visualization -> inference_output.png")
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Multi-task inference")
    p.add_argument("--image",     required=True, help="Path to input image")
    p.add_argument("--visualize", action="store_true",
                   help="Draw and save bounding box + segmentation overlay")
    args = p.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    results = run_inference(args.image)

    print(f"\nClassification : {results['breed']}  (class {results['breed_idx']}, "
          f"confidence {results['confidence']*100:.1f}%)")
    cx, cy, w, h = results['bbox']
    print(f"Bounding box   : cx={cx:.1f}  cy={cy:.1f}  w={w:.1f}  h={h:.1f}  (pixels, 224x224)")
    unique, counts = np.unique(results['seg_mask'], return_counts=True)
    total = results['seg_mask'].size
    print("Segmentation   :", end="")
    for cls, cnt in zip(unique, counts):
        print(f"  {SEG_CLASSES.get(cls, cls)}={cnt/total*100:.1f}%", end="")
    print()

    if args.visualize:
        visualize(args.image, results)


if __name__ == "__main__":
    main()