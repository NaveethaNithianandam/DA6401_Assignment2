import sys
sys.path.insert(0, '.')

from data.pets_dataset import OxfordPetsDataset, get_loc_val_transforms
import numpy as np

DATA = '/home/lab/Downloads/Naveetha/DA6401_Asg2/Dataset_2'

for split in ['trainval', 'test']:
    ds = OxfordPetsDataset(DATA, split=split, task='localization',
                           transforms=get_loc_val_transforms())

    xml_count = sum(1 for s in ds.samples
                    if ds._load_bbox_xyxy(s['img_name']) is not None)

    print(f'\n{split}: {len(ds.samples)} samples, XML coverage: {xml_count} ({100*xml_count/len(ds.samples):.1f}%)')

    boxes = []
    for i in range(min(300, len(ds))):
        img, box = ds[i]
        boxes.append(box.numpy())

    boxes = np.array(boxes)

    print(f'  cx  mean={boxes[:,0].mean():.1f}  std={boxes[:,0].std():.1f}')
    print(f'  cy  mean={boxes[:,1].mean():.1f}  std={boxes[:,1].std():.1f}')
    print(f'  w   mean={boxes[:,2].mean():.1f}  std={boxes[:,2].std():.1f}  min={boxes[:,2].min():.1f}')
    print(f'  h   mean={boxes[:,3].mean():.1f}  std={boxes[:,3].std():.1f}  min={boxes[:,3].min():.1f}')

    full_img = np.sum((boxes[:,2] > 210) & (boxes[:,3] > 210))
    print(f'  full-image fallbacks (w,h>210): {full_img}/{len(boxes)}')