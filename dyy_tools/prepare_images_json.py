import os, shutil, glob
import json
from PIL import Image
import numpy as np

root = '/Users/dyy/Desktop/ADE20k/'
mask_dir = os.path.join(root, 'indoor2')
save_dir = os.path.join(root, '2channels')
for set in ['train', 'val']:
    set_dir = os.path.join(save_dir, set)
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)

train_images = []
val_images = []
train_ids = 1
val_ids = 1
for name in sorted(os.listdir(mask_dir)):
    mask_path = os.path.join(mask_dir, name, 'final_mask.png')
    mask = Image.open(mask_path)
    w, h = mask.size
    if name.split('_')[1] == 'train':
        img_info = {
            'file_name': name + '.jpg',
            'id': train_ids,
            'height': h,
            'width': w
        }
        train_images.append(img_info)
        train_ids += 1
        dst = os.path.join(save_dir, 'train', name + '.png')
    else:
        img_info = {
            'file_name': name + '.jpg',
            'id': val_ids,
            'height': h,
            'width': w
        }
        val_images.append(img_info)
        val_ids += 1
        dst = os.path.join(save_dir, 'val', name+'.png')
    shutil.copy(mask_path, dst)

with open(os.path.join(save_dir, 'ADE_indoor_images_train.json'), 'w') as f:
    json.dump({'images': train_images}, f)
with open(os.path.join(save_dir, 'ADE_indoor_images_val.json'), 'w') as f:
    json.dump({'images': val_images}, f)

# masks = glob.glob(os.path.join(save_dir, 'train/*.png'))
# for mask_path in masks:
#     mask = np.array(Image.open(mask_path))
#     semantic_mask = mask[:, :, 0]
#     instance_mask = mask[:, :, 1]
#     semantic_mask[(semantic_mask == 40) | (semantic_mask == 151)] = 0
#     instance_mask[(semantic_mask == 40) | (semantic_mask == 151)] = 0
#     mask = np.stack([semantic_mask, instance_mask], axis=-1)
#     assert 40 not in np.unique(semantic_mask)
#     assert 151 not in np.unique(semantic_mask)
#     Image.fromarray(mask).save(mask_path)

