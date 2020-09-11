import os, shutil, glob
from PIL import Image
import numpy as np
import json
import pycocotools.mask as maskUtils

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.uint8):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

CATEGORIES = [
    {'supercategory': 'floor', 'id': 1, 'name': 'floor'},
    {'supercategory': 'wall', 'id': 2, 'name': 'wall'},
    {'supercategory': 'wall', 'id': 3, 'name': 'door'},
    {'supercategory': 'wall', 'id': 4, 'name': 'window'},
    {'supercategory': 'wall', 'id': 5, 'name': 'curtain'},
    {'supercategory': 'wall', 'id': 6, 'name': 'painting'},
    {'supercategory': 'wall', 'id': 7, 'name': 'wall_o'},
    {'supercategory': 'ceiling', 'id': 8, 'name': 'ceiling'},
    {'supercategory': 'ceiling', 'id': 9, 'name': 'fan'},
    {'supercategory': 'furniture', 'id': 10, 'name': 'bed'},
    {'supercategory': 'furniture', 'id': 11, 'name': 'desk'},
    {'supercategory': 'furniture', 'id': 12, 'name': 'cabinet'},
    {'supercategory': 'furniture', 'id': 13, 'name': 'chair'},
    {'supercategory': 'furniture', 'id': 14, 'name': 'sofa'},
    {'supercategory': 'furniture', 'id': 15, 'name': 'lamp'},
    {'supercategory': 'furniture', 'id': 16, 'name': 'furniture'},
    {'supercategory': 'electronics', 'id': 17, 'name': 'electronics'},
    {'supercategory': 'person', 'id': 18, 'name': 'person'},
    {'supercategory': 'cat', 'id': 19, 'name': 'cat'},
    {'supercategory': 'dog', 'id': 20, 'name': 'dog'},
    {'supercategory': 'plants', 'id': 21, 'name': 'plants'},
]


if __name__ == '__main__':
    root = '/Users/dyy/Desktop/ADE20k/'
    img_dir = os.path.join(root, 'ADEChallengeData2016/images')
    mask_dir = os.path.join(root, '2channels')

    # train, val = [], []
    # for name in os.listdir(mask_dir):
    #     if name.split('_')[1] == 'train':
    #         train.append(name)
    #     else:
    #         val.append(name)
    # print('train: ', len(train))
    # print('val: ', len(val))

    # for name in train:
    #     img_path = os.path.join(img_dir, 'training', name+'.jpg')
    #     dst = os.path.join(save_dir, 'train')
    #     if not os.path.exists(dst):
    #         os.makedirs(dst)
    #     shutil.copy(img_path, os.path.join(dst, name+'.jpg'))
    #
    # for name in val:
    #     img_path = os.path.join(img_dir, 'validation', name+'.jpg')
    #     dst = os.path.join(save_dir, 'val')
    #     if not os.path.exists(dst):
    #         os.makedirs(dst)
    #     shutil.copy(img_path, os.path.join(dst, name+'.jpg'))

    dataset_type = 'train'
    masks = glob.glob(os.path.join(mask_dir, dataset_type, '*.png'))
    json_out = {"images": [], "annotations": [], 'categories': CATEGORIES}
    image_id = 1
    segmentation_id = 1
    for i, mask_path in enumerate(masks):
        name = mask_path.split('/')[-1]
        mask = np.array(Image.open(mask_path))
        h, w, _ = mask.shape
        img_info = {
            "id": image_id,
            "file_name": name.replace('.png', '.jpg'),
            "width": w,
            "height": h,
        }
        json_out['images'].append(img_info)

        semantic = mask[:, :, 0]
        instance = mask[:, :, 1]
        for ins_id in np.unique(instance):
            if ins_id == 0:
                continue
            ins_mask = (instance == ins_id)
            cate_id = semantic[ins_mask][0]
            if cate_id in [0, 151]:
                continue

            binary_mask = maskUtils.encode(np.asfortranarray(ins_mask))
            area = maskUtils.area(binary_mask)
            bbox = maskUtils.toBbox(binary_mask)
            ann_info = {
                "id": segmentation_id,
                "image_id": image_id,
                "category_id": cate_id,
                "iscrowd": 0,
                "area": area.tolist(),
                "bbox": bbox.tolist(),
                "segmentation": binary_mask
            }
            json_out["annotations"].append(ann_info)
            segmentation_id += 1
        image_id += 1

    with open(os.path.join(mask_dir, 'instance_{}.json'.format(dataset_type)), 'w') as f:
        json.dump(json_out, f, cls=MyEncoder)
