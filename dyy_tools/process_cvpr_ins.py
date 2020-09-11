import os, glob, shutil
import numpy as np
from PIL import Image
from xlrd import open_workbook
import cv2
import json
import random
import pycocotools.mask as maskUtils

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.uint8):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

LABEL_DICT = {
    1: ["地板"],
    2: ["墙"],
    3: ["门"],
    4: ["窗户"],
    5: ["窗帘"],
    6: ["壁画"],
    7: ["墙面附属物"],
    8: ["天花板"],
    9: ["吊扇"],
    10: ["床"],
    11: ["桌子"],
    12: ["柜子"],
    13: ["椅子", "凳子"],
    14: ["沙发"],
    15: ["灯", "吊灯"],
    16: ["其他家具"],
    17: ["家电"],
    18: ["人物"],
    19: ["猫"],
    20: ["狗"],
    21: ["植物"]
}
CATE_DICT = {}
for k, v in LABEL_DICT.items():
    for k_, v_ in zip(len(v)*[k, ], v):
        CATE_DICT[v_] = k_


if __name__ == '__main__':
    root = '/Users/dyy/Desktop/cvpr13'
    img_dir = os.path.join(root, 'images')

    workbook = open_workbook(os.path.join(root, 'colors.xls'))
    sheet = workbook.sheet_by_index(0)
    nrows = sheet.nrows
    COLOR_TO_INS = {}
    for row in range(nrows):
        line = sheet.row_values(row)
        ins_name = line[0].strip()
        color = line[1].strip()[:-1]
        COLOR_TO_INS[color] = ins_name

    # imgs = glob.glob(os.path.join(root, 'mask/*.png'))
    # val = random.sample(imgs, 49)
    # train = set(imgs) - set(val)
    # with open(os.path.join(root, 'train.txt'), 'w') as f:
    #     for path in train:
    #         f.writelines(path)
    #         f.write('\n')
    # with open(os.path.join(root, 'val.txt'), 'w') as f:
    #     for path in val:
    #         f.writelines(path)
    #         f.write('\n')

    dataset_type = 'train'
    with open(os.path.join(root, '{}.txt'.format(dataset_type)), 'r') as f:
        imgs = f.readlines()
    imgs = [img.strip() for img in imgs]

    save_dir = os.path.join(root, 'data', dataset_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    json_out = {"images": [], "annotations": []}
    image_id = 1
    segmentation_id = 1
    for idx, img in enumerate(imgs):
        name = img.split('/')[-1].replace('.png', '.jpg')
        print(idx, name)
        img = np.array(Image.open(img))
        h, w, _ = img.shape
        img_info = {
            "id": image_id,
            "file_name": name,
            "width": w,
            "height": h,
        }
        json_out['images'].append(img_info)
        shutil.copy(os.path.join(img_dir, name),
                    os.path.join(save_dir, name))

        colors = []
        for i in range(h):
            for j in range(w):
                c = tuple(img[i, j, :])
                if c not in colors:
                    colors.append(c)
        for color in colors:
            key = ''.join(str(color).split(' '))
            ins_name = COLOR_TO_INS[key]
            for k, v in CATE_DICT.items():
                if ins_name.startswith(k):
                    cate_id = v
            ins_mask = cv2.inRange(img, lowerb=np.array(color), upperb=np.array(color))
            ins_mask = (ins_mask / 255.).astype(np.uint8)
            if ins_name == '植物' and ins_mask.sum() < 1000:
                continue
                # print(name, ins_mask.sum())
                # save_path = os.path.join(save_dir, name)
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                # cv2.imwrite(os.path.join(save_path, '{}.png'.format(ins_name)), ins_mask*255)

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

    with open(os.path.join(root, 'data/{}.json'.format(dataset_type)), 'w') as f:
        json.dump(json_out, f, cls=MyEncoder)