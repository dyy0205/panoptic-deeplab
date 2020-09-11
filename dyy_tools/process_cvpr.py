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
    # imgs = glob.glob('/Users/dyy/Desktop/indoor/bedroom/*.png')
    # for img in imgs:
    #     name = img.split('/')[-1].replace('图', 'bedroom')
    #     os.rename(img, img.replace(img.split('/')[-1], name))

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

    dataset_type = 'val'
    with open(os.path.join(root, '{}.txt'.format(dataset_type)), 'r') as f:
        imgs = f.readlines()
    imgs = [img.strip() for img in imgs]

    save_dir = os.path.join(root, '2channel', dataset_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, img in enumerate(imgs):
        name = img.split('/')[-1][:-4]
        print(idx, name)
        img = np.array(Image.open(img))
        h, w, _ = img.shape
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        colors = []
        for i in range(h):
            for j in range(w):
                c = tuple(img[i, j, :])
                if c not in colors:
                    colors.append(c)

        instance_id = 1
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
            semantic_mask[ins_mask == 1] = cate_id
            instance_mask[ins_mask == 1] = instance_id * 10
            instance_id += 1

        final_mask = np.stack([semantic_mask, instance_mask], axis=-1)
        Image.fromarray(final_mask).save(os.path.join(save_dir, '{}.png'.format(name)))