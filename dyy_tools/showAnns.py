import os, shutil
import json
import skimage.io as io
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def showAnns(json_file, img_dir, mask_dir):
    coco = COCO(json_file)
    with open(json_file,'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    for i, img in enumerate(images):
        print(i, img['file_name'])
        img_id = img['id']
        I = io.imread(os.path.join(img_dir, img['file_name']))
        plt.figure()
        plt.clf()
        plt.axis('off')
        plt.imshow(I)
        anns = []
        for ann in annotations:
            if ann['image_id'] == img_id:
                anns.append(ann)
        coco.showAnns(anns)
        plt.savefig(os.path.join(mask_dir, img['file_name']), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    root = '/Users/dyy/Desktop/ADE20k/indoor_data'
    json_file = os.path.join(root, 'indoor_val.json')
    img_dir = os.path.join(root, 'val')
    mask_dir = os.path.join(root, 'show')
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir)
    showAnns(json_file, img_dir, mask_dir)
