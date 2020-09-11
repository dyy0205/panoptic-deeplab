import os, glob, shutil
from PIL import Image
import numpy as np
import cv2


with open('mapFromADE.txt', 'r') as f:
    data = f.readlines()
CATE_TO_ADE = {}   # label 15,36,89 maps to two ADE categories
for line in data:
    semantic_id, ade_id = map(int, line.strip().split())
    CATE_TO_ADE.setdefault(semantic_id, []).append(ade_id)

LABEL_DICT = {
        1: [4],  # 29
        2: [1],
        3: [15, 59],
        4: [9, 64],
        5: [19],
        6: [23, 101],
        7: [28, 131, 142],
        8: [6],
        9: [140],
        10: [8],  # 40, 58, 82, 107, 132
        11: [16, 34, 57, 65],
        12: [11, 25, 36, 45, 63],
        13: [20, 31, 32, 70, 76, 98, 111],
        14: [24],
        15: [37, 83, 86, 135],
        16: [38, 46, 48, 50, 54, 66, 72, 74, 100, 118],
        17: [51, 75, 90, 108, 119, 125, 130, 144, 147],
        18: [13],
        19: [],
        20: [],
        21: [5, 18, 67, 73]
    }
CATE_DICT = {}
for key, values in LABEL_DICT.items():
    for v in values:
        CATE_DICT[v] = key


def label_convert(filename,
                  semantic_dir,
                  ade_dir,
                  save_dir):
    if filename.split('_')[1] == 'train':
        semantic_mask = np.array(Image.open(
            os.path.join(semantic_dir, 'training', filename + '.png')))
    else:
        semantic_mask = np.array(Image.open(
            os.path.join(semantic_dir, 'validation', filename + '.png')))
    h, w = semantic_mask.shape
    ade_mask = np.array(Image.open(os.path.join(ade_dir, filename + '_seg.png')).resize((w, h)))
    ade_classes = ade_mask[:, :, 0] / 10 * 256 + ade_mask[:, :, 1]
    ade_instance = ade_mask[:, :, 2]

    semantic_new = semantic_mask.copy()
    semantic_id_list = list(np.unique(semantic_mask)[1:])

    instance_mask = np.zeros((h, w), dtype=np.uint8)
    for semantic_id in sorted(semantic_id_list):
        ade_id = CATE_TO_ADE[semantic_id]
        if len(ade_id) == 1:
            instances = ade_instance[ade_classes == ade_id[0]]
        else:
            instances = ade_instance[(ade_classes == ade_id[0]) | (ade_classes == ade_id[1])]
        for ins_id in np.unique(instances):
            ins_mask = (ade_instance == ins_id).astype(np.uint8) * ins_id
            instance_mask += ins_mask

    for semantic_id in semantic_id_list:
        if semantic_id in CATE_DICT.keys():
            semantic_new[semantic_mask == semantic_id] = CATE_DICT[semantic_id]
        else:
            semantic_new[semantic_mask == semantic_id] = semantic_id

    new_semantic_id_list = list(np.unique(semantic_new)[1:])
    print('class id before merge:', new_semantic_id_list)

    for label in new_semantic_id_list:
        # TODO: 合并地板和地毯
        if label == 29:
            rugs = instance_mask[semantic_new == label]
            ins_mask = instance_mask.copy()
            for i in np.unique(rugs):
                rug_flag = False
                rug = (ins_mask == i).astype(np.uint8)
                rug_dilate = cv2.dilate(rug, kernel=(3, 3))
                if 1 in new_semantic_id_list:
                    floors = ins_mask[semantic_new == 1]
                    for ins_id in np.unique(floors):
                        floor = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(rug_dilate * floor) > 0:
                            semantic_new[ins_mask == i] = 1
                            instance_mask[ins_mask == i] = ins_id
                            floor = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(floor * 255).save(os.path.join(save_dir, 'mask_floor_{}.png'.format(ins_id)))
                            rug_flag = True
                            break
                if not rug_flag or 1 not in new_semantic_id_list:
                    semantic_new[ins_mask == i] = 0
                    instance_mask[ins_mask == i] = 0

        # TODO: 垫子归属床、凳子椅子还是沙发
        elif label in [40, 58, 82, 107, 132]:
            cushions = instance_mask[semantic_new == label]
            ins_mask = instance_mask.copy()
            for i in np.unique(cushions):
                cushion_flag = False
                cushion = (ins_mask == i).astype(np.uint8)
                cushion_dilate = cv2.dilate(cushion, kernel=(3, 3))
                if 10 in new_semantic_id_list:
                    beds = ins_mask[semantic_new == 10]
                    for ins_id in np.unique(beds):
                        bed = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(cushion_dilate * bed) > 0:
                            semantic_new[ins_mask == i] = 10
                            instance_mask[ins_mask == i] = ins_id
                            bed = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(bed * 255).save(os.path.join(save_dir, 'mask_bed_{}.png'.format(ins_id)))
                            cushion_flag = True
                            break
                if 13 in new_semantic_id_list:
                    chairs = ins_mask[semantic_new == 13]
                    for ins_id in np.unique(chairs):
                        chair = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(cushion_dilate * chair) > 0:
                            semantic_new[ins_mask == i] = 13
                            instance_mask[ins_mask == i] = ins_id
                            chair = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(chair * 255).save(os.path.join(save_dir, 'mask_chair_{}.png'.format(ins_id)))
                            cushion_flag = True
                            break
                if 14 in new_semantic_id_list:
                    sofas = ins_mask[semantic_new == 14]
                    for ins_id in np.unique(sofas):
                        sofa = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(cushion_dilate * sofa) > 0:
                            semantic_new[ins_mask == i] = 14
                            instance_mask[ins_mask == i] = ins_id
                            sofa = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(sofa * 255).save(os.path.join(save_dir, 'mask_sofa_{}.png'.format(ins_id)))
                            cushion_flag = True
                            break
                if not cushion_flag or (10 not in new_semantic_id_list and
                        13 not in new_semantic_id_list and 14 not in new_semantic_id_list):
                    semantic_new[ins_mask == i] = 0
                    instance_mask[ins_mask == i] = 0

        # TODO: 台面归属桌子还是柜子
        elif label == 71:
            countertops = instance_mask[semantic_new == label]
            ins_mask = instance_mask.copy()
            for i in np.unique(countertops):
                countertop_flag = False
                countertop = (ins_mask == i).astype(np.uint8)
                countertop_dilate = cv2.dilate(countertop, kernel=(3, 3))
                if 11 in new_semantic_id_list:
                    desks = ins_mask[semantic_new == 11]
                    for ins_id in np.unique(desks):
                        desk = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(countertop_dilate * desk) > 0:
                            semantic_new[ins_mask == i] = 11
                            instance_mask[ins_mask == i] = ins_id
                            desk = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(desk * 255).save(os.path.join(save_dir, 'mask_desk_{}.png'.format(ins_id)))
                            countertop_flag = True
                            break
                if 12 in new_semantic_id_list:
                    cabinets = ins_mask[semantic_new == 12]
                    for ins_id in np.unique(cabinets):
                        cabinet = (ins_mask == ins_id).astype(np.uint8)
                        if np.sum(countertop_dilate * cabinet) > 0:
                            semantic_new[ins_mask == i] = 12
                            instance_mask[ins_mask == i] = ins_id
                            cabinet = (instance_mask == ins_id).astype(np.uint8)
                            Image.fromarray(cabinet * 255).save(os.path.join(save_dir, 'mask_cabinet_{}.png'.format(ins_id)))
                            countertop_flag = True
                            break
                if not countertop_flag or (11 not in new_semantic_id_list and 12 not in new_semantic_id_list):
                    semantic_new[ins_mask == i] = 0
                    instance_mask[ins_mask == i] = 0

        # TODO: 书籍归属床、桌子、柜子、凳子椅子、沙发
        elif label == 68:
            books = instance_mask[semantic_new == label]
            ins_mask = instance_mask.copy()
            for i in np.unique(books):
                book_flag = False
                book = (ins_mask == i).astype(np.uint8)
                book_dilate = cv2.dilate(book, kernel=(3, 3))
                for cls in [10, 11, 12, 13, 14]:
                    if cls in new_semantic_id_list:
                        ms = ins_mask[semantic_new == cls]
                        for ins_id in np.unique(ms):
                            m = (ins_mask == ins_id).astype(np.uint8)
                            if np.sum(book_dilate * m) > 0 and not book_flag:
                                semantic_new[ins_mask == i] = cls
                                instance_mask[ins_mask == i] = ins_id
                                m = (instance_mask == ins_id).astype(np.uint8)
                                Image.fromarray(m * 255).save(os.path.join(save_dir, 'mask_book_{}.png'.format(ins_id)))
                                book_flag = True
                                break
                if not book_flag or (10 not in new_semantic_id_list and 11 not in new_semantic_id_list and
                        12 not in new_semantic_id_list and 13 not in new_semantic_id_list and
                        14 not in new_semantic_id_list):
                    semantic_new[ins_mask == i] = 0
                    instance_mask[ins_mask == i] = 0

        elif label not in LABEL_DICT.keys():
            mask = (semantic_new == label).astype(np.uint8) * 255
            Image.fromarray(mask).save(os.path.join(save_dir, '{}.png'.format(label)))
            semantic_new[semantic_new == label] = 0
            instance_mask[semantic_new == label] = 0

    print('class id after merge:', np.unique(semantic_new)[1:])
    assert max(np.unique(semantic_new)) < 22
    for label in np.unique(semantic_new)[1:]:
        mask = (semantic_new == label).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(save_dir, 'label_{}.png'.format(label)))

    final_mask = np.stack([semantic_new, instance_mask], axis=-1)
    Image.fromarray(final_mask).save(os.path.join(save_dir, 'final_mask.png'))


if __name__ == '__main__':
    root = '/Users/dyy/Desktop/ADE20k'
    semantic_dir = os.path.join(root, 'ADEChallengeData2016/annotations')
    ade_dir = os.path.join(root, 'raw_seg2')
    # if not os.path.exists(ade_dir):
    #     os.makedirs(ade_dir)

    with open('indoor_images2.txt', 'r') as f:
        images = f.readlines()
    # images = [name.strip() for name in images]
    images = ['ADE_val_00001372']

    # ade_masks = glob.glob('/Users/dyy/Desktop/datasets/ADE20k/ADE20K_2016_07_26/images/*/*/*/*_seg.png')
    # # ade_masks = glob.glob('/Users/dyy/Desktop/datasets/ADE20k/ADE20K_2016_07_26/images/*/*/*/*/*_seg.png')
    # for path in ade_masks:
    #     print(path)
    #     name = path.split('/')[-1]
    #     if name.replace('_seg.png', '') in images:
    #         shutil.copy(path, os.path.join(ade_dir, name))

    for i, filaname in enumerate(images):
        print(i, filaname)
        save_dir = os.path.join(root, 'indoor2', filaname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        label_convert(filaname, semantic_dir, ade_dir, save_dir)
        # try:
        #     label_convert(filaname, semantic_dir, ade_dir, save_dir)
        # except Exception as e:
        #     print(e)
