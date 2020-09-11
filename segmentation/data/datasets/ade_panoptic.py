# ------------------------------------------------------------------------------
# Loads COCO panoptic dataset.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import json
import os

import numpy as np

from .base_dataset import BaseDataset
from .utils import DatasetDescriptor
from ..transforms import build_transforms, Resize, PanopticTargetGenerator, SemanticTargetGenerator

_COCO_PANOPTIC_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 4989,
                     'val': 502},
    num_classes=21,
    ignore_label=255,
)

# Add 1 void label.
_COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID = (
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

_COCO_PANOPTIC_EVAL_ID_TO_TRAIN_ID = {
    v: k for k, v in enumerate(_COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID)
}

_COCO_PANOPTIC_THING_LIST = list(range(21))

ADE_CATEGORIES = [
    {'supercategory': 'floor', 'id': 1, 'name': 'floor', "isthing": 1, "color": [128, 0, 0]},
    {'supercategory': 'wall', 'id': 2, 'name': 'wall', "isthing": 1, "color": [128, 128, 0]},
    {'supercategory': 'wall', 'id': 3, 'name': 'door', "isthing": 1, "color": [0, 0, 210]},
    {'supercategory': 'wall', 'id': 4, 'name': 'window', "isthing": 1, "color": [125, 125, 125]},
    {'supercategory': 'wall', 'id': 5, 'name': 'curtain', "isthing": 1, "color": [64, 0, 0]},
    {'supercategory': 'wall', 'id': 6, 'name': 'painting', "isthing": 1, "color": [64, 128, 0]},
    {'supercategory': 'wall', 'id': 7, 'name': 'wall_o', "isthing": 1, "color": [64, 0, 128]},
    {'supercategory': 'ceiling', 'id': 8, 'name': 'ceiling', "isthing": 1, "color": [192, 0, 128]},
    {'supercategory': 'ceiling', 'id': 9, 'name': 'fan', "isthing": 1, "color": [114, 128, 128]},
    {'supercategory': 'furniture', 'id': 10, 'name': 'bed', "isthing": 1, "color": [192, 128, 128]},
    {'supercategory': 'furniture', 'id': 11, 'name': 'desk', "isthing": 1, "color": [128, 64, 0]},
    {'supercategory': 'furniture', 'id': 12, 'name': 'cabinet', "isthing": 1, "color": [0, 128, 0]},
    {'supercategory': 'furniture', 'id': 13, 'name': 'chair', "isthing": 1, "color": [0, 0, 128]},
    {'supercategory': 'furniture', 'id': 14, 'name': 'sofa', "isthing": 1, "color": [255, 128, 100]},
    {'supercategory': 'furniture', 'id': 15, 'name': 'lamp', "isthing": 1, "color": [192, 0, 0]},
    {'supercategory': 'furniture', 'id': 16, 'name': 'furniture', "isthing": 1, "color": [192, 128, 0]},
    {'supercategory': 'electronics', 'id': 17, 'name': 'electronics', "isthing": 1, "color": [0, 64, 0]},
    {'supercategory': 'person', 'id': 18, 'name': 'person', "isthing": 1, "color": [0, 192, 0]},
    {'supercategory': 'cat', 'id': 19, 'name': 'cat', "isthing": 1, "color": [0, 128, 255]},
    {'supercategory': 'dog', 'id': 20, 'name': 'dog', "isthing": 1, "color": [100, 192, 0]},
    {'supercategory': 'plants', 'id': 21, 'name': 'plants', "isthing": 1, "color": [0, 255, 0]},
]


class ADEPanoptic(BaseDataset):
    """
    COCO panoptic segmentation dataset.
    Arguments:
        root: Str, root directory.
        split: Str, data split, e.g. train/val/test.
        is_train: Bool, for training or testing.
        crop_size: Tuple, crop size.
        mirror: Bool, whether to apply random horizontal flip.
        min_scale: Float, min scale in scale augmentation.
        max_scale: Float, max scale in scale augmentation.
        scale_step_size: Float, step size to select random scale.
        mean: Tuple, image mean.
        std: Tuple, image std.
        semantic_only: Bool, only use semantic segmentation label.
        ignore_stuff_in_offset: Boolean, whether to ignore stuff region when training the offset branch.
        small_instance_area: Integer, indicates largest area for small instances.
        small_instance_weight: Integer, indicates semantic loss weights for small instances.
    """
    def __init__(self,
                 root,
                 split,
                 min_resize_value=641,
                 max_resize_value=641,
                 resize_factor=32,
                 is_train=True,
                 crop_size=(641, 641),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 **kwargs):
        super(ADEPanoptic, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale,
                                           scale_step_size, mean, std)

        assert split in _COCO_PANOPTIC_INFORMATION.splits_to_sizes.keys()

        self.num_classes = _COCO_PANOPTIC_INFORMATION.num_classes
        self.ignore_label = _COCO_PANOPTIC_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.has_instance = True
        self.label_divisor = 256
        self.label_dtype = np.float32
        self.thing_list = _COCO_PANOPTIC_THING_LIST

        # Get image and annotation list.
        if 'test' in split:
            self.img_list = []
            self.ann_list = None
            self.ins_list = None
            json_filename = os.path.join(self.root, 'annotations', 'image_info_{}.json'.format(self.split))
            dataset = json.load(open(json_filename))
            for img in dataset['images']:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(self.root, 'test2017', img_file_name))
        else:
            self.img_list = []
            self.ann_list = []
            self.ins_list = []
            json_filename = os.path.join(self.root, 'annotations', 'ADE_indoor_{}_panoptic.json'.format(self.split))
            dataset = json.load(open(json_filename))
            # First sort by image id.
            images = sorted(dataset['images'], key=lambda i: i['id'])
            annotations = sorted(dataset['annotations'], key=lambda i: i['image_id'])
            for img in images:
                img_file_name = img['file_name']
                self.img_list.append(os.path.join(self.root, self.split, img_file_name))
            for ann in annotations:
                ann_file_name = ann['file_name']
                self.ann_list.append(os.path.join(
                    self.root, 'annotations', 'ADE_indoor_{}'.format(self.split), ann_file_name))
                self.ins_list.append(ann['segments_info'])

        assert len(self) == _COCO_PANOPTIC_INFORMATION.splits_to_sizes[self.split]

        self.pre_augmentation_transform = Resize(min_resize_value, max_resize_value, resize_factor)
        self.transform = build_transforms(self, is_train)
        if semantic_only:
            self.target_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)
        else:
            self.target_transform = PanopticTargetGenerator(self.ignore_label, self.rgb2id, _COCO_PANOPTIC_THING_LIST,
                                                            sigma=8, ignore_stuff_in_offset=ignore_stuff_in_offset,
                                                            small_instance_area=small_instance_area,
                                                            small_instance_weight=small_instance_weight)
        # Generates semantic label for evaluation.
        self.raw_label_transform = SemanticTargetGenerator(self.ignore_label, self.rgb2id)

    @staticmethod
    def train_id_to_eval_id():
        return _COCO_PANOPTIC_TRAIN_ID_TO_EVAL_ID

    @staticmethod
    def rgb2id(color):
        """Converts the color to panoptic label.
        Color is created by `color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]`.
        Args:
            color: Ndarray or a tuple, color encoded image.
        Returns:
            Panoptic label.
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    @staticmethod
    def create_label_colormap():
        """Creates a label colormap used in COCO panoptic benchmark.
        Returns:
            A colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i, color in enumerate(ADE_CATEGORIES):
            colormap[i] = color['color']
        return colormap
