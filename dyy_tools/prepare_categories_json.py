import json

CATEGORIES = [
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

with open('ADE_indoor_categories.json', 'w') as f:
    json.dump(CATEGORIES, f)
