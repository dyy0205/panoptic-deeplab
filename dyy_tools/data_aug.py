import json

type = 'val'

with open('/Users/dyy/Desktop/ADE20k/2channels/instance_{}.json'.format(type)) as f:
    data = json.load(f)
images = data['images']
annotations = data['annotations']
categories = data['categories']
num_imgs = max([img['id'] for img in images])
num_anns = max([ann['id'] for ann in annotations])
print('before: images {}, annotations {}'.format(num_imgs, num_anns))


with open('/Users/dyy/Desktop/cvpr13/data/{}.json'.format(type)) as f:
    cvpr = json.load(f)
cvpr_imgs = cvpr['images']
cvpr_anns = cvpr['annotations']
for i, img in enumerate(cvpr_imgs):
    cvpr_imgs[i]['file_name'] = 'cvpr_' + cvpr_imgs[i]['file_name']
    cvpr_imgs[i]['id'] = img['id'] + num_imgs
for i, ann in enumerate(cvpr_anns):
    cvpr_anns[i]['image_id'] = ann['image_id'] + num_imgs
    cvpr_anns[i]['id'] = ann['id'] + num_anns

images.extend(cvpr_imgs)
annotations.extend(cvpr_anns)
num_imgs = max([img['id'] for img in images])
num_anns = max([ann['id'] for ann in annotations])
print('after: images {}, annotations {}'.format(num_imgs, num_anns))

aug_json = {"images": images, "annotations": annotations, "categories": categories}
with open('/Users/dyy/Desktop/{}.json'.format(type), 'w') as f:
    json.dump(aug_json, f)