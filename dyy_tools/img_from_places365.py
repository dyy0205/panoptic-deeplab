import os, glob, shutil, cv2

# root = '/Users/dyy/Desktop/datasets/places365/'
# img_dir = os.path.join(root, 'val_large')
# save_dir = os.path.join(root, 'indoor2')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# with open(os.path.join(root, 'filelist_places365-standard/places365_val.txt'), 'r') as f:
#     data = f.readlines()
# for line in data:
#     filename, cate = line.strip().split()
#     if int(cate) in [
#         # 45, 52, 215, 121,
#         155, 203, 176, 177,
#     ]:
#         src = os.path.join(img_dir, filename)
#         dst = os.path.join(save_dir, filename)
#         shutil.copy(src, dst)


root = '/Users/dyy/Desktop/datasets/Indoor Scene Recognition/'
imgs = sorted(glob.glob(os.path.join(root, 'Images/*/*.jpg')))
for i, img in enumerate(imgs):
    name = img.split('/')[-1]
    cate = img.split('/')[-2]
    if cate in ['bedroom', 'dining_room', 'livingroom',
                'nursery', 'kitchen', 'children_room',
                'waitingroom', 'office', 'meeting_room'
                ]:
        image = cv2.imread(img)
        try:
            h, w, _ = image.shape
            if 300 <= min(h, w) < 400:
                save_dir = os.path.join(root, 'indoor2', cate)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                dst = os.path.join(save_dir, name)
                shutil.copy(img, dst)
        except:
            print(img)
            continue
