import os

with open('sceneCategories.txt', 'r') as f:
    data = f.readlines()

scenes = []
images = []
for line in data:
    filename, scene = line.strip().split(' ')
    # print(filename, scene)
    if scene in [
        # 'bedroom', 'living_room', 'dining_room',
        'bathroom', 'art_gallery', 'ballroom', 'breakroom', 'childs_room',
        'classroom', 'clean_room', 'computer_room', 'conference_room',
        'office', 'dinette_home', 'dorm_room', 'game_room', 'home_office',
        'home_theater', 'hotel_room', 'kitchen', 'playroom', 'waiting_room'
        ]:
        images.append(filename)
    # if scene not in scenes:
    #     scenes.append(scene)
# with open('scenes.txt', 'w') as f:
#     for scene in scenes:
#         f.writelines(scene)
#         f.write('\n')
with open('indoor_images2.txt', 'w') as f:
    for image in images:
        f.writelines(image)
        f.write('\n')
