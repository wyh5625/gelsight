import numpy as np
import cv2
import random
from affine_transform import affine_transform
from marker_mask import *
from circle_mask import get_touch_mask_by_selection
from gradient import get_gradient


pixmm = 0.1  # 0.1mm/pixel
Rmm = 1.98  # ball radius
R = Rmm / pixmm
data_path = 'data/calibration7/'
meta_path = 'data/calibration7/ball_position/'
bg = cv2.imread(data_path+ 'bg-0.jpg')

data_list = []

ballfiles = [f for f in listdir(meta_path) if isfile(join(meta_path, f)) and "cal" in f and f.endswith('.txt')]

for ballfile in ballfiles:
    num = int(''.join(filter(str.isdigit, ballfile)))
    img = cv2.imread(data_path + 'cal-%d.jpg'%num)
    meta_file = meta_path + ballfile
    with open(meta_file, 'r+') as f:
        # line: x y radius
        line = f.readline()
        circle_data = line.split()
        x = int(circle_data[0])
        y = int(circle_data[1])
        radius = int(circle_data[2])

    dot_mask = get_marker_mask(affine_transform(img))
    _, touch_mask, center, _ = get_touch_mask_by_selection("circle_selector", affine_transform(bg), affine_transform(img),
                                                           int(R),(x, y), radius, visual=False)
    valid_mask = (touch_mask / 255) * (1 - dot_mask / 255)

    RGB, X, Y, GX, GY = get_gradient(affine_transform(img), center, R, valid_mask)

    # test the gradient data
    # shape of data: (,7)
    data = np.column_stack((RGB, X, Y, GX, GY))
    data_list.extend(data)


np.save('./data/cal_dataset7',np.array(data_list))