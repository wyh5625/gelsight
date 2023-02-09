import numpy as np
import cv2
import random
from affine_transform import affine_transform
from marker_mask import get_marker_mask_by_blob
from circle_mask import get_touch_mask_by_selection

# [r,g,b,x,y] -> [gx. gy]
# if radius of ball is R, normal vector(inward) to its surface is gx = (-x)/(R^2-x^2-y^2)^(1/2), gy = (-y)/(R^2-x^2-y^2)^(1/2), gz = -1
# Input: img -> the contact image
#        ref -> the reference image
#        center -> center of contacted ball in the img
#        R -> radius of ball in pixel
#        mask -> mask of selected circle region with markers removed
# Output: RGB,X,Y,GX,GY
def get_gradient(img, center, R, mask):


    # # rgb of diff image in valid pixel
    # blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
    # img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    # diff = img_smooth - blur


    # calculate the gx and gy
    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    xv = x - center[0]
    yv = y - center[1]

    gx = (-xv) / np.sqrt(R ** 2 - (xv*mask) ** 2 - (yv*mask) ** 2)
    gy = (-yv) / np.sqrt(R ** 2 - (xv*mask) ** 2 - (yv*mask) ** 2)

    # if distance from the point to the center of the circle is greater than the radius of calibrated ball,
    # the gx or gy value would become Nan value, which should be ignored

    mask[np.isnan(gx)] = 0
    mask[np.isnan(gy)] = 0

    # print(mask.shape)
    RGB = img[mask>0]

    X = x[mask>0]
    Y = y[mask>0]
    GX = gx[mask>0]
    GY = gy[mask>0]

    return RGB,X,Y,GX,GY


def visualize_gradient_xy(mask, center, data):
    vec_len = 15

    canvas = cv2.circle(np.array(mask), (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)
    canvas = cv2.merge([canvas, canvas, canvas])

    while (1):
        # get a random data in data list
        a_data = data[random.randint(0, len(data) - 1)]

        # normal vector is not necessary to be with length 1
        # print("Length of normal vector: " + str(np.sqrt(a_data[5]**2 + a_data[6]**2 + 1)))

        back_img = canvas.copy()

        xy_v = np.array([-a_data[5], -a_data[6]])
        xy_v = xy_v / np.linalg.norm(xy_v)
        end_p = (a_data[3], a_data[4]) + vec_len * xy_v
        cv2.arrowedLine(back_img, (int(a_data[3]), int(a_data[4])), (int(end_p[0]), int(end_p[1])), (0, 255, 0), 2)

        cv2.imshow('contact mask', back_img)

        k = cv2.waitKey(0)
        if k == 27:
            break


if __name__ == "__main__":
    pixmm = 0.1  # 0.1mm/pixel
    Rmm = 2.42  # ball radius
    R = Rmm / pixmm

    img = cv2.imread('data/calibration/ball-10.jpg')
    bg = cv2.imread('data/calibration/bg-0.jpg')

    dot_mask = get_marker_mask_by_blob(affine_transform(img))
    _, touch_mask, center, _ = get_touch_mask_by_selection("circle_selector", affine_transform(bg), affine_transform(img), int(R))
    valid_mask = (touch_mask/255)*(1-dot_mask/255)

    RGB,X,Y,GX,GY = get_gradient(affine_transform(img), affine_transform(bg), center, R, valid_mask)

    # test the gradient data
    # shape of data: (,7)
    data = np.column_stack((RGB, X, Y, GX, GY))

    visualize_gradient_xy(valid_mask, center, data)