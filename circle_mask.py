import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join, exists
from affine_transform import affine_transform


def get_touch_mask(img_touch, center, radius):
    touch_mask = np.zeros((img_touch.shape[0], img_touch.shape[1]), dtype=int)
    touch_mask = cv2.circle(np.array(touch_mask), (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)

    return touch_mask

def get_touch_mask_by_selection(img_name,img_bg, img_touch, ball_radius_p, circle_center=None, circle_radius=None, visual = True):

    if circle_center == None or circle_radius == None:
        blur = cv2.GaussianBlur(img_bg.astype(np.float32), (3, 3), 0)
        # cv2.imshow('blur',blur.astype(np.uint8))
        # cv2.waitKey(0)
        diff_img = np.abs(np.sum(blur - np.abs(img_touch.astype(np.float32)),axis = 2)) #shape = m x n
        # cv2.imshow('diff_img',diff_img.astype(np.uint8))
        # cv2.waitKey(0)
        contact_mask = (diff_img > 10).astype(np.uint8)*255
        # cv2.imshow('contact_mask',contact_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour

        (x, y), radius = cv2.minEnclosingCircle(cnt)
    else:
        x = circle_center[0]
        y = circle_center[1]
        radius = circle_radius

    key = -1
    if visual:

        while key != 32 and key!=27:
            center = (int(x), int(y))
            radius = min(int(radius), ball_radius_p)

            cirloc_img = cv2.circle(np.array(img_touch), center, int(radius), (0, 40, 0), 2)
            cv2.imshow(img_name, cirloc_img)
            key = cv2.waitKey(0)
            if key == 119:
                y -= 1
            elif key == 115:
                y += 1
            elif key == 97:
                x -= 1
            elif key == 100:
                x += 1
            elif key == 109:
                radius += 1
            elif key == 110:
                radius -= 1

    cirloc_img = cv2.circle(np.array(img_touch), (int(x), int(y)), int(radius), (0, 40, 0), 2)

    contact_mask = np.zeros((cirloc_img.shape[0], cirloc_img.shape[1]), dtype=np.uint8)
    contact_mask = cv2.circle(np.array(contact_mask), (int(x), int(y)), int(radius), (255, 255, 255), -1)

    contact_center = (int(x),int(y))

    cv2.destroyAllWindows()

    return key, contact_mask, contact_center, int(radius)

# param:
# pixmm -> mm/pixel
# pixmm.x = 0.1mm/pixel
# pixmm.y = 0.109mm/pixel

def calibration_data_cropper(data_folder, ref_file, ball_radius_p):
    ballfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and "cal" in f and f.endswith('.jpg')]

    img_bg_filepath = join(data_folder, ref_file)
    img_bg = affine_transform(cv2.imread(img_bg_filepath))
    file_pointer = 0

    key = -1
    while(1):
        img_filepath = join(data_folder, ballfiles[file_pointer])
        #print("filepath: " + img_filepath)
        img = affine_transform(cv2.imread(img_filepath))

        pre, ext = os.path.splitext(ballfiles[file_pointer])
        crop_filename = pre + ".txt"
        crop_filepath = join(data_folder+"ball_position/", crop_filename)

        if exists(crop_filepath):
            # read the center and radius
            with open(crop_filepath, 'r+') as f:
                # line: x y radius
                line = f.readline()
                circle_data = line.split()
                x = int(circle_data[0])
                y = int(circle_data[1])
                radius = int(circle_data[2])
                key, _, new_center, new_radius = get_touch_mask_by_selection(pre, img_bg, img, ball_radius_p, (x, y), radius)
                new_line = str(new_center[0]) + " " + str(new_center[1]) + " " + str(new_radius)
                f.truncate(0)
                f.seek(0)
                f.write(new_line)

        else:
            with open(crop_filepath, "w") as f:
                key, _, center, radius = get_touch_mask_by_selection(pre, img_bg, img, ball_radius_p)
                line = str(center[0]) + " " + str(center[1]) + " " + str(radius)
                f.write(line)

        #key = cv2.waitKey(0)
        if key == 27:
            break

        file_pointer = (file_pointer + 1) % len(ballfiles)


if __name__ == "__main__":
    calibration_folder = "data/calibration2/"
    ball_ref = "bg-0.jpg"
    pixmm = 0.1  # 0.1mm/pixel
    Rmm = 2.42  # ball radius
    R = Rmm / pixmm

    calibration_data_cropper(calibration_folder, ball_ref, R)
