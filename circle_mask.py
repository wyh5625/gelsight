import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join, exists
from affine_transform import affine_transform
import torch
from fast_possion import fast_poisson

def get_input(img, mask=None):

    # calculate the gx and gy
    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)

    RGB = img[mask>0]

    X = x[mask>0]
    Y = y[mask>0]


    return RGB,X,Y

def get_touch_mask(img_touch, center, radius):
    touch_mask = np.zeros((img_touch.shape[0], img_touch.shape[1]), dtype=int)
    touch_mask = cv2.circle(np.array(touch_mask), (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)

    return touch_mask

def get_touch_mask_by_selection(img_name,img_bg, img_touch, ball_radius_p, circle_center=None, circle_radius=None, predicted=False, visual = True, dir=None):

    if circle_center == None or circle_radius == None:
        if predicted:
            x,y,radius = get_touch_mask_by_prediction(img_touch, ball_radius_p)
        else:
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

    else:
        if dir is not None:
            # auto-mode without vision, need to save the calibration img for checking
            center = (int(x), int(y))
            radius = min(int(radius), ball_radius_p)

            cirloc_img = cv2.circle(np.array(img_touch), center, int(radius), (0, 40, 0), 2)
            # cv2.imshow(img_name, cirloc_img)
            cv2.imwrite(dir + f"auto_calibration/loc-{img_name}.jpg", cirloc_img)




    print(f"Touch mask calibrated: {img_name}.jpg")

    cirloc_img = cv2.circle(np.array(img_touch), (int(x), int(y)), int(radius), (0, 40, 0), 2)

    contact_mask = np.zeros((cirloc_img.shape[0], cirloc_img.shape[1]), dtype=np.uint8)
    contact_mask = cv2.circle(np.array(contact_mask), (int(x), int(y)), int(radius), (255, 255, 255), -1)

    contact_center = (int(x),int(y))

    cv2.destroyAllWindows()

    return key, contact_mask, contact_center, int(radius)

def infer_gradient(feature):
    model = torch.load('model/model2.pt', map_location=torch.device('cpu'))
    gradient = model(torch.tensor(feature,dtype=torch.float32,device='cpu'))

    return gradient

def get_ground_image(img,R, center, radius,mask):

    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    xv = x - center[0]
    yv = y - center[1]

    depth = np.sqrt((R*mask/255) ** 2 - (xv*mask/255) ** 2 - (yv*mask/255) ** 2) - np.sqrt( (R*mask/255) ** 2 - (radius*mask/255) **2 )

    return depth

def get_ground_gradient(R):
    radius = int(R)
    x = np.linspace(0, 2*radius-1, 2*radius)
    y = np.linspace(0, 2*radius-1, 2*radius)
    x, y = np.meshgrid(x, y)
    center = [(2*radius-1)/2, (2*radius-1)/2]
    xv = x - center[0]
    yv = y - center[1]

    # create a circle mask
    touch_mask = np.zeros(xv.shape,dtype=np.uint8)
    touch_mask = cv2.circle(np.array(touch_mask), (int(center[0]), int(center[1])), int(radius), 1, -1)

    gx = -xv*touch_mask/np.sqrt(R**2 - xv**2*touch_mask - yv**2*touch_mask)
    gy = -yv*touch_mask/np.sqrt(R**2 - xv**2*touch_mask - yv**2*touch_mask)

    # depth = np.sqrt((R*mask/255) ** 2 - (xv*mask/255) ** 2 - (yv*mask/255) ** 2) - np.sqrt( (R*mask/255) ** 2 - (radius*mask/255) **2 )

    return gx, gy

import matplotlib.pyplot as plt
import pyvista as pv
def on_key_press(interactor, event):
    key = interactor.GetKeySym()
    if key == "Space":
        interactor.ExitCallback()



def depth_map_plot(img):

    X = np.arange(0, img.shape[1], 1)
    Y = np.arange(0, img.shape[0], 1)

    # print(img.shape)

    Z = img
    data = []
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            data.append([X[i],Y[j],Z[j,i]])
    data = np.array(data)
    mesh = pv.PolyData(data)
    mesh['Data'] = data[:,2]
    # mesh.plot(point_size=2, screenshot='random_nodes.png')
    p = pv.Plotter()

    # def on_key_press(*args):
    #     print("pressed")
    #     p.close()
    #
    # p.add_key_event('v', on_key_press)

    p.add_mesh(mesh,point_size=2, show_edges= True, lighting=False)
    p.show(auto_close=True)
    p.close()


def close(event):
    if event.key == ' ':
        plt.close()

def get_touch_mask_by_prediction(img, R):

    # predict gx,gy from img
    border_width = 0
    data_mask = np.ones((img.shape[0] - border_width * 2, img.shape[1] - border_width * 2))
    data_mask = np.pad(data_mask, pad_width=border_width, mode='constant', constant_values=0)
    RGB, X, Y = get_input(img, data_mask)


    feature = np.column_stack((RGB, X, Y))
    gradient = infer_gradient(feature)
    gradient = np.array(gradient.tolist())
    gx = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gy = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)


    for i in range(feature.shape[0]):
        gx[int(feature[i,4]),int(feature[i,3])] = gradient[i,0]
        gy[int(feature[i,4]),int(feature[i,3])] = gradient[i,1]

    ground_gx, ground_gy = get_ground_gradient(R)


    circle_mask = np.zeros(ground_gx.shape, dtype=np.uint8)
    circle_mask = cv2.circle(np.array(circle_mask), (int(ground_gx.shape[0] / 2), int(ground_gx.shape[0] / 2)), int(R), 1, -1)
    vis = img.copy()

    # ground_gy = np.abs(ground_gy)

    min_value = np.min(ground_gx)
    max_value = np.max(ground_gx)
    # print(min_value)
    # print(max_value)
    # scaled_patch = (groud_gx - min_value) / (max_value - min_value)
    # min_value = np.min(scaled_patch)
    # max_value = np.max(scaled_patch)
    # print(min_value)
    # print(max_value)
    # print(scaled_patch)
    # print(groud_gx)

    # img = cv2.normalize(ground_gy, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # cv2.imshow('patch Square', img)
    # cv2.waitKey(0)
    min_error = 9999
    loc_x = 0
    loc_y = 0
    # slide a search window
    for x in range(0, img.shape[1] - ground_gx.shape[1], 1):
        for y in range(0, img.shape[0] - ground_gx.shape[0], 1):
            # match the search window with tp
            roi_gx = (gx[y:y + ground_gx.shape[0], x:x + ground_gx.shape[1]]).copy()
            roi_gy = (gy[y:y + ground_gx.shape[0], x:x + ground_gx.shape[1]]).copy()

            # touch_mask = np.zeros(xv.shape, dtype=np.uint8)
            # circle_mask = cv2.circle(np.array(roi_gx), (int(groud_gx.shape[0]/2), int(groud_gx.shape[0]/2), int(radius), 1, -1)
            roi_gx = roi_gx * circle_mask
            roi_gy = roi_gy * circle_mask

            # Compute the SSD similarity between the patch and the search window
            # print(groud_gx)
            ssd_similarity = np.sum((roi_gx - ground_gx) ** 2) + np.sum((roi_gy - ground_gy) ** 2)
            avg_error = ssd_similarity/np.sum(circle_mask)
            if avg_error < min_error:
                min_error = avg_error
                loc_x = x
                loc_y = y
                # print(f'SSD similarityq: {avg_error}')
            # Print the SSD and NCC similarities for the current search window


    # print("------------ Match radius --------------")

    # match radius
    roi_gx = (gx[loc_y:loc_y + ground_gx.shape[0], loc_x:loc_x + ground_gx.shape[1]]).copy()
    roi_gy = (gy[loc_y:loc_y + ground_gx.shape[0], loc_x:loc_x + ground_gx.shape[1]]).copy()

    last_error = None
    best_radius = R
    for ratio in np.linspace(1, 0.2, 20):
        circle_mask = np.zeros(ground_gx.shape, dtype=np.uint8)
        circle_mask = cv2.circle(np.array(circle_mask), (int(ground_gx.shape[0] / 2), int(ground_gx.shape[0] / 2)),
                                 int(R*ratio), 1, -1)
        roi_gx = roi_gx * circle_mask
        roi_gy = roi_gy * circle_mask

        roi_ground_gx = ground_gx * circle_mask
        roi_ground_gy = ground_gy * circle_mask

        ssd_similarity = np.sum((roi_gx - roi_ground_gx) ** 2) + np.sum((roi_gy - roi_ground_gy) ** 2)
        avg_error = ssd_similarity / np.sum(circle_mask)

        if last_error is None:
            last_error = avg_error
        else:
            if avg_error > last_error:
                break
            else:
                last_error = avg_error
                best_radius = R * ratio

        # print(f'SSD similarityq: {avg_error}')

    return loc_x + ground_gx.shape[1]/2, loc_y + ground_gx.shape[1]/2, best_radius

# param:
# pixmm -> mm/pixel
# pixmm.x = 0.1mm/pixel
# pixmm.y = 0.109mm/pixel

def get_numeric_value(filename):
    return int(''.join(filter(str.isdigit, filename)))

def calibration_data_cropper(data_folder, ref_file, ball_radius_p):
    ballfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and "cal" in f and f.endswith('.jpg')]
    ballfiles = sorted(ballfiles, key=get_numeric_value)

    img_bg_filepath = join(data_folder, ref_file)
    img_bg = affine_transform(cv2.imread(img_bg_filepath))
    start_pointer = 0

    key = -1
    for file_pointer in range(len(ballfiles)):
        img_filepath = join(data_folder, ballfiles[file_pointer])
        #print("filepath: " + img_filepath)
        img = affine_transform(cv2.imread(img_filepath))

        pre, ext = os.path.splitext(ballfiles[file_pointer])
        crop_filename = pre + ".txt"
        crop_filepath = join(data_folder+"ball_position/", crop_filename)
        new_create = False
        if exists(crop_filepath):
            # read the center and radius
            with open(crop_filepath, 'r+') as f:
                # line: x y radius
                line = f.readline()
                circle_data = line.split()
                x = int(circle_data[0])
                y = int(circle_data[1])
                radius = int(circle_data[2])
                key, _, new_center, new_radius = get_touch_mask_by_selection(pre, img_bg, img, ball_radius_p, (x, y), radius, visual=False, dir=data_folder)
                new_line = str(new_center[0]) + " " + str(new_center[1]) + " " + str(new_radius)
                f.truncate(0)
                f.seek(0)
                f.write(new_line)

        else:
            new_create = True
            with open(crop_filepath, "w") as f:
                key, _, center, radius = get_touch_mask_by_selection(pre, img_bg, img, ball_radius_p, predicted=True, visual=False, dir=data_folder)
                line = str(center[0]) + " " + str(center[1]) + " " + str(radius)
                f.write(line)

        #key = cv2.waitKey(0)
        if key == 27:
            if os.path.exists(crop_filepath) and new_create:
                os.remove(crop_filepath)
            break

        # file_pointer = (file_pointer + 1) % len(ballfiles)


if __name__ == "__main__":
    calibration_folder = "data/calibration7/"
    ball_ref = "bg-0.jpg"
    pixmm = 0.1  # 0.1mm/pixel
    Rmm = 1.98  # ball radius
    R = Rmm / pixmm

    calibration_data_cropper(calibration_folder, ball_ref, R)
