# 1. get real-time img
# 2. compute marker mask and contact area
# 3. infer gradient Gx,Gy
# 4. depth reconstrubtion
# 5. interpolate the marker area
import cv2
from affine_transform import affine_transform
from marker_mask import *
from circle_mask import get_touch_mask_by_selection
from gradient import get_gradient
from fast_possion import fast_poisson
from model import MLPEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pyvista as pv

# plt.style.use('_mpl-gallery')

pixmm = 0.1  # 0.1mm/pixel
Rmm = 2.42  # ball radius
R = Rmm / pixmm

bg = cv2.imread('data/calibration/bg-0.jpg')

def infer_gradient(feature):
    model = torch.load('model/model2.pt')
    gradient = model(torch.tensor(feature,dtype=torch.float32,device='cuda:0'))

    return gradient

def get_input(img, mask):

    # calculate the gx and gy
    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)

    RGB = img[mask>0]

    X = x[mask>0]
    Y = y[mask>0]

    return RGB,X,Y

def get_ground_image(img,R, center, radius,mask):

    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)
    xv = x - center[0]
    yv = y - center[1]

    depth = np.sqrt((R*mask/255) ** 2 - (xv*mask/255) ** 2 - (yv*mask/255) ** 2) - np.sqrt( (R*mask/255) ** 2 - (radius*mask/255) **2 )

    return depth

def plt_show(img):

    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6),
    #                         subplot_kw={'xticks': [], 'yticks': []})
    # axs.imshow(img, interpolation='nearest', cmap='viridis')
    #
    # plt.show()

    plt.figure()
    plt.imshow(img)
    plt.colorbar(label='Depth')
    plt.title('Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()


def depth_map_plot(img):
    X = np.arange(0, 320, 1)
    Y = np.arange(0, 240, 1)
    Z = img
    data = []
    for i in range(320):
        for j in range(240):
            data.append([X[i], Y[j], Z[i, j]])
    data = np.array(data)
    mesh = pv.PolyData(data)
    mesh['Data'] = data[:, 2]
    # mesh.plot(point_size=2, screenshot='random_nodes.png')
    p = pv.Plotter()
    p.add_mesh(mesh, point_size=2, show_edges=True, lighting=False)
    p.show()

for i in range(1,2):

    data_path = 'data/calibration2/'
    img = cv2.imread(data_path + 'cal-%d.jpg'%i)

    filepath = data_path + "ball_position/" + "cal-%d.txt"%i
    with open(filepath, 'r+') as f:
        # line: x y radius
        line = f.readline()
        circle_data = line.split()
        x = int(circle_data[0])
        y = int(circle_data[1])
        radius = int(circle_data[2])

    # get valid mask
    dot_mask = get_marker_mask(affine_transform(img))

    _, touch_mask, center, radius = get_touch_mask_by_selection("circle_selector", affine_transform(bg), affine_transform(img),
                                                           int(R),(x, y), radius, visual=False)

    valid_mask = (1 - dot_mask / 255) * touch_mask

    # infer gradient
    RGB, X, Y = get_input(affine_transform(img), valid_mask)

    feature = np.column_stack((RGB, X, Y))
    gradient = infer_gradient(feature)
    gradient = np.array(gradient.tolist())
    gx = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gy = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(feature.shape[0]):
        gx[int(feature[i,4]),int(feature[i,3])] = gradient[i,0]
        gy[int(feature[i,4]),int(feature[i,3])] = gradient[i,1]

    # RECONSTRUCTION
    img_ = fast_poisson(gx,gy)

    ground_image = get_ground_image(affine_transform(img),R, center,radius,touch_mask.astype('float'))

    error = np.linalg.norm((img_ - ground_image).flatten(),ord=2)
    print(error)

    depth_map_plot(img_)

    plt_show(img_)
    plt_show(ground_image)
    plt_show(abs(img_ - ground_image))
