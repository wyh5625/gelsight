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
from model import MLPEncoder

# plt.style.use('_mpl-gallery')

pixmm = 0.1  # 0.1mm/pixel
Rmm = 2.42  # ball radius
R = Rmm / pixmm

ref = cv2.imread('data/calibration/bg-0.jpg')

def infer_gradient(feature):
    state_dict = torch.load('model/model7.pt')
    model = MLPEncoder()
    model.load_state_dict(state_dict)
    model.eval()
    model.to('cuda:0')
    # model = torch.load('model/model5.pt')
    gradient = model(torch.tensor(feature,dtype=torch.float32,device='cuda:0'))

    return gradient

def get_input(img, mask=None):

    # calculate the gx and gy
    x = np.linspace(0, img.shape[1] - 1, img.shape[1])
    y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        return img, x, y
    else:
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

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6),
                            subplot_kw={'xticks': [], 'yticks': []})
    axs.imshow(img, interpolation='nearest', cmap='viridis')

    plt.show()

def get_contact_mask(img,ref,valid_mask):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # gray = gray * valid_mask
    # gray_ref = gray_ref * valid_mask
    dot_mask = valid_mask<1
    gray[dot_mask] = 0
    gray_ref[dot_mask] = 0

    # cv2.imshow('gray2',gray)
    # cv2.waitKey(0)
    # cv2.imshow('gray_ref',gray_ref)
    # cv2.waitKey(0)

    diff = abs(gray.astype('float')-gray_ref.astype('float'))

    # cv2.imshow('diff',diff.astype('uint8'))
    # cv2.waitKey(0)

    ret,thresh_img = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    cv2.imshow('diff',thresh_img)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh_img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    index = np.where(np.array(areas)>1)[0]
    cnt = contours[index[0]]
    for i in range(len(index)-1):
        cnt = np.concatenate([cnt,contours[index[i+1]]],axis=0)

    [x, y, w, h] = cv2.boundingRect(cnt)

    #create an empty image for contours
    img_contours = np.zeros(img.shape)

    cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # draw the contours on the empty image
    cv2.drawContours(img_contours, cnt, -1, (0,255,0), 3)
    cv2.imshow('contours',img_contours)
    cv2.waitKey(0)

    return x , y, x + w, y + h

def depth_map_plot(img):

    X = np.arange(0, 320, 1)
    Y = np.arange(0, 240, 1)
    Z = img
    data = []
    for i in range(320):
        for j in range(240):
            data.append([X[i],Y[j],Z[i,j]])
    data = np.array(data)
    # mesh = pv.PolyData(data)
    # mesh['Data'] = data[:,2]
    # surface = mesh.delaunay_3d(alpha=0.5)
    # mesh.plot(point_size=2, screenshot='random_nodes.png')
    p = pv.Plotter()

    # Create a PyVista structured grid from the points
    grid = pv.StructuredGrid()
    grid.points = data
    grid.dimensions = 240, 320, 1

    # Plot the surface mesh
    surf = grid.extract_surface()

    # plot the surface
    p = pv.Plotter()
    p.add_mesh(surf)
    p.show()


for i in range(1,2):

    # img = cv2.imread('data/texture/1-31-0-3.jpg')
    # img = cv2.imread('data/calibration5/cal-80.jpg')
    img = cv2.imread('data/test/screw-2.jpg')

    dot_mask = get_marker_mask(affine_transform(img), threshold = 100)

    dot_mask_ref = get_marker_mask(affine_transform(ref), threshold = 100)

    valid_mask = (1 - dot_mask / 255) * (1 - dot_mask_ref / 255)

    # bounding_box = get_contact_mask(affine_transform(img),affine_transform(ref),valid_mask)
    # bounding_box = [40, 50, 200, 280]
    # touch_mask = np.zeros_like(dot_mask)
    # touch_mask[bounding_box[1]:bounding_box[3],bounding_box[0]:bounding_box[2]] = 1
    # touch_mask = np.ones_like(dot_mask)

    valid_mask = (1 - dot_mask / 255)

    ## infer gradient
    RGB, X, Y = get_input(affine_transform(img), valid_mask)

    feature = np.column_stack((RGB, X, Y))
    gradient = infer_gradient(feature)
    gradient = np.array(gradient.tolist())
    gx = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gy = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for i in range(feature.shape[0]):
        gx[int(feature[i,4]),int(feature[i,3])] = gradient[i,0]
        gy[int(feature[i,4]),int(feature[i,3])] = gradient[i,1]

    ## reconstruction

    img_ = fast_poisson(gx,gy)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6),
                            subplot_kw={'xticks': [], 'yticks': []})
    axs.imshow(img_, interpolation='nearest', cmap='viridis')

    plt.show()

    depth_map_plot(img_)

