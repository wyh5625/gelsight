import cv2
import numpy as np
import math
from affine_transform import affine_transform


x1 = 0
y1 = 0
x2 = 0
y2 = 0
drag = False

two_lines = []

def draw_line(event,x,y,flags,param):
    global x1, y1, x2, y2, drag
    if(event == cv2.EVENT_LBUTTONDOWN):
        if(len(two_lines) == 2):
            two_lines.clear()
        x1 = x
        y1 = y
        drag = True
    elif(event == cv2.EVENT_MOUSEMOVE):
        pass
        # if drag:
        #     cv2.line()
    elif(event == cv2.EVENT_LBUTTONUP):
        x2 = x
        y2 = y
        drag = False
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        two_lines.append((p1, p2))
        # draw a line

def point2line_dist(l1, p):
    # d1
    p1_v = p - l1[0]

    l1_v = l1[1] - l1[0]
    l1_v_std = l1_v / math.sqrt(l1_v[0] ** 2 + l1_v[1] ** 2)

    p1_proj = (p1_v[0] * l1_v_std[0] + p1_v[1] * l1_v_std[1]) * l1_v_std

    d1_v = p1_v - p1_proj
    d1 = math.sqrt(d1_v[0] ** 2 + d1_v[1] ** 2)

    return d1

def line_dist(l1, l2):
    d1 = point2line_dist(l1, l2[0])
    d2 = point2line_dist(l1, l2[1])

    return (d1 + d2)/2







# draw two line on image to get the pixel unit of 1 mm
if __name__ == "__main__":
    img = cv2.imread('data/calibration/5mm-1.jpg')
    img = affine_transform(img)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    #img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('line_drawing')
    cv2.setMouseCallback('line_drawing', draw_line)

    while (1):
        canvas = img.copy()
        for (p1,p2) in two_lines:
            cv2.line(canvas, p1, p2, (0,255,0))

        if len(two_lines) == 2:
            # calculate the distance of two lines
            d = line_dist(two_lines[0], two_lines[1])
            print(d)
        cv2.imshow('line_drawing', canvas)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()