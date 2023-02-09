# import the necessary packages
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from affine_transform import affine_transform

# 'optional' argument is required for trackbar creation parameters
def nothing(a):
    pass

# return a 3-channel mask
def get_marker_mask_hsv(img):
    frame = cv2.GaussianBlur(img, (5, 5), 0)
    # convert from a BGR stream to an HSV stream
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    HSVLOW = np.array([0, 0, 0])
    HSVHIGH = np.array([180, 255, 75])

    mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    masks = cv2.merge((mask, mask, mask))

    return masks

pto_o = np.float32([[33,63],[213,63],[54,248],[187,248]])
pts_d = np.float32([[0,0],[240,0],[0,320],[240,320]])
M = cv2.getPerspectiveTransform(pto_o,pts_d)

def pt(img):
    pt_img = cv2.warpPerspective(img, M, (240, 320))
    return pt_img

def pt_and_to_hsv(img):
    img = cv2.warpPerspective(img, M, (240, 320))
    # it is common to apply a blur to the frame
    frame = cv2.GaussianBlur(img, (5, 5), 0)
    # convert from a BGR stream to an HSV stream
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return hsv

def get_marker_by_hsv_range():
    mypath = "data/calibration/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)

    file_pointer = 0

    img = cv2.imread(join(mypath, onlyfiles[file_pointer]))
    hsv = pt_and_to_hsv(img)

    cv2.namedWindow('Origin')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Slider')

    # assign strings for ease of coding
    hh = 'Hue High'
    hl = 'Hue Low'
    sh = 'Saturation High'
    sl = 'Saturation Low'
    vh = 'Value High'
    vl = 'Value Low'
    ori = 'Origin'
    wnd = 'Mask'
    slider = 'Slider'
    # Begin Creating trackbars for each
    cv2.createTrackbar(hl, slider, 0, 180, nothing)
    cv2.createTrackbar(hh, slider, 180, 180, nothing)
    cv2.createTrackbar(sl, slider, 0, 255, nothing)
    cv2.createTrackbar(sh, slider, 255, 255, nothing)
    cv2.createTrackbar(vl, slider, 0, 255, nothing)
    cv2.createTrackbar(vh, slider, 110, 255, nothing)

    while (1):

        # read trackbar positions for each trackbar
        hul = cv2.getTrackbarPos(hl, slider)
        huh = cv2.getTrackbarPos(hh, slider)
        sal = cv2.getTrackbarPos(sl, slider)
        sah = cv2.getTrackbarPos(sh, slider)
        val = cv2.getTrackbarPos(vl, slider)
        vah = cv2.getTrackbarPos(vh, slider)

        # make array for final values
        HSVLOW = np.array([hul, sal, val])
        HSVHIGH = np.array([huh, sah, vah])

        # create a mask for that range
        mask = cv2.inRange(hsv, HSVLOW, HSVHIGH)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        masks = cv2.merge((mask, mask, mask))

        # res = cv2.bitwise_and(img, img, mask=mask)

        added = cv2.add(pt(img), masks)

        cv2.imshow(ori, pt(img))
        cv2.imshow(wnd, added)
        k = cv2.waitKey(5)  # amp;&amp; 0xFF
        if k == ord('q'):
            break
        elif k == 32:
            file_pointer = (file_pointer + 1) % len(onlyfiles)
            img = cv2.imread(join(mypath, onlyfiles[file_pointer]))
            hsv = pt_and_to_hsv(img)

    cv2.destroyAllWindows()

def get_marker_mask_by_blob(src=None):
    # create a blob detector
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
    params.minDistBetweenBlobs = 1
    params.filterByArea = False
    params.minArea = 1
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    detector = cv2.SimpleBlobDetector_create(params)

    if src is None:
        mypath = "data/calibration/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        file_pointer = 0

        img = cv2.imread(join(mypath, onlyfiles[file_pointer]))

        gray = cv2.cvtColor(pt(img), cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(gray, (5, 5), 0)
        # binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)


        keypoints = detector.detect(frame)

        old_pointer = file_pointer
        pt_img = pt(img)
        mask = np.zeros_like(img)

        while(1):

            if old_pointer != file_pointer:
                # pointer change, update the image and keypoints
                img = cv2.imread(join(mypath, onlyfiles[file_pointer]))
                gray = cv2.cvtColor(pt(img), cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(gray, (5, 5), 0)

                keypoints = detector.detect(frame)
                mask = np.zeros_like(img)
                pt_img = pt(img)

                old_pointer = file_pointer

            for i in range(len(keypoints)):
                cv2.circle(pt_img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (255, 255, 255), 1)
                # cv2.ellipse(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), (9, 7), 0, 0, 360, (1), -1)

            cv2.imshow("blob", pt_img)
            k = cv2.waitKey(0)

            if k == ord('q'):
                break
            elif k == 100:
                file_pointer = (file_pointer + 1) % len(onlyfiles)

            elif k == 97:
                file_pointer = ((file_pointer - 1) + len(onlyfiles))% len(onlyfiles)
    else:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(gray, (5, 5), 0)
        keypoints = detector.detect(frame)
        mask = np.zeros((src.shape[0],src.shape[1]), dtype=np.uint8)
        for i in range(len(keypoints)):
            cv2.circle(mask, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (255, 255, 255), -1)

        cv2.imshow("blob", mask)
        cv2.waitKey(0)
    return mask

def get_marker_mask(src=None, threshold = 60 ):

    if src is None:
        mypath = "data/calibration/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        for i in range(len(onlyfiles)):

            img = cv2.imread(join(mypath, onlyfiles[i]))
            img = affine_transform(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            mask = np.zeros_like(gray)
            mask[(gray < threshold)] = 255
            cv2.imshow("img", img)
            cv2.waitKey(0)

            img[(gray < threshold)] = 255

            cv2.imshow("img", img)
            cv2.waitKey(0)

    else:

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
        mask[(gray < threshold)] = 255

    return mask
if __name__ == "__main__":

    # img = cv2.imread('data/img_.jpg')
    # img = cv2.imread('data/calibration/ball-11.jpg')
    # get_marker_by_hsv_range()
    # img = cv2.imread('data/calibration/ball-1.jpg')
    # img= pt(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # ref = cv2.imread('data/calibration/bg-0.jpg')
    # ref = pt(ref)
    # gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    #
    # dot_mask = ((gray < 50) + (gray_ref < 50)) >0
    # img[dot_mask] = 255
    # ref[dot_mask] = 255
    #
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    #
    # cv2.imshow("ref", ref)
    # cv2.waitKey(0)
    #
    #
    # diff_img = np.abs(np.sum(img.astype(np.float32) - ref.astype(np.float32), axis=2))
    # contact_mask = (diff_img > 30).astype(np.uint8)*255
    # cv2.imshow('diff_img', diff_img.astype(np.uint8))
    # cv2.waitKey(0)

    get_marker_mask()
