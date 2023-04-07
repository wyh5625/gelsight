import numpy as np
import cv2



def affine_transform(img):
    # pto_o = np.float32([[33, 58], [208, 61], [56, 250], [185, 250]])
    pto_o = np.float32([[33, 58], [208, 61], [56, 250], [185, 244]])
    pts_d = np.float32([[0, 0], [240, 0], [0, 320], [240, 320]])
    M = cv2.getPerspectiveTransform(pto_o, pts_d)

    img_ = cv2.warpPerspective(img, M, (240, 320))

    return img_

if __name__ == "__main__":
    img = cv2.imread('data/img.jpg')
    # img_ = img[78:235,57:184]
    # cv2.imshow("ref", img_)

    # [57,78],[184,78],[69,235],[169,235]
    # [33,63],[213,63],[54,248],[187,248]

    img_ = affine_transform(img)

    cv2.imwrite('data/img_.jpg', img_)

    cv2.imshow("img", img)
    cv2.imshow("ybk", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()