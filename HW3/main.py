import cv2
import numpy as np


def RGB2YCbCr(img):
    """
    Convert RGB to YCbCr color space
    """

    ycbcr = np.matmul(
        img, [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
    )

    return ycbcr


def YCbCr2RGB(img):
    """
    Convert YCbCr to RGB color space
    """

    rgb = np.matmul(img, [[1, 0, 1.402], [1, -0.344, -0.714], [1, 1.772, 0]])
    rgb = rgb.round()
    rgb_clip = np.clip(rgb, 0, 255)

    return rgb_clip.astype(np.uint8)


def YCbCr420(img):
    """
    keep only half the pixel in Cb and Cr using linear interpolation
    
    input: YCbCr
    
    output: YCbCr
    """
    Y = img[:, :, 0]
    Cb = img[:, :, 1]
    Cr = img[:, :, 2]

    h = Y.shape[0]
    w = Y.shape[1]
    half_h = int(h / 2)
    half_w = int(w / 2)

    half_Cb = cv2.resize(Cb, (half_h, half_w))
    half_Cr = cv2.resize(Cr, (half_h, half_w))

    new_Cb = cv2.resize(half_Cb, (h, w))
    new_Cr = cv2.resize(half_Cr, (h, w))

    img420 = np.copy(img)
    img420[:, :, 1] = new_Cb
    img420[:, :, 2] = new_Cr

    return img420


def C420(img):
    ycbcr = RGB2YCbCr(img)
    ycbcr420 = YCbCr420(ycbcr)
    rgb = YCbCr2RGB(ycbcr420)
    # no need to convert it back to BGR for some reason???
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return rgb


if __name__ == "__main__":
    A = cv2.imread("color_gradient.png", cv2.COLOR_BGR2RGB)
    B = C420(A)
    cv2.imwrite("B.png", B)
