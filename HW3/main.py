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
    keep only half the pixels in Cb and Cr using linear interpolation
    
    input: YCbCr
    
    output: YCbCr
    """
    Y = img[:, :, 0]
    Cb = img[:, :, 1]
    Cr = img[:, :, 2]

    half_Cb = cv2.resize(Cb, dsize=None, fx=0.5, fy=0.5)
    half_Cr = cv2.resize(Cr, dsize=None, fx=0.5, fy=0.5)

    new_Cb = cv2.resize(half_Cb, dsize=Y.shape[::-1])
    new_Cr = cv2.resize(half_Cr, dsize=Y.shape[::-1])

    img420 = np.copy(img)
    img420[:, :, 1] = new_Cb
    img420[:, :, 2] = new_Cr

    return img420


def C420(img):
    ycbcr = RGB2YCbCr(img)
    ycbcr420 = YCbCr420(ycbcr)
    rgb = YCbCr2RGB(ycbcr420)
    # no need to convert it back to BGR
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return rgb


if __name__ == "__main__":
    A = cv2.imread("img.png", cv2.COLOR_BGR2RGB)

    # save the original image
    cv2.imwrite("A.png", A)
    # compress the image with 4:2:0
    B = C420(A)
    # save the 4:2:0 image
    cv2.imwrite("B.png", B)
