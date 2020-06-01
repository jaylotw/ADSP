from cv2 import cv2 as cv 
import argparse
import numpy as np

def SSIM(x, y, c1, c2):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    L = 255.0

    # mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # variance of x and y
    corrcoef = np.corrcoef(x.ravel(), y.ravel())
    var_x = corrcoef[0][0]
    var_y = corrcoef[1][1]

    # covariance of x and y
    cov_xy = corrcoef[0][1]

    sq_c1L = (c1*L)**2
    sq_c2L = (c2*L)**2

    return ((2*mean_x*mean_y + sq_c1L)*(2*cov_xy + sq_c2L)) / ((mean_x**2 + mean_y**2 + sq_c1L)*(var_x + var_y + sq_c2L))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SSIM")
    parser.add_argument("--image1", dest="img1", default="./HW4/A.png")
    parser.add_argument("--image2", dest="img2", default="./HW4/B.png")
    parser.add_argument("--c1", dest="c1", type=float, default=0.1)
    parser.add_argument("--c2", dest="c2", type=float, default=0.1)

    args = parser.parse_args()

    A = cv.imread(args.img1, cv.IMREAD_GRAYSCALE)
    B = cv.imread(args.img2, cv.IMREAD_GRAYSCALE)

    print("SSIM: ", SSIM(A, B, args.c1, args.c2))