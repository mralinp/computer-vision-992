import cv2
import numpy as np
from math import sqrt, exp
import matplotlib.pyplot as plt


def conv2D(mat: np, kernel: np) -> np:
    kernel = np.flipud(np.fliplr(kernel))
    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
    padding = (kernel_width - 1)
    offset = padding // 2
    output = np.zeros_like(mat)
    image_padded = np.zeros((mat.shape[0] + padding, mat.shape[1] + padding))
    image_padded[offset:-offset, offset:-offset] = mat

    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            output[y, x] = np.sum(
                image_padded[y:y+kernel_width, x:x+kernel_height]*kernel)
    return output


def boxFilter(img: np) -> np:
    filter = np.ones((3, 3), dtype=np.float32)
    filter = filter/filter.sum()
    res = conv2D(img, filter)
    return res


def gaussian(s: int, t: int, std: float) -> float:
    x = (s*s + t*t)/(2*std*std)
    return exp(-x)


def gaussianKernel(size: int, std: float) -> np:
    kernel = np.zeros((size, size), np.float32)
    k = 0
    if(size % 2 != 0):
        for s in range(-int(size/2), int(size/2) + 1, 1):
            for t in range(-int(size/2), int(size/2) + 1, 1):
                kernel[int(size/2) + s, int(size/2) + t] = gaussian(s, t, std)
                k += kernel[int(size/2) + s, int(size/2) + t]
        kernel = kernel/k
    return kernel


def gaussianFilter(img: np, sigma: float) -> np:
    n = int(6*sigma)
    if (n % 2 == 0):
        n -= 1
    filter = gaussianKernel(n, sigma)
    res = conv2D(img, filter)
    return res


def erms(src: np, img: np) -> float:
    return np.square(np.subtract(src, img)).mean()


if __name__ == "__main__":
    img = cv2.imread("einstein.jpg", cv2.IMREAD_GRAYSCALE)
    box = boxFilter(img)
    gaussianWithSigma1 = gaussianFilter(img, 1)
    gaussianWithSigma1Dubble = gaussianFilter(gaussianWithSigma1, 1)
    gaussianWithSigmaSqrt2 = gaussianFilter(img, sqrt(2))

    rmsBox = erms(img, box)
    rmsGaussianWithSigma1 = erms(img, gaussianWithSigma1)
    rmsGaussianWithSigmaSqrt2 = erms(img, gaussianWithSigmaSqrt2)
    rmsGaussianWithSigma1Dubble = erms(img, gaussianWithSigma1Dubble)

    plt.figure(figsize=(4, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(box, cmap="gray")
    plt.title("Box filter RMSE: %.2f" % (rmsBox))
    plt.subplot(2, 2, 2)
    plt.imshow(gaussianWithSigma1, cmap="gray")
    plt.title("Gaussian filter with sigma=1, RMSE: %.2f" %
              (rmsGaussianWithSigma1))
    plt.subplot(2, 2, 3)
    plt.imshow(gaussianWithSigmaSqrt2, cmap="gray")
    plt.title("Gaussian filter with sigma=sqrt(2), RMSE: %.2f" %
              (rmsGaussianWithSigmaSqrt2))
    plt.subplot(2, 2, 4)
    plt.imshow(gaussianWithSigma1Dubble, cmap="gray")
    plt.title("Gaussian filter with sigma=1 dubble, RMSE: %.2f" %
              (rmsGaussianWithSigma1Dubble))
    plt.show()
