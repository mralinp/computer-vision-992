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


if __name__ == "__main__":
    img = cv2.imread("einstein.jpg", cv2.IMREAD_GRAYSCALE)
    blurred = gaussianFilter(gaussianFilter(img, 1), 1)
    sub = np.subtract(img, blurred)
    sub = np.abs(sub)

    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if(sub[i, j] > 128):
                sub[i, j] = 255
            else:
                sub[i, j] = 0

    canny = cv2.Canny(img, 60, 160)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(sub, cmap="gray")
    plt.title("Subtracted image")
    plt.subplot(1, 3, 3)
    plt.imshow(canny, cmap="gray")
    plt.title("Canny edges")
    plt.show()
