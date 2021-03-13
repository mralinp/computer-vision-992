import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2


def combainImgs(img1: np, img2: np) -> np:
    res = np.zeros(img1.shape)
    f = fft2(img1)
    f = np.log(f)
    f = fftshift(f)
    plt.imshow(f)
    plt.title("FFT of img1")
    plt.show()
    return res


if __name__ == "__main__":
    img1 = cv2.imread("leopard.jpg")
    img2 = cv2.imread("elephant.jpg")
    img3 = combainImgs(img1, img2)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title("Combained Image")
    plt.imshow()
