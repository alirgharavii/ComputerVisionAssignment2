import math

import cv2 as cv
import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt

def mse(img,i , j):
    height, width = img.shape
    gauss_noise = np.zeros((height, width), dtype=np.uint8)
    cv.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * 0.5 * j).astype(np.uint8)
    noised_img = cv.add(img,gauss_noise)
    gaussian_img = cv.GaussianBlur(noised_img,(2 * i + 1, 2 * i  + 1),0)
    sum = 0
    for i in range(height):
        for j in range(width):
            sum = sum + (img[i][j] - gaussian_img[i][j]) ** 2
    return sum/ (width * height)

def psnr(img, i, j):
    return (10 * math.log10(255 ** 2)/mse(img, i, j))


lena = cv.imread('lena.tif')
lena = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
caman = cv.imread('caman.tif')
caman = cv.cvtColor(caman, cv.COLOR_BGR2GRAY)
baboon = cv.imread('baboon.bmp')
baboon = cv.cvtColor(baboon, cv.COLOR_BGR2GRAY)

lena_low_mse = []
lena_mid_mse = []
lena_high_mse = []
caman_low_mse = []
caman_mid_mse = []
caman_high_mse = []
baboon_low_mse = []
baboon_mid_mse = []
baboon_high_mse = []
lena_low_psnr = []
lena_mid_psnr = []
lena_high_psnr = []
caman_low_psnr = []
caman_mid_psnr = []
caman_high_psnr = []
baboon_low_psnr = []
baboon_mid_psnr = []
baboon_high_psnr = []
gaussians = []
for i in range (10):
    lena_noise_low =  np.random.normal(loc=1, scale=100, size=lena.shape)
    caman_noise_low = np.random.normal(loc=1, scale=100, size=caman.shape)
    baboon_noise_low = np.random.normal(loc=1, scale=100, size=baboon.shape)
    lena_noise_mid = np.random.normal(loc=5, scale=100, size=lena.shape)
    caman_noise_mid = np.random.normal(loc=5, scale=100, size=caman.shape)
    baboon_noise_mid = np.random.normal(loc=5, scale=100, size=baboon.shape)
    lena_noise_high = np.random.normal(loc=10, scale=100, size=lena.shape)
    caman_noise_high = np.random.normal(loc=10, scale=100, size=caman.shape)
    baboon_noise_high = np.random.normal(loc=10, scale=100, size=baboon.shape)
    lena_low_mse.append(mse(lena, i, 1))
    lena_mid_mse.append(mse(lena, i, 2))
    lena_high_mse.append(mse(lena, i, 3))
    caman_low_mse.append(mse(caman, i, 1))
    caman_mid_mse.append(mse(caman, i, 2))
    caman_high_mse.append(mse(caman, i, 3))
    baboon_low_mse.append(mse(baboon, i, 1))
    baboon_mid_mse.append(mse(baboon, i, 2))
    baboon_high_mse.append(mse(baboon, i, 3))
    lena_low_psnr.append(psnr(lena, i, 1))
    lena_mid_psnr.append(psnr(lena, i, 2))
    lena_high_psnr.append(psnr(lena, i, 3))
    caman_low_psnr.append(psnr(caman, i, 1))
    caman_mid_psnr.append(psnr(caman, i, 2))
    caman_high_psnr.append(psnr(caman, i, 3))
    baboon_low_psnr.append(psnr(baboon, i, 1))
    baboon_mid_psnr.append(psnr(baboon, i, 2))
    baboon_high_psnr.append(psnr(baboon, i, 3))
    gaussians.append((i * 2  + 1) ** 2)

plt.plot(gaussians, lena_low_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena low MSE")
plt.show()

plt.plot(gaussians, lena_mid_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena mid MSE")
plt.show()

plt.plot(gaussians, lena_high_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena high MSE")
plt.show()

plt.plot(gaussians, caman_low_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("camman low MSE")
plt.show()

plt.plot(gaussians, caman_mid_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("caman mid MSE")
plt.show()

plt.plot(gaussians, caman_high_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("caman high MSE")
plt.show()

plt.plot(gaussians, baboon_low_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon low MSE")
plt.show()

plt.plot(gaussians, baboon_mid_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon mid MSE")
plt.show()

plt.plot(gaussians, baboon_high_mse)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon high MSE")
plt.show()

plt.plot(gaussians, lena_low_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena low PSNR")
plt.show()

plt.plot(gaussians, lena_mid_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena mid PSNR")
plt.show()

plt.plot(gaussians, lena_high_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Lena high PSNR")
plt.show()

plt.plot(gaussians, caman_low_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("camman low PSNR")
plt.show()

plt.plot(gaussians, caman_mid_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("caman mid PSNR")
plt.show()

plt.plot(gaussians, caman_high_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("caman high PSNR")
plt.show()

plt.plot(gaussians, baboon_low_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon low PSNR")
plt.show()

plt.plot(gaussians, baboon_mid_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon mid PSNR")
plt.show()

plt.plot(gaussians, baboon_high_psnr)
plt.xlabel("gaussian size")
plt.ylabel("MSE")
plt.title("Baboon high PSNR")
plt.show()

