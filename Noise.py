import cv2 as cv
import numpy as np
import matplotlib as plot

lena = cv.imread('lena.tif')
caman = cv.imread('caman.tif')
baboon = cv.imread('baboon.bmp')

scales = []
for i in range (40):
    lena_noise =  np.random.normal(loc=0, scale=i * 5, size=lena.shape)
    caman_noise = np.random.normal(loc=0, scale=i * 5, size=lena.shape)
    baboon_noise = np.random.normal(loc=0, scale=i * 5, size=lena.shape)
    scales.append(i * 5)

lena = lena + noise
cv.imshow("noise", lena)
cv.waitKey(0)
