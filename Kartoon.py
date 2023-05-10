import math

import cv2 as cv
import numpy as np

def apply_kmeans(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # k defines the number of colors
    K = 20
    # center is the [k][3] matrix which contains colors code
    # label is a [height][width] matrix which contain the index of colors in the center
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

# getting image and adding blur for better edge detection
img = cv.imread('img2.jpg')
img = cv.blur(img, (4, 4))
height , width , color  = img.shape
# edge detection
img_edges = cv.Canny(img,100,200)

# making edges black
for i in range(height):
    for j in range (width):
        if(int(img_edges[i][j]) == 255):
            img[i][j] = [0, 0, 0]

# previous exercise k means function for limited color pool
img = apply_kmeans(img)
img = cv.resize(img, (0, 0), fx = 0.5, fy = 0.5)


cv.imshow('cartonish', img)
cv.waitKey(0)