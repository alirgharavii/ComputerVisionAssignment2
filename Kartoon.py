import math

import cv2 as cv
import numpy as np

def apply_kmeans(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # k defines the number of colors
    K = 30
    # center is the [k][3] matrix which contains colors code
    # label is a [height][width] matrix which contain the index of colors in the center
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

img = cv.imread('img2.jpg')
img =apply_kmeans(img)
img = cv.blur(img, (4, 4))
height , width , color  = img.shape
#img = cv.resize(img, (int(width/2), int(height/2)))
img_edges = cv.Canny(img,100,200)

height , width  = img_edges.shape


for i in range(height):
    for j in range(width):
        if (img_edges[i][j] == 255):
            img[i][j] = [0, 0, 0]




cv.imshow('cartonish', img)
cv.waitKey(0)