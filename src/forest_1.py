import cv2
import numpy as np


image = cv2.imread('forest/img/000000.png', cv2.IMREAD_COLOR)
blur = cv2.GaussianBlur(image, (0, 0), 2, 2)

res = np.hstack((image, blur))

cv2.imshow('1', res)
cv2.waitKey(0)