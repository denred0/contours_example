import cv2
import numpy as np

from helpers import *

img = cv2.imread('data/stone1.jpg', cv2.IMREAD_COLOR)

blur = cv2.GaussianBlur(img, (3, 3), 0)

saturation = MorphClose(blur, 1)
saturation = BrightnessAndContrastAuto(saturation, 0)
cv2.imshow("BrightnessAndContrastAuto", saturation)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(saturation, cv2.CV_64F, 1, 0, ksize=3, scale=2)
y = cv2.Sobel(saturation, cv2.CV_64F, 0, 1, ksize=3, scale=2)
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
