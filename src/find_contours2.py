import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from helpers import BrightnessAndContrastAuto, MorphClose


def onStonesTb(self):
    thStone = cv2.getTrackbarPos('thStone', 'Stones')
    thShadow = cv2.getTrackbarPos('thShadow', 'Stones')
    minSizeMM = cv2.getTrackbarPos('Size(mm)', 'Stones')

    blur = cv2.GaussianBlur(image, (0, 0), 2, 2)

    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    saturation = hsv_image[..., 1]
    brightness = hsv_image[..., 2]
    print()

    minSizePx = round(minSizeMM / pix2mm)
    closerSize = minSizePx / 2.0

    saturation = MorphClose(saturation, 1)
    saturation = BrightnessAndContrastAuto(saturation, 1)

    ret, bwStones = cv2.threshold(saturation, thStone, 255, cv2.THRESH_BINARY_INV)

    k = 6

    bwStones[:k, :] = 255
    bwStones[bwStones.shape[0] - k:, k:] = 255
    bwStones[:, :k] = 255
    bwStones[k:, bwStones.shape[1] - k:] = 255

    contours_stone, hierarchy = cv2.findContours(bwStones, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(saturation, contours_stone, -1, (255, 255, 255), 2)
    cv2.imshow("Threshold Stones+Shadow on Saturation", saturation)

    # if removeShadow:
    brightness = MorphClose(brightness, 1)
    brightness = BrightnessAndContrastAuto(brightness, 1)

    ret, bwShadow = cv2.threshold(brightness, thShadow, 255, cv2.THRESH_BINARY)

    k = 6

    bwShadow[:k, :] = 255
    bwShadow[bwShadow.shape[0] - k:, k:] = 255
    bwShadow[:, :k] = 255
    bwShadow[k:, bwShadow.shape[1] - k:] = 255

    # contours.clear()
    contours_shadow, hierarchy = cv2.findContours(bwShadow, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(brightness, contours_shadow, -1, (255, 255, 255), 2)
    cv2.imshow("Threshold Shadow on Brightness", brightness)

    cv2.bitwise_or(bwShadow, bwStones, bwStones)

    dst = image.copy()
    contours, hierarchy = cv2.findContours(bwStones, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(dst, contours_stone + contours_shadow, -1, (255, 255, 255), 2)
    cv2.drawContours(dst, contours, -1, (255, 255, 255), 2)

    # for i in range(len(contours)):
    #     cv2.polylines(dst, contours[i], True, (0, 255, 0), 2)

    cv2.imshow(winName, dst)


if __name__ == '__main__':
    knowDistancePX = 170
    knowDistanceMM = 150
    pix2mm = knowDistancePX / knowDistanceMM

    image = cv2.imread('data/stone1.jpg')

    # set defaults
    minSizeMM = 10  # width in millimetre
    thStone = 100  # max saturation for stones
    thShadow = 128  # max brightness for shadow
    removeShadow = 1  # try to remove shadows (1=Yes 0=No)

    winName = 'Stones'
    cv2.imshow(winName, image)
    cv2.createTrackbar("Size(mm)", winName, minSizeMM, 100, onStonesTb)
    cv2.createTrackbar("thStone", winName, thStone, 255, onStonesTb)
    cv2.createTrackbar("thShadow", winName, thShadow, 255, onStonesTb)
    # cv2.createTrackbar("remove Shadow", winName, removeShadow, 1, onStonesTb)
    onStonesTb(1)
    cv2.waitKey(0)
