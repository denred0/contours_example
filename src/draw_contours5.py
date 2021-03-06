import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

from helpers import BrightnessAndContrastAuto, MorphClose


def onStonesTb(img):
    image = cv2.imread(str(img), cv2.IMREAD_COLOR)

    # thStone = cv2.getTrackbarPos('thStone', 'Stones')
    # thShadow = cv2.getTrackbarPos('thShadow', 'Stones')
    # minSizeMM = cv2.getTrackbarPos('Size(mm)', 'Stones')

    h, w = image.shape[:2]

    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow("blur", blur)

    # imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # src_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # saturation = src_hsv[..., 1]
    # brightness = src_hsv[..., 2]

    saturation = MorphClose(blur, 1)
    saturation = BrightnessAndContrastAuto(saturation, 0)
    cv2.imshow("BrightnessAndContrastAuto", saturation)
    cv2.waitKey(0)

    # # if removeShadow:
    # brightness = MorphClose(brightness, 1)
    # brightness = BrightnessAndContrastAuto(brightness, 0)
    #
    # ret, bwShadow = cv2.threshold(brightness, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bwShadow", brightness)
    # cv2.waitKey(0)

    # contours_shadow, hierarchy = cv2.findContours(bwShadow, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(brightness, contours_shadow, -1, (255, 255, 255), 2)
    # cv2.imshow("Threshold Shadow on Brightness", brightness)



    # # contours.clear()
    #
    # bwShadow[:k, :] = 255
    # bwShadow[bwStones.shape[0] - k:, k:] = 255
    # bwShadow[:, :k] = 255
    # bwShadow[k:, bwShadow.shape[1] - k:] = 255

    thres = 100
    cont = []
    while not cont:
        ret, bwStones = cv2.threshold(saturation, thres, 255, cv2.THRESH_BINARY_INV)
        # cv2.bitwise_or(bwStones, bwShadow, bwStones)

        k = 6

        bwStones[:k, :] = 255
        bwStones[bwStones.shape[0] - k:, k:] = 255
        bwStones[:, :k] = 255
        bwStones[k:, bwStones.shape[1] - k:] = 255

        contours_stone, hierarchy = cv2.findContours(bwStones, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # c = max(contours_stone, key=cv2.contourArea)

        for c in contours_stone:
            if cv2.contourArea(c) > 2500:
                ex = False
                for x_y in c:
                    if x_y[0][0] == h or x_y[0][1] == w or x_y[0][0] == 0 or x_y[0][1] == 0:
                        ex = True
                if not ex:
                    cont.append(c)
        thres += 1

        if thres > 255:
            break

    cv2.drawContours(saturation, cont, -1, (0, 0, 255), 2)
    # cv2.imshow("bwStones", saturation)

    # blur2 = cv2.GaussianBlur(image, (0, 0), 2, 2)

    # hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #
    # saturation = hsv_image[..., 1]
    # brightness = hsv_image[..., 2]
    #
    # minSizePx = round(minSizeMM / pix2mm)
    # closerSize = minSizePx / 2.0
    #
    # # saturation = MorphClose(saturation, closerSize)
    # saturation = BrightnessAndContrastAuto(saturation, 1)
    #
    # ret, bwStones = cv2.threshold(saturation, thStone, 255, cv2.THRESH_BINARY_INV)
    # # cv2.imshow("threshold", bwStones)
    #
    # # contours_stone, hierarchy = cv2.findContours(bwStones, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # c = max(contours_stone, key=cv2.contourArea)
    # cv2.drawContours(saturation, contours_stone, -1, (0, 0, 255), 2)
    # # cv2.imshow("Threshold Stones+Shadow on Saturation", saturation)
    #
    # # if removeShadow:
    # # brightness = MorphClose(brightness, closerSize)
    # brightness = BrightnessAndContrastAuto(brightness, 1)
    #
    # ret, bwShadow = cv2.threshold(brightness, thShadow, 255, cv2.THRESH_BINARY)
    # # contours.clear()
    #
    # bwShadow[:k, :] = 255
    # bwShadow[bwStones.shape[0] - k:, k:] = 255
    # bwShadow[:, :k] = 255
    # bwShadow[k:, bwShadow.shape[1] - k:] = 255
    #
    #
    # contours_shadow, hierarchy = cv2.findContours(bwShadow, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(brightness, contours_shadow, -1, (0, 0, 255), 2)
    # # cv2.imshow("Threshold Shadow on Brightness", brightness)
    #
    # cv2.bitwise_not(bwShadow, bwStones, bwStones)
    # # cv2.imshow("bwStones", bwStones)

    dst = image.copy()
    # contours, hierarchy = cv2.findContours(bwStones, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(dst, contours_stone + contours_shadow, -1, (255, 255, 255), 2)
    # c = max(cont, key=cv2.contourArea)
    cv2.drawContours(dst, cont, -1, (0, 255, 0), 2)

    # for i in range(len(contours)):
    #     cv2.polylines(dst, contours[i], True, (0, 255, 0), 2)

    # cv2.imshow(winName, dst)
    cv2.imwrite('result/' + img.name, dst)


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    # files_grabbed = [file[:-4] for file in files_grabbed if 'orig' in file.lower()]
    return files_grabbed


if __name__ == '__main__':
    knowDistancePX = 170
    knowDistanceMM = 150
    pix2mm = knowDistancePX / knowDistanceMM

    image = cv2.imread('data/stone4.jpg')
    # pad_size = 8
    # image = cv2.copyMakeBorder(
    #     image, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # set defaults
    minSizeMM = 80  # width in millimetre
    thStone = 100  # max saturation for stones
    thShadow = 128  # max brightness for shadow
    removeShadow = 1  # try to remove shadows (1=Yes 0=No)

    dir = 'data'
    for subdir, dirs, files in os.walk(dir):
        for folder in dirs:
            p = os.path.join(dir, folder) + os.path.sep
            images = get_all_files_in_folder(Path(p), ['*.jpg'])

            for img in images:
                onStonesTb(img)

    # winName = 'Stones'
    # cv2.imshow(winName, image)
    # # cv2.createTrackbar("Size(mm)", winName, minSizeMM, 100, onStonesTb)
    # # cv2.createTrackbar("thStone", winName, thStone, 255, onStonesTb)
    # # cv2.createTrackbar("thShadow", winName, thShadow, 255, onStonesTb)
    # # cv2.createTrackbar("remove Shadow", winName, removeShadow, 1, onStonesTb)
    #
    #
    # cv2.waitKey(0)
