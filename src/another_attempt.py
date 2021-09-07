import cv2
import numpy as np


def onStonesTb(self):

    B = cv2.getTrackbarPos('B', 'Stones')
    G = cv2.getTrackbarPos('G', 'Stones')
    R = cv2.getTrackbarPos('R', 'Stones')

    # img = cv2.imread("data/stone1.jpg", cv2.IMREAD_COLOR)
    # convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # set lower and upper color limits
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([B, G, R])
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # remove noise
    kernel = np.ones((1, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((1, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # find contours in mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # select large contours (menu items only)
    # for cnt in contours:
    #     print(cv2.contourArea(cnt))
    #     if cv2.contourArea(cnt) > 500:
    #         # draw a rectangle around the items
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # show image
    # cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('data/stone1.jpg')
    # pad_size = 8
    # image = cv2.copyMakeBorder(
    #     image, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # set defaults
    B = 100  # width in millimetre
    G = 100  # max saturation for stones
    R = 100  # max brightness for shadow
    # removeShadow = 1  # try to remove shadows (1=Yes 0=No)

    winName = 'Stones'
    cv2.imshow(winName, img)
    cv2.createTrackbar("B", winName, B, 255, onStonesTb)
    cv2.createTrackbar("G", winName, G, 255, onStonesTb)
    cv2.createTrackbar("R", winName, R, 255, onStonesTb)
    # cv2.createTrackbar("remove Shadow", winName, removeShadow, 1, onStonesTb)

    onStonesTb(1)
    cv2.waitKey(0)
