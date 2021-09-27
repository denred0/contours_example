import os
import cv2
import shutil

import numpy as np

from tqdm import tqdm

img_dir = "forest/img/"
out_dir = "forest/thresh/"

images_list = sorted(os.listdir(img_dir))

max_threshold = 255
num_thresholding_stages = 5
step_thresholding_stage = 20
min_thresh_list = [20 * x for x in range(num_thresholding_stages)]

mean_pixel_of_removable_area = 35.0
max_std_of_removable_area = 10.0

mask_dilation_size = 3
mask_dilation_shape = 2

resize_size = 512

image = cv2.imread('forest/img/000000.png', cv2.IMREAD_COLOR)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_src_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# cv2.imshow('1', np.hstack((cv2.resize(image, (resize_size, resize_size)), cv2.resize(img_src_denoised, (resize_size, resize_size)))))
# cv2.waitKey()

lower_dark = np.array([0, 0, 0])
upper_dark = np.array([30, 30, 30])

img_src_inrange = cv2.inRange(img_src_denoised, lower_dark, upper_dark)

contours, hierarchy = cv2.findContours(img_src_inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = [c for c in contours if 1020000 > cv2.contourArea(c)]

for cnt in contours:
    if cv2.contourArea(cnt) < 200:
        cv2.drawContours(img_src_inrange, [cnt], -1, 0, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dilated = cv2.dilate(img_src_inrange, kernel, iterations=7)

mask_dilated[0, :] = 255
mask_dilated[:, 0] = 255
mask_dilated[1023, :] = 255
mask_dilated[:, 1023] = 255

contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = [c for c in contours if 1020000 > cv2.contourArea(c)]


for cnt in contours:
    mask = np.zeros(image_gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    cnt_mean_all_channels, cnt_std_all_channels = cv2.meanStdDev(img_src_denoised, mask=mask)
    cnt_mean_pixel_value = cnt_mean_all_channels[0][0]
    cnt_std_pixel_value = cnt_std_all_channels[0][0]

    if cnt_mean_pixel_value > mean_pixel_of_removable_area and cnt_std_pixel_value < max_std_of_removable_area:
        cv2.drawContours(mask_dilated, [cnt], -1, 0, -1)

cv2.imshow('1', np.hstack((cv2.resize(image_gray, (resize_size, resize_size)),
                           cv2.resize(img_src_inrange, (resize_size, resize_size)),
                           cv2.resize(mask_dilated, (resize_size, resize_size))
                           )))
cv2.waitKey()

#
# for i in tqdm(images_list):
#
#     img_path = "{}{}".format(img_dir, i)
#     img_src = cv2.imread(img_path)
#
#     img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
#     img_src_denoised = cv2.fastNlMeansDenoisingColored(img_src, None, 10, 10, 7, 21)
#
#     lower_dark = np.array([0, 0, 0])
#     upper_dark = np.array([30, 30, 30])
#
#     img_src_inrange = cv2.inRange(img_src_denoised, lower_dark, upper_dark)
#
#     contours, hierarchy = cv2.findContours(img_src_inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     contours = [c for c in contours if 1020000 > cv2.contourArea(c)]
#
#     for cnt in contours:
#
#         if cv2.contourArea(cnt) < 200:
#             cv2.drawContours(img_src_inrange, [cnt], -1, 0, -1)
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask_dilated = cv2.dilate(img_src_inrange, kernel, iterations=7)
#
#     mask_dilated[0, :] = 255
#     mask_dilated[:, 0] = 255
#     mask_dilated[1023, :] = 255
#     mask_dilated[:, 1023] = 255
#
#     contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     contours = [c for c in contours if 1020000 > cv2.contourArea(c)]
#
#     for cnt in contours:
#
#         mask = np.zeros(img_src_gray.shape, np.uint8)
#         cv2.drawContours(mask, [cnt], -1, 255, -1)
#
#         cnt_mean_all_channels, cnt_std_all_channels = cv2.meanStdDev(img_src_denoised, mask=mask)
#         cnt_mean_pixel_value = cnt_mean_all_channels[0][0]
#         cnt_std_pixel_value = cnt_std_all_channels[0][0]
#
#         if cnt_mean_pixel_value > mean_pixel_of_removable_area and cnt_std_pixel_value < max_std_of_removable_area:
#             cv2.drawContours(mask_dilated, [cnt], -1, 0, -1)
#
#         #     print("\nSTD: {}".format(cnt_std_pixel_value))
#         #     print("MEAN: {}".format(cnt_mean_pixel_value))
#
#         # else:
#
#         #     text_std = "STD: {:.3f}".format(cnt_std_pixel_value)
#         #     text_mean = "MEAN: {:.3f}".format(cnt_mean_pixel_value)
#
#         #     fontType  = cv2.FONT_HERSHEY_SIMPLEX
#         #     fontScale = 0.6
#
#         #     M = cv2.moments(cnt)
#         #     cX = int(M["m10"] / M["m00"])
#         #     cY = int(M["m01"] / M["m00"])
#
#         #     cv2.putText(
#         #         img_src_denoised,
#         #         text_std,
#         #         (cX, cY),
#         #         fontType,
#         #         fontScale,
#         #         (0, 255, 0)
#         #         )
#
#         #     cv2.putText(
#         #         img_src_denoised,
#         #         text_mean,
#         #         (cX, cY + 17),
#         #         fontType,
#         #         fontScale,
#         #         (0, 255, 0)
#         #         )
#
#     water_pixels = np.where(mask_dilated == 255)
#     img_src_denoised[water_pixels[0], water_pixels[1], :] = (255, 255, 255)
#
#     min_threshold = cv2.mean(img_src_denoised, mask=(255 - mask_dilated))[0]
#     ret_final, thresh_final = cv2.threshold(img_src_denoised, min_threshold, max_threshold, cv2.THRESH_BINARY)
#
#     out_path = "{}{}".format(out_dir, i)
#     cv2.imwrite(out_path, thresh_final)
#
#     out = cv2.hconcat([img_src, img_src_denoised, thresh_final])
#     cv2.imshow("res", cv2.resize(out, (1800, 720)))
#     cv2.waitKey()
#     cv2.destroyAllWindows()
