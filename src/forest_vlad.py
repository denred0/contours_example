import os
import cv2
import shutil

import numpy as np

from tqdm import tqdm

img_dir = "forest/img/"
out_dir = "forest/annot_thresh_4/"

images_list = sorted(os.listdir(img_dir))

max_threshold = 255
num_thresholding_stages = 5
step_thresholding_stage = 20
min_thresh_list = [20 * x for x in range(num_thresholding_stages)]

mean_pixel_of_removable_area = 40.0
mask_dilation_size = 3
mask_dilation_shape = 2

for i in tqdm(images_list):

    img_path = "{}{}".format(img_dir, i)
    img_src = cv2.imread(img_path)
    img_src_init = img_src.copy()
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(img_src_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_src_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    ret, thresh = cv2.threshold(grad , 10, 255, cv2.THRESH_BINARY)

    thresh[0, :] = 255
    thresh[:, 0] = 255
    thresh[1023, :] = 255
    thresh[:, 1023] = 255

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours = [c for c in contours if 1000000 > cv2.contourArea(c) > 500]

    for cnt in contours:

        mask = np.zeros(img_src_gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        cnt_mean_pixel_value = cv2.mean(img_src_gray, mask=mask)[0]

        if cnt_mean_pixel_value <= mean_pixel_of_removable_area:

            kernel = np.ones((3,3),np.uint8)

            dilated_mask = cv2.dilate(mask, kernel, iterations=7)

            water_pixels = np.where(dilated_mask == 255)
            img_src[water_pixels[0], water_pixels[1], :] = (255, 255, 255)

    average_img_src = np.average(img_src, weights=(img_src < 255))
    ret_final, thresh_final = cv2.threshold(img_src , average_img_src, max_threshold, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (2, 2))
    mask_denoise = cv2.morphologyEx(thresh_final, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_eroded = cv2.erode(mask_denoise, kernel, iterations = 5)

    mask_eroded_end_source = np.hstack((img_src_init, mask_eroded))


    # thresh_final[thresh_final == 255] = 0
    # thresh_final[thresh_final == 0] = 1

    out_path = "{}{}".format(out_dir, i)
    cv2.imwrite(out_path, mask_eroded_end_source)