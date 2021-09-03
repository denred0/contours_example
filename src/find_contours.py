import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

start_time = time.time()

image_name = 'stone1.jpg'

image = cv2.imread('data/' + image_name)
im_orig = image.copy()
# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Result", imgray)
# cv2.waitKey(0)

blur = cv2.GaussianBlur(image, (0, 0), 2, 2)
cv2.imshow("Result", blur)
cv2.waitKey(0)

hsv_image =

# average = imgray.mean(axis=0).mean(axis=0)

# pixels = np.float32(im.reshape(-1, 3))
#
# n_colors = 5
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
# flags = cv2.KMEANS_RANDOM_CENTERS
#
# _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
# _, counts = np.unique(labels, return_counts=True)
#
# dominant = palette[np.argmax(counts)]

# avg_patch = np.ones(shape=im.shape, dtype=np.uint8) * np.uint8(average)

# indices = np.argsort(counts)[::-1]
# freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
# rows = np.int_(im.shape[0] * freqs)
#
# dom_patch = np.zeros(shape=im.shape, dtype=np.uint8)
# for i in range(len(rows) - 1):
#     dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
#
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# ax0.imshow(avg_patch)
# ax0.set_title('Average color')
# ax0.axis('off')
# ax1.imshow(dom_patch)
# ax1.set_title('Dominant colors')
# ax1.axis('off')
# plt.show()

threshold = np.mean(average) * 0.5

ret, thresh = cv2.threshold(blur, np.mean(average) * 1, 255, 0)
cv2.imshow("Result", thresh)

cv2.waitKey(0)

# canny_output = cv2.Canny(thresh, threshold, threshold * 2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) != 0:
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(im, c, -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw in blue the contours that were founded

    # find the biggest countour (c) by the area

    # x,y,w,h = cv2.boundingRect(c)

    # cv2.drawContours(im, c, -1, (0, 255, 0), 3)

    # draw the biggest contour (c) in green
    # cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
cv2.imshow("Result", np.hstack([im_orig, im]))

cv2.waitKey(0)

# cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
print("--- %s seconds ---" % (time.time() - start_time))

cv2.imwrite('result/' + image_name, np.hstack([im_orig, im]))
