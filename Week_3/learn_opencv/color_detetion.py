import cv2
import numpy as np

image = cv2.imread('color_detection_red_version.jpg')
boundaries = [
    ([17, 15, 100], [50, 56, 200]), #red
    ([86, 31, 4], [220, 88, 50]), #blue
    ([25, 146, 190], [62, 174, 250]), #yellow
    ([103, 86, 65], [145, 133, 128]) #gray
]

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("image", np.hstack([image, output]))
    cv2.waitKey(0)
