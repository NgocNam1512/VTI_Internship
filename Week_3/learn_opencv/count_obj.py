import imutils
import cv2

image = cv2.imread('tetris_blocks.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("image", image)
cv2.waitKey(0)

cv2.imshow('gray', gray)
cv2.waitKey(0)

edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edge", edged)
cv2.waitKey(0)

thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
output = image.copy()
# for c in cnts:
#     cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
#     cv2.imshow("output", output)
#     cv2.waitKey(0)
cv2.drawContours(output, cnts, -1, (240, 0, 159), 3)

text = "Found {} objects".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            (240, 0, 159), 2)
cv2.imshow("Contour", output)
cv2.waitKey(0)
