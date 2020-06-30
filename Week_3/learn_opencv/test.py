import cv2
import imutils

image = cv2.imread("original.jpeg")
h, w, d = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

roi = image[0:h, 320:420]
resized = imutils.resize(image, width=300)
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
rotated2 = imutils.rotate(image, -45)

blurred = cv2.GaussianBlur(image, (11, 11), 0)
output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.circle(output, (300, 150), 20, (255,0,0), -1)
# cv2.imshow('Rectangle', output)
# cv2.imshow('blured', blurred)
# cv2.imshow('rotated', rotated2)
# cv2.imshow('resize', resized)
cv2.imshow("Image", image)
cv2.waitKey()

cv2.imshow("roi", roi)
cv2.waitKey()