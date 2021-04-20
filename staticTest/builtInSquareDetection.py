import cv2 as cv
import imutils
from squares import find_squares

img = cv.imread('8.jpeg')
squares = find_squares(img)

cv.drawContours(img, squares, -1, (0, 255, 0), 3)
res = imutils.resize(img, width=300)
cv.imshow('squares', res)
cv.waitKey(0)
