#!/usr/bin/env python
# https://gist.github.com/kefir-/03cea3e3b17b7a74a7cdcf57a2159a79
import cv2
c = cv2.imread('images/7.jpg')
height, width = c.shape[:2]
cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('jpg', width, height)
cv2.imshow('jpg', c)
r = cv2.waitKey(0)
print ("DEBUG: waitKey returned:", chr(r))
cv2.destroyAllWindows()