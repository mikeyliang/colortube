import cv2
import imutils
import numpy as np

img = cv2.imread('images/tubecolor.jpeg')

original = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours[0], key=cv2.contourArea, reverse= True)

rects = []

maxc = cv2.contourArea(contours[0])/100

# Get Triangle Contours
for cnt in contours:
    p = cv2.arcLength(cnt, True)
    epsilon = 0.01*p # Tubes -> Rectangle
    poly = cv2.approxPolyDP(cnt, epsilon, True)
    rect = cv2.boundingRect(poly)
    # if len(poly) == 3:
    if rect[2] * rect[3] > maxc:
        rects.append(rect)

    

# area = [a/max(area) for a in area]

detect_thresh = 0.8




padding = 20

#yTop = tubes[:, :, :, 1].min() - padding # Top Extrema y value
#yBottom = tubes[:, :, :, 1].max() + padding # Bottom Extrema y value

#xLeft = tubes[:, :, :, 0].min() - padding # Left Extrema x value
#xRight = tubes[:, :, :, 0].max() + padding # Right Extrema x value

# original = cv2.rectangle(original, (xLeft, yTop), (xRight, yBottom), (0, 255, 0), 4)

# cv2.drawContours(original, polys, -1, (0, 255, 0), 3)

for rect in rects:
    cv2.rectangle(original, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,0 ,0), 5)

cv2.imshow('Tube', original)
cv2.waitKey()




