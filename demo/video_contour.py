import numpy as np

import cv2

cap = cv2.VideoCapture('res/demo1.avi')
cv2.namedWindow('frame')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    cv2.imshow('frame', mask)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
