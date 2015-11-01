import cv2

cap = cv2.VideoCapture('res/demo.mp4')

bg_sub = cv2.BackgroundSubtractorMOG()

while 1:
    ret, frame = cap.read()
    if not ret:
        break
    fg_mask = bg_sub.apply(frame)
    cv2.imshow('original', frame)
    cv2.imshow('mask', fg_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
