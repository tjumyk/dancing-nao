import cv2

cap = cv2.VideoCapture('res/demo1.avi')


def nothing(x):
    pass

cv2.namedWindow('frame')
cv2.createTrackbar('canny min', 'frame', 100, 1000, nothing)
cv2.createTrackbar('canny max', 'frame', 200, 1000, nothing)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cv2.getTrackbarPos('canny min', 'frame'), cv2.getTrackbarPos('canny max', 'frame'))
    cv2.imshow('frame', edges)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
