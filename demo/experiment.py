import numpy as np

import cv2

debugging = False

cap = None
bg_sub = None
arrow_cnt = None

pause = False
wait_time = 50
frame_delay = 0
frame_count = 0
saturation_split = 145


def nothing(x):
    pass


cv2.namedWindow('edge')
cv2.createTrackbar('canny min', 'edge', 100, 1000, nothing)
cv2.createTrackbar('canny max', 'edge', 200, 1000, nothing)
cv2.namedWindow('result')
cv2.createTrackbar('diff% max', 'result', 20, 100, nothing)
cv2.createTrackbar('area min', 'result', 300, 1000, nothing)
font = cv2.FONT_HERSHEY_SIMPLEX


def remove_duplicates(hierarchy, targets, index, is_root=True):
    first_child_index = hierarchy.item(0, index, 2)
    current_index = first_child_index
    dup_count = 0
    while current_index >= 0:
        dup_count += remove_duplicates(hierarchy, targets, current_index, False)
        current_index = hierarchy.item(0, current_index, 0)
    if not is_root and index in targets:
        targets.remove(index)
        dup_count += 1
    return dup_count


def handle_frame(frame):
    frame = cv2.pyrDown(frame)
    fg_mask = bg_sub.apply(frame)
    # cv2.imshow('mask_original', fg_mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow('mask_optimized', fg_mask)
    gray_m = cv2.bitwise_and(gray, gray, mask=fg_mask)
    # cv2.imshow('gray_m', gray_m)
    edges = cv2.Canny(gray_m, cv2.getTrackbarPos('canny min', 'edge'), cv2.getTrackbarPos('canny max', 'edge'))
    cv2.imshow('edge', edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter based on shape and area
    diff_max = cv2.getTrackbarPos('diff% max', 'result') / 100.0
    area_min = cv2.getTrackbarPos('area min', 'result')
    candidate_indexes = []
    for ind, cnt in enumerate(contours):
        diff = cv2.matchShapes(cnt, arrow_cnt, 1, 0.0)
        if diff > diff_max:
            continue
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        candidate_indexes.append(ind)

    # for ind in candidate_indexes:
    #     print '[', ind, '] -->', hierarchy[0, ind, :]
    # remove redundant candidates
    # print 'Before cleaning duplicates:', len(candidate_indexes)
    dup_sum = 0
    for ind in candidate_indexes:
        dup_sum += remove_duplicates(hierarchy, candidate_indexes, ind)
    # print 'Duplicates:', dup_sum
    # print 'After cleaning duplicates:', len(candidate_indexes)

    # classification
    slots = []
    arrows = []
    for ind in candidate_indexes:
        contour_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(contour_mask, [contours[ind]], 0, 255, -1)
        color = cv2.mean(frame, mask=contour_mask)
        hsv = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_BGR2HSV)[0][0]
        if hsv[1] >= saturation_split:
            arrows.append(ind)
        else:
            slots.append(ind)

    for ind in slots:
        x, y, w, h = cv2.boundingRect(contours[ind])
        cv2.putText(frame, str(ind), (x, y-2), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for ind in arrows:
        x, y, w, h = cv2.boundingRect(contours[ind])
        cv2.putText(frame, str(ind), (x, y-2), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print 'slots:', len(slots), ', arrows:', len(arrows), '(diff<=', diff_max, ', area>=', area_min, ')'
    cv2.imshow('result', frame)


def init():
    global cap, bg_sub, arrow_cnt
    cap = cv2.VideoCapture('res/demo1.avi')
    bg_sub = cv2.BackgroundSubtractorMOG2()
    arrow = cv2.pyrDown(cv2.imread('res/arrow.png', 0))
    ret, thresh = cv2.threshold(arrow, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arrow_cnt = contours[0]
    # canvas = np.zeros(arrow.shape, np.uint8)
    # cv2.drawContours(canvas, contours, 0, 255, 2)
    # cv2.imshow('arrow', canvas)


def start():
    global pause, wait_time, frame_count, debugging
    init()
    frame_count = 0
    while 1:
        if not pause:
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break
            if frame_count < frame_delay:
                continue
            handle_frame(frame)
        k = cv2.waitKey(wait_time) & 0xff
        if k == 27:
            break
        elif k == 32:
            pause = not pause
        elif k == ord(','):
            wait_time += 25
            wait_time = min(200, wait_time)
        elif k == ord('.'):
            wait_time -= 25
            wait_time = max(25, wait_time)
        elif k == ord('d'):
            debugging = not debugging
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()
