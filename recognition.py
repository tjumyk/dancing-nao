#!/usr/bin/env python
# coding=utf-8

__author__ = 'kelvin'

import numpy as np
import heapq
import math
from cv_bridge import CvBridge, CvBridgeError
import time
import rospkg
from threading import Timer

import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from nao_dance.srv import *


# ROS related variables
image_topic = '/nao_robot/camera/top/camera/image_raw'
compressed_image_topic = '/phone1/camera/image/compressed'
make_move_service_name = '/make_move'
make_move_service = None
use_compression = False
bridge = None
package_path = None

# local test variables
cap = None
debugging = False
pause = False
wait_time = 50
frame_delay = 0
frame_count = 0

# common variables
bg_sub = None
arrow_cnt = None
max_frame_size = 640 * 360
slot_directions = ['L', 'D', 'U', 'R']
make_move_commands = ['left', 'backward', 'forward', 'right']
slot_angle_threshold = math.pi / 4.0
slot_dist_percent_threshold = 0.3
font = cv2.FONT_HERSHEY_SIMPLEX
move_queues = [[], [], [], []]
move_queue_timestamp = None
move_translate_speed_ratio = 6.0  # times average height of arrow, per second
move_duplicate_threshold = 0.8  # times average height of arrow


def nothing(_):
    pass


def init():
    global bg_sub, arrow_cnt, package_path, make_move_service
    rospy.init_node('color_extraction')
    rospy.wait_for_service(make_move_service_name)
    make_move_service = rospy.ServiceProxy(make_move_service_name, MakeMove)
    bg_sub = cv2.BackgroundSubtractorMOG2()
    ros_pack = rospkg.RosPack()
    package_path = ros_pack.get_path('nao_dance')
    arrow = cv2.pyrDown(cv2.imread(package_path + '/res/arrow.png', 0))
    ret, thresh = cv2.threshold(arrow, 50, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arrow_cnt = contours[0]
    # canvas = np.zeros(arrow.shape, np.uint8)
    # cv2.drawContours(canvas, contours, 0, 255, 2)
    # cv2.imshow('arrow', canvas)
    cv2.namedWindow('edge')
    cv2.createTrackbar('canny min', 'edge', 100, 1000, nothing)
    cv2.createTrackbar('canny max', 'edge', 200, 1000, nothing)
    cv2.namedWindow('result')
    cv2.createTrackbar('diff% max', 'result', 20, 100, nothing)
    cv2.createTrackbar('area min', 'result', 300, 1000, nothing)
    cv2.createTrackbar('saturation split', 'result', 145, 255, nothing)
    cv2.namedWindow('contours')
    cv2.createTrackbar('approx', 'contours', 34, 100, nothing)


def handle_frame(frame):
    update_move_queue()
    while frame.shape[0] * frame.shape[1] > max_frame_size:
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

    bounding_rectangles = {}
    for ind in candidate_indexes:
        bounding_rectangles[ind] = cv2.boundingRect(contours[ind])

    arrows, slots = classify_contours(candidate_indexes, contours, frame)
    for ind in slots:
        x, y, w, h = bounding_rectangles[ind]
        cv2.putText(frame, str(ind), (x, y - 2), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for ind in arrows:
        x, y, w, h = bounding_rectangles[ind]
        cv2.putText(frame, str(ind), (x, y - 2), font, 0.5, (255, 255, 255), 1, cv2.CV_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print 'slots:', len(slots), ', arrows:', len(arrows), '(diff<=', diff_max, ', area>=', area_min, ')'

    directions = direction_recognition(candidate_indexes, contours, frame)
    check_new_move(arrows, slots, directions, bounding_rectangles)
    cv2.imshow('result', frame)


def send_stand():
    send_move_command('stand')


def send_move_command(command):
    request = MakeMoveRequest()
    request.direction = command
    try:
        make_move_service(request)
    except Exception, e:
        rospy.logwarn(e)


def make_move(direction):
    print 'Move:', slot_directions[direction], time.time()
    if make_move_service is not None:
        send_move_command(make_move_commands[direction])
        Timer(0.1, send_stand).start()


def update_move_queue():
    global move_queue_timestamp
    now = time.time()
    if move_queue_timestamp is None:
        move_queue_timestamp = now
        return
    translate = move_translate_speed_ratio * (now - move_queue_timestamp)
    for i in range(4):
        queue = move_queues[i]
        invalidated = []
        for j in range(len(queue)):
            move = queue[j]
            # noinspection PyTypeChecker
            new_move = move - translate
            if new_move <= 0 <= move:
                make_move(i)
            if new_move <= -2:
                invalidated.append(new_move)
            queue[j] = new_move
        for move in invalidated:
            queue.remove(move)
    move_queue_timestamp = now


def check_new_move(arrows, slots, directions, bounding_rectangles):
    if len(slots) < 2:
        # print 'slots less than 2'
        return
    slots_in_order = [None, None, None, None]
    for slot in slots:
        if slot in directions:
            direction = directions[slot]
            old_rect = slots_in_order[direction]
            new_rect = bounding_rectangles[slot]
            if old_rect is not None and old_rect[1] <= new_rect[1]:  # compare y position, pick smaller one
                continue
            slots_in_order[direction] = new_rect
    width_list = []
    height_list = []
    for i in range(4):
        rect = slots_in_order[i]
        if rect is not None:
            width_list.append(rect[2])
            height_list.append(rect[3])
    if len(width_list) < 2:
        # print 'slot directions less than 2'
        return
    avg_width = sum(width_list) / len(width_list)
    avg_height = sum(height_list) / len(height_list)
    dist_min = avg_width * (1 - slot_dist_percent_threshold)
    dist_max = avg_width * (1 + slot_dist_percent_threshold)
    valid_slot_combinations = []
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:  # iterate over all possible combinations
        slot_a = slots_in_order[a]
        slot_b = slots_in_order[b]
        if slot_a is not None and slot_b is not None:
            dy = slot_b[1] - slot_a[1]
            dx = slot_b[0] - slot_a[0]
            angle = math.atan2(dy, dx)
            dist = math.sqrt(dx * dx + dy * dy) / (b - a)
            if -slot_angle_threshold <= angle <= slot_angle_threshold and dist_min <= dist <= dist_max:
                valid_slot_combinations.append((a, b, angle, dist))
            else:
                # print 'invalid combination:', a, b, angle, dist
                pass
    angles = []
    dists = []
    valid_slots = [None, None, None, None]
    valid_comb_count = len(valid_slot_combinations)
    if valid_comb_count == 0:
        # print 'valid slot directions less than 2'
        return
    take_combs = valid_comb_count
    if valid_comb_count == 2:  # two exclusive combinations, take first one
        take_combs = 1
    for i in range(take_combs):
        a, b, angle, dist = valid_slot_combinations[i]
        valid_slots[a] = slots_in_order[a]
        valid_slots[b] = slots_in_order[b]
        angles.append(angle)
        dists.append(dist)
    # print valid_slots
    avg_angle = sum(angles) / len(angles)
    avg_dist = sum(dists) / len(dists)
    offset_x = avg_dist * math.cos(avg_angle)
    offset_y = avg_dist * math.sin(avg_angle)

    base_point = None
    first_slot = valid_slots[0]
    if first_slot is not None:
        base_point = (first_slot[0], first_slot[1])
    else:
        for i in range(1, 4):
            slot = valid_slots[i]
            if slot is not None:
                dx = i * offset_x
                dy = i * offset_y
                base_point = (slot[0] - dx, slot[1] - dy)
                break
    # print "base point:", base_point, "angle:", avg_angle, "distance:", avg_dist

    for arrow in arrows:
        if arrow not in directions:
            continue
        direction = directions[arrow]
        direction_base_point = (base_point[0] + direction * offset_x, base_point[1] - direction * offset_y)
        rect = bounding_rectangles[arrow]
        dx = rect[0] - direction_base_point[0]
        dy = rect[1] - direction_base_point[1]
        angle = math.atan2(dy, dx) - math.pi / 2.0
        delta_angle = angle - avg_angle
        dist = math.sqrt(dx * dx + dy * dy)
        delta = dist * math.sin(delta_angle)
        if abs(delta) > avg_dist * slot_dist_percent_threshold:
            # print 'wrong direction:', slot_directions[direction], dist
            continue
        relative_dist = dist / avg_height
        # print slot_directions[direction], relative_dist, time.time()
        is_new_move = True
        for move in move_queues[direction]:
            # noinspection PyTypeChecker
            if abs(move - relative_dist) <= move_duplicate_threshold:
                is_new_move = False
                break
        if is_new_move:
            # print slot_directions[direction], relative_dist, time.time()
            # noinspection PyTypeChecker
            move_queues[direction].append(relative_dist)


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


def classify_contours(candidate_indexes, contours, frame):
    slots = []
    arrows = []
    saturation_split = cv2.getTrackbarPos('saturation split', 'result')
    for ind in candidate_indexes:
        contour_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(contour_mask, [contours[ind]], 0, 255, -1)
        color = cv2.mean(frame, mask=contour_mask)
        hsv = cv2.cvtColor(np.uint8([[list(color)]]), cv2.COLOR_BGR2HSV)[0][0]
        if hsv[1] >= saturation_split:
            arrows.append(ind)
        else:
            slots.append(ind)
    return arrows, slots


def direction_recognition(candidate_indexes, contours, frame):
    img_contours = np.zeros(frame.shape[:2], np.uint8)
    directions = {}
    for ind in candidate_indexes:
        cnt = contours[ind]
        epsilon = cv2.getTrackbarPos('approx', 'contours') / 1000.0 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        # cv2.putText(img_contours, str(approx.shape[0]), (x, y - 20), font, 0.5, 255, 1, cv2.CV_AA)
        cv2.drawContours(img_contours, [approx], 0, 255, 1, cv2.CV_AA)
        hull = cv2.convexHull(approx, returnPoints=False)
        if approx.shape[0] <= 3:
            # print 'Failed to detect arrow direction for ', ind
            continue
        defects = cv2.convexityDefects(approx, hull)
        if defects is None or defects.shape[0] < 2:
            # print 'Failed to detect arrow direction for ', ind
            continue
        heap = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start_p = tuple(approx[s][0])
            end_p = tuple(approx[e][0])
            dx = start_p[0] - end_p[0]
            dy = start_p[1] - end_p[1]
            area = d * (dx * dx + dy * dy)
            heapq.heappush(heap, (area, i))
        far_points = []
        line_center_points = []
        for i in range(2):  # range(len(heap)):
            s, e, f, d = defects[heap[-i - 1][1], 0]
            start_p = tuple(approx[s][0])
            end_p = tuple(approx[e][0])
            far = tuple(approx[f][0])
            far_points.append(far)
            line_center_points.append(((start_p[0] + end_p[0]) / 2.0, (start_p[1] + end_p[1]) / 2.0))
            # cv2.line(img_contours, start_p, end_p, 50, 2, cv2.CV_AA)
            # cv2.circle(img_contours, far, 3, 125, -1, cv2.CV_AA)
        angle = abs(math.atan2(far_points[0][1] - far_points[1][1], far_points[0][0] - far_points[1][0]))
        avg_line_center = ((line_center_points[0][0] + line_center_points[1][0]) / 2.0,
                           (line_center_points[0][1] + line_center_points[1][1]) / 2.0)
        if math.pi / 4 <= angle < math.pi * 3 / 4:
            if avg_line_center[0] >= (far_points[0][0] + far_points[1][0]) / 2:
                directions[ind] = 0  # Left
            else:
                directions[ind] = 3  # Right
        else:
            if avg_line_center[1] >= (far_points[0][1] + far_points[1][1]) / 2:
                directions[ind] = 2  # Up
            else:
                directions[ind] = 1  # Down
        cv2.putText(img_contours, slot_directions[directions[ind]], (x, y - 2), font, 0.5, 255, 1, cv2.CV_AA)
    cv2.imshow('contours', img_contours)
    return directions


def handle_compressed_image(data):
    """
    :type data: CompressedImage
    """
    frame = cv2.imdecode(np.fromstring(data.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
    handle_frame(frame)
    cv2.waitKey(1)


def handle_image(data):
    """
    :type data: Image
    """
    try:
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
        rospy.logwarn(e)
        return
    handle_frame(frame)
    cv2.waitKey(1)


# noinspection PyTypeChecker
def start_camera():
    global bridge
    init()
    if use_compression:
        rospy.Subscriber(compressed_image_topic, CompressedImage, handle_compressed_image)
    else:
        bridge = CvBridge()
        rospy.Subscriber(image_topic, Image, handle_image)
    rospy.loginfo("Recognition node started")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node")
    cv2.destroyAllWindows()


def start_local_test(video_file):
    global cap, pause, wait_time, frame_count, debugging
    # noinspection PyArgumentList
    cap = cv2.VideoCapture(video_file)
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
    if len(sys.argv) > 1 and sys.argv[1].endswith('.avi'):
        start_local_test(sys.argv[1])
    else:
        start_camera()
