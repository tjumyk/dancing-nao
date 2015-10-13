__author__ = 'kelvin'

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image

image_topic = '/phone1/camera/image/raw'
compressed_image_topic = '/phone1/camera/image/compressed'
use_compression = True
bridge = None


def handle_compressed_image(data):
    """
    :type data: CompressedImage
    """
    frame = cv2.imdecode(np.fromstring(data.data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
    handle_frame(frame)


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


def handle_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('original', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', res)
    cv2.waitKey(3)


# noinspection PyTypeChecker
def start_node():
    global bridge
    rospy.init_node('color_extraction')
    if use_compression:
        rospy.Subscriber(compressed_image_topic, CompressedImage, handle_compressed_image)
    else:
        bridge = CvBridge()
        rospy.Subscriber(image_topic, Image, handle_image)
    rospy.loginfo("Node started")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_node()


