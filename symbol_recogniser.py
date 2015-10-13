#!/usr/bin/env python

import numpy as np
#import operator
#import math

from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
from threading import RLock
from sensor_msgs.msg import Image

class SymbolRecogniser:
    def __init__(self):
        self.image_sub =  rospy.Subscriber("nao_camera", Image, self.callback)
        #self.moveMaker = rospy.Publisher("", anonymous = True, log_level=rospy.DEBUG)

    def callback(self, data):
        try:
            self.identify_symbol(CvBridge.imgmsg_to_cv2(data, "bgr8"))
        except CvBridgeError, e:
            rospy.logerr(e)

    def identify_symbol(self, im):
        print "Processing image"

def main():
    SymbolRecogniser()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
