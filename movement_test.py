#!/usr/bin/env python

import rospy
from naoqi_bridge_msgs.msg import BodyPoseActionGoal

def main():
    rospy.init_node()
    move_publisher = rospy.Publisher("naoqi_msgs/BodyPose", BodyPoseActionGoal, queue_size=5)
    move_publisher.publish("LeftArmUp")
    rospy.spin()

if __name__ == '__main__':
    main()


