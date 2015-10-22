#!/usr/bin/env python

#_xap:="/home/z5021315/catkin_ws/src/dancing-nao/poses.xap"

import rospy
from naoqi_bridge_msgs.msg import BodyPoseActionGoal

def main():
    rospy.init_node("movement_test", anonymous=True)
    move_publisher = rospy.Publisher("/body_pose/goal", BodyPoseActionGoal, queue_size=1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        move = BodyPoseActionGoal()
        move.goal.pose_name = "leftArmUp"
        move_publisher.publish(move)
        rospy.loginfo("Move published")
        rate.sleep()


if __name__ == '__main__':
    main()


