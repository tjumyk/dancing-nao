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
        move.goal.pose_name = "crouch"
        move_publisher.publish(move)
        rospy.loginfo("Move published")
        rate.sleep()
    #rospy.wait_for_service("body_pose")
    #try:
    #    execute_pose = rospy.ServiceProxy("body_pose", BodyPoseActionGoal)
    #    move = BodyPoseActionGoal()
    #    move.goal.pose_name = "leftArmUp"
    #    execute_pose(move)
    #    rospy.loginfo("Service call made")
    #except rospy.ServiceException, e:
    #    print ("Service call failed: %s"%e)


if __name__ == '__main__':
    main()


