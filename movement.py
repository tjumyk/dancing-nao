#!/usr/bin/env python

import rospy
from naoqi_bridge_msgs.msg import BodyPoseActionGoal
from nao_dance.srv import MakeMove

class Move_maker:
    def __init__(self):
        rospy.init_node("movement", anonymous=True)
        self.move_publisher = rospy.Publisher("/body_pose/goal", BodyPoseActionGoal, queue_size=3)
        self.move_service =rospy.Service("make_move", MakeMove, self.handle_make_move)
        self.nextForward = "right"
        self.nextBackward = "right"
        rospy.loginfo("Movement node started")

    def handle_make_move(self, direction):
        move = BodyPoseActionGoal()
        if direction == "forward":
            if self.nextFoward == "right":
                move.goal.pose_name = "rightFootForward"
                self.nextForward = "left"
            else:
                move.goal.pose_name = "leftFootForward"
                self.nextForward = "right"
        elif direction == "backward":
            if self.nextBackward == "right":
                move.goal.pose_name = "rightFootBack"
                self.nextBackward = "left"
            else:
                move.goal.pose_name = "leftFootBack"
                self.nextBackward = "right"
        elif direction == "right":
            move.goal.pose_name = "right"
        elif direction == "left":
            move.goal.pose_name = "left"
        else:
            #back to standing position
            move.goal.pose_name = "stand"
        self.move_publisher.publish(move)
        rospy.loginfo("Move published %s" % move.goal.pose_name)

def main():
    Move_maker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"

if __name__ == '__main__':
    main()