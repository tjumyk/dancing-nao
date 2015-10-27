#!/usr/bin/env python

import rospy
from naoqi_bridge_msgs.msg import BodyPoseWithSpeedGoal
from nao_dance.srv import *

class Move_maker:
    def __init__(self):
        rospy.init_node("movement", anonymous=True)
        self.move_publisher = rospy.Publisher("/body_pose/goal", BodyPoseWithSpeedGoal, queue_size=10)
        self.move_service =rospy.Service("make_move", MakeMove, self.handle_make_move)
        self.nextForward = "right"
        self.nextBackward = "right"
        rospy.loginfo("Movement node started")

    def handle_make_move(self, request):
        """
        :type request MakeMoveRequest
        """
        direction = request.direction
        move = BodyPoseWithSpeedGoal()
        if direction == "forward":
            if self.nextForward == "right":
                move.posture_name = "rightFootForward"
                self.nextForward = "left"
            else:
                move.posture_name = "leftFootForward"
                self.nextForward = "right"
        elif direction == "backward":
            if self.nextBackward == "right":
                move.posture_name = "rightFootBack"
                self.nextBackward = "left"
            else:
                move.posture_name = "leftFootBack"
                self.nextBackward = "right"
        elif direction == "right":
            move.posture_name = "right"
        elif direction == "left":
            move.posture_name = "left"
        else:
            #back to standing position
            move.posture_name = "Stand"
        move.speed = 1
        self.move_publisher.publish(move)
        rospy.loginfo("Move published %s" % move.posture_name)
        return MakeMoveResponse()

def main():
    Move_maker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"

if __name__ == '__main__':
    main()