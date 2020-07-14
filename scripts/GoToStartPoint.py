#!/usr/bin/env python
#  __________________________________________________________________________
# | This script simply takes us to (0, 0, 1) before starting the simulation. |
# |                                                                          |
# |__________________________________________________________________________|
#
from utils import *
import sys

rospy.init_node('GoToStart') 

rospy.sleep(2)

start_point = [5, 0, 2.0]

pose_msg = xyz_2_Pose(start_point)

PosePublisher = rospy.Publisher(
    '/elios_VLP16/goal', Pose, queue_size=1,
)

def pose_callback(msg):

    dist = euclidean_dist(pose_2_np_arr_xyz(msg), start_point)

    if dist < 0.1:
        rospy.signal_shutdown("Now at starting point.")
    else:
        # print('\nMoving to starting position (0, 0, 1)\n')
        PosePublisher.publish(pose_msg)

    return


rospy.Subscriber(
    '/elios_VLP16/ground_truth/pose', Pose, pose_callback, 
)

rospy.spin()

