#!/usr/bin/env python
# This script simply takes us to (0, 0, 1) before starting the simulation.
from utils import *

rospy.init_node('GoToStart') 

rospy.sleep(3)

# RateThrustPublisher = rospy.Publisher(
#     '/elios_VLP16/command/rate_thrust', RateThrust, queue_size=45
# )

# first_act_msg = acc_to_RateThrust([0, 0, 1])

pose_msg = xyz_2_Pose([0, 0, 1])

PosePublisher = rospy.Publisher(
    '/elios_VLP16/goal', Pose, queue_size=45,
)

print('\n\n\nMoving to starting position (0, 0, 1)\n\n\n')
    
# if not rospy.is_shutdown():
#     for i in range (300):
#         PosePublisher.publish(pose_msg)
