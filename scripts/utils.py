#!/usr/bin/env python

import rospy
import numpy as np

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
# from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

ActionTime = 0.3

def xyz_2_PoseStamped(xyz, angles=None):
    """
    converts numpy array (xyz point) to PoseStamped message
    """
    ps = PoseStamped()
    ps.pose.position.x = xyz[0]
    ps.pose.position.y = xyz[1]
    ps.pose.position.z = xyz[2]
    if angles is None:
        ps.pose.orientation.x = 0
        ps.pose.orientation.y = 0
        ps.pose.orientation.z = 0
        ps.pose.orientation.w = 1
    else:
        ps.pose.orientation.x = angles[0]
        ps.pose.orientation.y = angles[1]
        ps.pose.orientation.z = angles[2]
        ps.pose.orientation.w = angles[3]

    ps.header.frame_id = 'world'
    return ps


def xyz_2_Pose(xyz, angles=None):
    """
    converts numpy array (xyz point) to Pose message
    """
    p = Pose()
    p.position.x = xyz[0]
    p.position.y = xyz[1]
    p.position.z = xyz[2]
    if angles is None:
        p.orientation.x = 0
        p.orientation.y = 0
        p.orientation.z = 0
        p.orientation.w = 1
    else:
        p.orientation.x = angles[0]
        p.orientation.y = angles[1]
        p.orientation.z = angles[2]
        p.orientation.w = angles[3]

    return p


def get_pose_after_action(start_point, initial_velocity, action, time=ActionTime):
    """
    Gets the final point after taking a given action starting from 
    the current position. Note: this function assumes no inital velocity.
    arguments: 
        start_point (x, y, z), action (3d acceleration vector), time (seconds)
    returns:
        final pose after taking action (np array of length 3)
    """

    action = [float(a) for a in action]
    time = float(time)
    initial_velocity = [float(v) for v in initial_velocity]
    
    xyz = np.zeros(shape=(3,))
    xyz[0] = start_point[0] + ( (initial_velocity[0] + (action[0] * time) / 2) * time ) # Formula for finding distance traveled 
    # given constant acceleration and time, is ( initial_velocity + (a * t) / 2 ) * t. In our case initial_velocity is 0.
    xyz[1] = start_point[1] + ( (initial_velocity[1] + (action[1] * time) / 2) * time )
    xyz[2] = start_point[2] + ( (initial_velocity[2] + (action[2] * time) / 2) * time )

    return xyz

def get_vel_after_action(initial_velocity, action, time=ActionTime):
    """
    Gets the final velocity (single value) after taking a given action (given current linear velocity)
    arguments:
        initial linear velocity (i.e. vel in [x, y, z] in m/s), action (3d acceleration vector), time (seconds)
    returns:
        final velocity after taking action 
    """

    initial_velocity = [float(v) for v in initial_velocity]
    action = [float(a) for a in action]
    time = float(time)
    
    # Formula for finding final velocity given constant acceleration and time, and initial_velocity,
    # is ( initial_velocity + (a * t) ). Note that we also need the direction in which the robot is heading.

    final_velocity = np.zeros(shape=(3,)) # final_velocity will be the 3d final velocity.
    final_velocity[0] = initial_velocity[0] + ( action[0] * time )
    final_velocity[1] = initial_velocity[1] + ( action[1] * time )
    final_velocity[2] = initial_velocity[2] + ( action[2] * time )

    # Now we must get a single value for the final velocity, i.e. the velocity in the direction the robot would 
    # actually be moving at the projected final position
    directional_final_velocity = np.sqrt( np.sum ( np.square(final_velocity), axis=0 ) )

    return directional_final_velocity

def euclidean_dist(start, end):
    """
    calculates euclidean distance between points 
    """
    r_0 = np.array(
        [start[0], start[1], start[2]]
    )
    r_1 = np.array(
        [end[0], end[1], end[2]]
    )

    diff = r_1 - r_0 
    dist = np.sqrt(np.dot(diff, diff))
    return dist


def pose_2_np_arr_xyz(msg):
    """
    converts pose message to x, y, z coordinates
    arguments:
        msg (pose message)
    returns:
        numpy array containing x,y,z coordinates in world frame
    """
    x = msg.position.x 
    y = msg.position.y
    z = msg.position.z
    return np.array([x,y,z])

def acc_to_RateThrust(acc):
    """
    converts 3D acceleration to geometry_msgs/Vector3
    and therefore creates the mav_msgs/RateThrust message.
    """

    acc_vec3 = Vector3()
    acc_vec3.x = acc[0]
    acc_vec3.y = acc[1]
    acc_vec3.z = acc[2]

    direction_vec3 = Vector3()
    direction_vec3.x = 0
    direction_vec3.y = 0
    direction_vec3.z = 0

    rate_thrust_msg = RateThrust()
    rate_thrust_msg.thrust = acc_vec3
    rate_thrust_msg.angular_rates = direction_vec3
    return rate_thrust_msg
