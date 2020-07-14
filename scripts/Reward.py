from utils import *

from sensor_msgs.point_cloud2 import read_points
import numpy as np

import math

class Reward():

    def __init__(self, GoalPoint):

        self.GoalPoint = GoalPoint
        self.VelocityThresh = 0.3
        self.MaxAllowedVelocity = 0.6

        self.ObjectDistanceThresh = 1.0
        self.ClosestSafeObjectDist = 0.2 # Note that we still wish to allow collisions, so
        # all this closest safe object distance variable does is set the loss from distance
        # to object constant if we are within this distance to the closest surface.

    def SetGoalPoint(self, GoalPoint):

        self.GoalPoint = Goalpoint

        return
    
    def GetReward(self, pose, pointcloud, lin_vel):

        curr_reward = 0.0

        dist_to_goal = euclidean_dist(pose, self.GoalPoint)
        if dist_to_goal <= 1.0:
            curr_reward += 15.0
        elif dist_to_goal > 1.0:
            curr_reward += 55 * 1/dist_to_goal # math.exp(-dist_to_goal)

        # dir_vel = get_dir_vel(lin_vel)
        # if dir_vel >= self.VelocityThresh:

        #     if dir_vel >= self.MaxAllowedVelocity:
        #         curr_reward -= 7
        #     elif dir_vel < self.MaxAllowedVelocity:
        #         curr_reward -= 3 * math.exp(dir_vel - self.MaxAllowedVelocity)
        
        # dist_to_closest_point = get_distance_to_closest_point(pointcloud)
        # if dist_to_closest_point <= self.ObjectDistanceThresh:

        #     if dist_to_closest_point <= self.ClosestSafeObjectDist:
        #         curr_reward -= 5
        #     elif dist_to_closest_point > self.ClosestSafeObjectDist:
        #         curr_reward -= 5 * math.exp(self.ClosestSafeObjectDist - dist_to_closest_point)


        return curr_reward
