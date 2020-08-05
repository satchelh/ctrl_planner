from utils import *

from sensor_msgs.point_cloud2 import read_points
import numpy as np

import math

class Reward():

    def __init__(self, StartPoint, GoalPoint):

        self.StartPoint = StartPoint
        self.GoalPoint = GoalPoint
        self.StartDist = euclidean_dist(StartPoint, GoalPoint)
        self.VelocityThresh = 0.4
        self.MaxAllowedVelocity = 0.8

        self.ObjectDistanceThresh = 1.0
        self.ClosestSafeObjectDist = 0.2 # Note that we still wish to allow collisions, so
        # all this closest safe object distance variable does is set the loss from distance
        # to object constant if we are within this distance to the closest surface.

    def SetGoalPoint(self, GoalPoint):

        self.GoalPoint = Goalpoint
        self.StartDist = euclidean_dist(self.StartPoint, self.GoalPoint)

        return
    
    def SetStartPoint(self, StartPoint):

        self.StartPoint = StartPoint
        self.StartDist = euclidean_dist(self.StartPoint, self.GoalPoint)

        return
    
    def GetReward(self, pose, pointcloud, lin_vel, caused_collision=False):

        curr_reward = 0.0
        # print('START_DIST ' + str(self.StartDist))
        dist_to_goal = euclidean_dist(pose, self.GoalPoint)
        # print('CURR_DIST ' + str(dist_to_goal))
        dist_to_goal = self.StartDist - dist_to_goal
        curr_reward += dist_to_goal
        curr_reward *= 15
        # if dist_to_goal <= 1.0:
        #     curr_reward += 100.0
        # elif dist_to_goal > 1.0:
        #     curr_reward += 300 / dist_to_goal # math.exp(-dist_to_goal)


        # dir_vel = get_dir_vel(lin_vel)
        # if dir_vel >= self.VelocityThresh:

        #     if dir_vel >= self.MaxAllowedVelocity:
        #         curr_reward -= 14
        #     elif dir_vel < self.MaxAllowedVelocity:
        #         curr_reward -= 14 * math.exp(dir_vel - self.MaxAllowedVelocity)

        #     if caused_collision:
        #         curr_reward -= 10 * dir_vel
        
        # dist_to_closest_point = get_distance_to_closest_point(pointcloud)
        # if dist_to_closest_point <= self.ObjectDistanceThresh:

        #     if dist_to_closest_point <= self.ClosestSafeObjectDist:
        #         curr_reward -= 70
        #     elif dist_to_closest_point > self.ClosestSafeObjectDist:
        #         curr_reward -= 70 * math.exp(self.ClosestSafeObjectDist - dist_to_closest_point)

        
        curr_reward *= 5
        print('CURR_REWARD ' + str(curr_reward))


        return curr_reward
