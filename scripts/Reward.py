import rospy
import numpy as np
import matplotlib.pyplot as plt

import heapq
import math

from utils import *

# PossibleActions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 
#     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
X = np.arange(-0.2, 0.2, 0.1).tolist()
Y = np.arange(-0.2, 0.2, 0.1).tolist()
Z = np.arange(-0.2, 0.2, 0.1).tolist()

PossibleActions = []


for x in X:
    for y in Y:
        for z in Z:
            PossibleActions.append([x, y, z])


PossibleActions.insert(0, [0, 0, 0])

class Reward():

    def __init__(self, GoalPoint):

        self.GoalPoint = GoalPoint
        self.PossibleActions = PossibleActions
        self.NumActionsConsidering = 7
        self.LookaheadDepth = 1

        self.VelocityThreshold = 0.4 # m/s
        self.MaxAllowedVelocity = 0.8 # m/s
        self.StartTime = rospy.get_time()

        return 
    
    def SetGoalPoint(self, goal_point):
        
        self.GoalPoint = goal_point

        return 
    
    def SetLookaheadDepth(self, lookahead_depth):
        
        self.LookaheadDepth = lookahead_depth

        return 
    
    def GetLookaheadDepth(self, lookahead_depth):

        return 
    
    # def SetRewardInputList(self, reward_input_list):
    #     """
    #     This function is used to reset the reward input list, i.e.
    #     in the simulation we are have moved to the next step and have 
    #     thus recalculated our rewards.
    #     """

    #     self.reward_input_list = reward_input_list # Here reward_input_list
    #     # is a 2D list with elements of the form [action_index, dist_to_goal, vel_after_action].

    #     return 
    
    def ResetPlot(self):

        plt.clf()
        self.StartTime = rospy.get_time()

        return

    def PlotRewards(self, Rewards, best_reward):
        time_now = rospy.get_time()
        for r in Rewards:
            if(r[1] == best_reward):
                plt.plot(time_now - self.StartTime, r[1], 'bo-')
                plt.annotate("Action " + str(r[0]), (time_now - self.StartTime, r[1]))
            else:
                plt.plot(time_now - self.StartTime, r[1], 'rx')
                # plt.annotate("Action " + str(r[0]), (time_now - self.StartTime, r[1]))
    
        plt.title('Reward vs Time')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.xlim([time_now - 1.0 - self.StartTime, time_now + 1.0 - self.StartTime])
        plt.ylim([0, 1.1 * self.LookaheadDepth])

        plt.draw()
        plt.pause(0.01)
        plt.clf()
        
        return 


    def HeapSort(self, List, k):

        heap = []
        # Note: below is for illustration. It can be replaced by 
        # return heapq.nlargest( List, k )
        for item in List:
            # If we have not yet found k items, or the current item is larger than
            # the smallest item on the heap,
            if len(heap) < k or item[1] > heap[0][1]:
                # If the heap is full, remove the smallest element on the heap.
                if len(heap) == k: heapq.heappop( heap )
                # add the current element as the new smallest.
                heapq.heappush( heap, item )

        return heap


    def CreateRewardInputList(self, start_point, initial_velocity):

        BestAction = self.PossibleActions[0]
        best_end_point_rf = get_pose_after_action(start_point, initial_velocity, BestAction)

        sorted_reward_input_list = []

        action_index = 0
        for action in self.PossibleActions:
            
            # First find velocity of robot after taking current action (given the current velocty  ):
            direc_vel_after_action, lin_vel_after_action = get_vel_after_action(initial_velocity, action)
            # Now find endpoint in robot frame after taking current action:
            end_point_rf = get_pose_after_action(start_point, initial_velocity, action)
            # Now get distance to the goal:
            dist_to_goal = euclidean_dist(end_point_rf, self.GoalPoint)
            # Now that we have the velocity after action and distance to goal terms, we add to the reward input list:
            if len(sorted_reward_input_list) < self.NumActionsConsidering or dist_to_goal > sorted_reward_input_list[0][1]:
                # If the heap is full, remove the smallest element on the heap.
                if len(sorted_reward_input_list) == self.NumActionsConsidering: heapq.heappop( sorted_reward_input_list )
                # add the current element as the new smallest.
                heapq.heappush( sorted_reward_input_list, [action_index, dist_to_goal, direc_vel_after_action, lin_vel_after_action, end_point_rf] )

            action_index+=1
        
        return sorted_reward_input_list # Each element in this list contains an action index, and all of the ingredients 
        # for that action to calculate it's reward. sorted_reward_input_list additionally has only the NumActionsConsidering
        # best actions (for efficiency purposes). i.e. the top k actions that get us closest to the goal.

    

    def GetBestRewardIndex(self, sorted_reward_input_list, LookaheadDepth=1, gamma=0.9, plot_rewards=False):
        """ 
        This function finds the action index that gives the best reward, 
        i.e. returns the index in PossibleActions of the action that gives the best reward at the current step.
        
        """
        # sorted_reward_input_list = self.HeapSort(reward_input_list, self.NumActionsConsidering)
        # sorted_reward_input_list = sorted(reward_input_list, key=lambda x: x[1])[:self.NumActionsConsidering] 
        # Here the reward input list is sorted by distance to goal (only top NumActionsConsidering elements), in decreasing order.
        sorted_reward_input_list = np.array(sorted_reward_input_list)
        # We will now calculate the reward for the first five elements in the sorted list. For the closest five distances to 
        # the goal, we assign a reward value of 1 to 0.5 in increments of 0.1.
        Rewards = []

        for i in range (self.NumActionsConsidering):

            curr_end_vel_direc = sorted_reward_input_list[i, 2] # Directional (single value) and veloctity
            curr_end_vel_lin = sorted_reward_input_list[i, 3]
            curr_end_pose = sorted_reward_input_list[i, 4]
            curr_action_index = sorted_reward_input_list[i, 0]

            curr_reward = self.GetReward(i, end_directional_velocity=curr_end_vel_direc)
            
            ############ The following section performs multi-step lookahead: ############
            
            ############ End multi-step lookahead section. ###############################


            Rewards.append([curr_action_index, curr_reward])
        


        
        # Recursive step:

        if (LookaheadDepth == 0):

            best_action_and_reward = max(Rewards, key=lambda x: x[1])
            best_reward = best_action_and_reward[1]
            best_reward_action_index = best_action_and_reward[0]

            return int(best_reward_action_index), best_reward

        else: 
            
            for j in range (len(Rewards)):

                curr_end_vel_lin = sorted_reward_input_list[i, 3]
                curr_end_pose = sorted_reward_input_list[i, 4]

                curr_sorted_reward_input_list = self.CreateRewardInputList(curr_end_pose, curr_end_vel_lin) # We view the expected end pose and 
                # velocity after taking (theoretically) the current action as the initial pose and velocity for the next step in our multi-
                # step lookahead.

                curr_best_next_action_index, curr_future_reward = self.GetBestRewardIndex(
                curr_sorted_reward_input_list, LookaheadDepth-1, plot_rewards=False
                )

                Rewards[j][1] += gamma * curr_future_reward


            best_action_and_reward = max(Rewards, key=lambda x: x[1])
            best_reward = best_action_and_reward[1]
            best_reward_action_index = best_action_and_reward[0]
            print(best_reward_action_index)

            if (plot_rewards):
                self.PlotRewards(Rewards, best_reward)

            return int(best_reward_action_index), best_reward



    def GetReward(self, index, end_directional_velocity):
        """ 
        This function is used to return the reward for the current action in the 
        GetBestRewardIndex() function.
        """

        reward = 1.0 - float(index) / (2.0 * float(self.NumActionsConsidering))
        if (end_directional_velocity > self.VelocityThreshold):

                reward = reward - ( math.exp(end_directional_velocity - self.MaxAllowedVelocity) / 2 )
                # if(curr_end_vel_direc > self.MaxAllowedVelocity):
                #     reward = 0
        
        return reward

    # def GetLookaheadReward()