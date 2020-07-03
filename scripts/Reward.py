import rospy
import numpy as np
import matplotlib.pyplot as plt

import math

# PossibleActions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 
#     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
X = np.arange(-0.3, 0.3, 0.1).tolist()
Y = np.arange(-0.3, 0.3, 0.1).tolist()
Z = np.arange(0.5, 1.5, 1.0).tolist()

PossibleActions = []


for x in X:
    for y in Y:
        for z in Z:
            PossibleActions.append([x, y, z])


PossibleActions.insert(0, [0, 0, 0])

class Reward():

    def __init__(self):

        self.VelocityThreshold = 0.3 # m/s
        self.MaxAllowedVelocity = 0.7 # m/s
        self.StartTime = rospy.get_time()

        return 
    
    def SetRewardInputList(self, reward_input_list):
        """
        This function is used to reset the reward input list, i.e.
        in the simulation we are have moved to the next step and have 
        thus recalculated our rewards.
        """

        self.reward_input_list = reward_input_list # Here reward_input_list
        # is a 2D list with elements of the form [action_index, dist_to_goal, vel_after_action].

        return 
    
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
        plt.ylim([0, 1.2])

        plt.draw()
        plt.pause(0.01)
        plt.clf()
        
        return 

    def GetBestRewardIndex(self, plot_rewards=False):
        """ 
        This function finds the action index that gives the best reward, 
        i.e. returns the index in PossibleActions of the action that gives the best reward at the current step.
        
        """
        sorted_reward_input_list = sorted(self.reward_input_list, key=lambda x: x[1]) # Here the reward input list is sorted by 
        # distance to goal, in decreasing order.
        # sorted_reward_input_list.reverse()
        sorted_reward_input_list = np.array(sorted_reward_input_list)
        # We will now calculate the reward for the first five elements in the sorted list. For the closest five distances to 
        # the goal, we assign a reward value of 1 to 0.5 in increments of 0.1.
        best_reward = 0.0
        best_reward_action_index = PossibleActions.index([0, 0, 0]) # This is action [0, 0, 0], i.e. don't move. If there are no safe actions to take in the next
        # part of code, we will take this action.
        Rewards = []

        num_of_actions_to_consider = 7
        for i in range (num_of_actions_to_consider):

            curr_reward = 1.0 - float(i) / (2.0 * float(num_of_actions_to_consider))
            curr_end_vel = sorted_reward_input_list[i, 2]
            curr_action_index = sorted_reward_input_list[i, 0]

            if (curr_end_vel > self.VelocityThreshold):

                curr_reward = curr_reward - ( math.exp(curr_end_vel - self.MaxAllowedVelocity) / 2 )
                # if(curr_end_vel > self.MaxAllowedVelocity):
                #     curr_reward = 0
            
            Rewards.append([curr_action_index, curr_reward])

            if (curr_reward > best_reward):
                best_reward = curr_reward
                best_reward_action_index = curr_action_index
            # Right now we only consider the action if it produces an end velocity that's less than the threshold.
        
        if (plot_rewards):
            self.PlotRewards(Rewards, best_reward)

        return int(best_reward_action_index)


    # def GetLookaheadReward()