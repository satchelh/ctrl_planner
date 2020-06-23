#!/usr/bin/env python

from utils import *
from depthmapper.msg import DepthMap_msg
import tf2_ros
import tf2_geometry_msgs

import sys
import math

sys.path.insert(1, '/home/satchel/m100_ws/src/ctrl_planning/scripts/AutoEncoder')

from AutoEncoder import *

PossibleActions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 
    [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]

class SimulationClass():

    def __init__(self, GoalPoint, Mode='Train', Network=None): # Here mode is either Train or Test, where train uses the loss function to move and collects
        # training data along the way, and test uses a trained CNN to move.

        self.GoalPoint_rf = GoalPoint
        self.Mode = Mode
        self.Network = Network
        self.AutoEncoder = AutoEncoder()
        self.distance_to_goal_thresh = 0.5 # meters
        self.VelocityThreshold = 1.5 # m/s
        self.MaxAllowedVelocity = 2.0 # m/s
        self.RangeImageList = []
        self.TrainingDataRIs = [] # This will be the list of state values collected during simulation, 
        # where state is our range image and action is the selected 3d accleration
        self.TrainingDataEncodedRIs = []
        self.TrainingDataLabels = [] # And this will be the list of respective action values collected during simulation, 
        # where action is the selected 3d accleration

        self.PoseList = []
        self.MaxPoseListSize = 100
        self.MaxRangeImageListSize = 100
        self.VelocityList = [] # To store most recent velocity 
        self.PossibleActions = PossibleActions
        # publishers
        self.PosePublisher = rospy.Publisher(
            '/Elios/command/pose', PoseStamped, queue_size=45,
        )


        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        ### Now convert goal point to world frame:
        self.GoalPoint_rf_ps = xyz_2_PoseStamped(self.GoalPoint_rf)

        self.GoalPoint_wf_ps = self.RF_to_WF(self.GoalPoint_rf_ps)
        self.GoalPoint_wf = pose_2_np_arr_xyz(self.GoalPoint_wf_ps.pose)
        print('Distance to goal per dimension: ', self.GoalPoint_wf)

        self.topic_range_img = "/depth_map"
        self.topic_pose = '/Elios/ground_truth/pose'
        self.topic_odom = '/Elios/ground_truth/odometry'

        self.prev_publish_time = rospy.get_time()
        self.time_between_waypoints = 0.5 # The rate at which we publish the next waypoint (seconds)
        # I.e. every time_between_waypoints seconds we publish a waypoint.

        self.save_dir = '/home/satchel/m100_ws/src/ctrl_planning/training_data/' # Dir to save the collected training data files to

        print('\nEnter whether or not to save collected training information unitl the next step, y or n: ')
        self.save_data = input()


        self.AutoEncoder.load_model()
        self.init_subscribers()

        return

    def encode_RI(self, range_image):
        """ Accepts as input a flattened range image, direct from the topic,
        and ouputs the encoded version (1D) with the current Goalpoint added to the end.
        """
        RI = np.reshape(range_image, (16, 90))

        RI = np.expand_dims(RI, -1)
        RI = np.expand_dims(RI, 0)

        encoded_RI = self.AutoEncoder.encoder.predict(RI)

        encoded_RI_with_GP = np.append(encoded_RI[0], self.GoalPoint_rf, axis=0)

        return encoded_RI_with_GP

    
    def add_to_training_data(self, action):
        curr_RI = self.RangeImageList[0]

        self.TrainingDataRIs.append(curr_RI)

        encoded_RI_with_GP = self.encode_RI(curr_RI)

        self.TrainingDataEncodedRIs.append(encoded_RI_with_GP)

        self.TrainingDataLabels.append(self.PossibleActions.index(action))


    def find_and_publish_next_waypoint(self):
        start_point_rf = self.get_current_pose_xyz()
        ## Note that start_poit_rf is the point the robot is currently at in terms of the acual origin.
        # action_wf, end_point_wf = self.getBestActionWF(start_point_rf)
        if(self.Mode == 'Train'):
            action_rf, end_point_rf = self.getBestActionRF(start_point_rf)
        elif(self.Mode == 'Test'):
            # First reshape range image:
            range_image = self.RangeImageList[0]
            # range_image = np.expand_dims(range_image, -1)
            # range_image = np.expand_dims(range_image, 0)
            encoded_RI_with_GP = self.encode_RI(range_image)
            encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, -1)
            encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, 0)

            action_index = np.argmax(self.Network.predict(encoded_RI_with_GP)) # action_index = np.argmax(self.Network.predict(range_image))
            action_rf = self.PossibleActions[action_index]
            end_point_rf = get_pose_after_action(start_point_rf, action_rf)



        print('start point: ', start_point_rf)
        print('Taking action ', action_rf)
        print('This takes us to point ', end_point_rf)

        distance_to_goal = euclidean_dist(end_point_rf, self.GoalPoint_rf)
        print('The optimal distance to the goal was ', distance_to_goal)

        # if (distance_to_goal < self.distance_to_goal_thresh):
        #     print('\n\nWe have reached the goal (within distance to goal threshold)\n\n')
        #     print(self.TrainingData)  
        #     rospy.signal_shutdown('Shutdown')

        if (action_rf == [0, 0, 0] and distance_to_goal < self.distance_to_goal_thresh): # I.e. the best action is to not move (we have reached the goal), 
        # or we are simply trying to slow down a little bit. For the latter case, we don't want to do anything in this if statement, just continue the sim. So we 
        # also check in this if statement if the goal has actually been reached.
            
            if (self.Mode == 'Train'):
                with open (self.save_dir + 'TrainingDataRIs.txt', 'ab') as f:
                    np.savetxt(f, self.TrainingDataRIs, fmt='%.5f')
                with open (self.save_dir + 'TrainingDataEncodedRIs.txt', 'ab') as f:
                    np.savetxt(f, self.TrainingDataEncodedRIs, fmt='%.5f')
                with open (self.save_dir + 'TrainingDataLabels.txt', 'ab') as f:
                    np.savetxt(f, self.TrainingDataLabels, fmt='%.5f')
            
            self.TrainingDataRIs *= 0
            self.TrainingDataEncodedRIs *= 0
            self.TrainingDataLabels *= 0

            if (self.GoalPoint_rf == [0, 0, 2]):

                x = np.random.uniform(low=0, high=35)
                y = np.random.uniform(low=-1, high=0.5)
                z = np.random.uniform(low=0.5, high=2.8)

                self.GoalPoint_rf = [x, y, z]
                self.save_data = 'y'
        
            elif (self.GoalPoint_rf != [0, 0, 2]):
                self.GoalPoint_rf = [0, 0, 2]
                self.save_data = 'n'

        
        elif (self.Mode=='Test' and distance_to_goal < self.distance_to_goal_thresh): # This is how we stop the robot in test mode (using the net)
        # when it has reached it's goal
            print('\nEnter new goal point in format [x, y, z]: ')
            self.GoalPoint_rf = input()
            print('\nEnter whether or not to save collected training information unitl the next step, y or n: ')
            self.save_data = input()



            
        ## At this point we must account for offset in the low-level controller (z axis):
        # The offset is roughly 0.16 m:
        end_point_rf[2] = end_point_rf[2] - 0.15
        ## Now we convert endpoint to PoseStamped message:
        # end_point_wf_ps = xyz_2_PoseStamped(end_point_wf)
        end_point_rf_ps = xyz_2_PoseStamped(end_point_rf)

        if (self.save_data == 'y'):
            self.add_to_training_data(action_rf)

        self.publish_waypoint(end_point_rf_ps)
        self.prev_publish_time = rospy.get_time()

        return


    def RangeImageCallback(self, data):
        # rospy.loginfo("Recieved Range Image")
        # print(data.map.data[0])
        range_image = data.map.data 

        if (self.Mode == 'Test'):
            range_image = np.reshape(range_image, (16, 90))
        
        self.RangeImageList.insert(0, range_image)

        if(len(self.RangeImageList) > self.MaxRangeImageListSize):
            self.RangeImageList.pop()
        
        return
    
    def callback_velocity(self, msg):
        """ update queue which tracks recent velocity data """
        # if running in simulation, pose with covariance stamped is not used
        Vector3message = msg.twist.twist.linear
        LinVel = []
        LinVel.append(Vector3message.x)
        LinVel.append(Vector3message.y)
        LinVel.append(Vector3message.z)

        self.VelocityList.insert(0, LinVel)
        if len(self.VelocityList) > self.MaxPoseListSize: # We'll just use same size as the pose list
            self.VelocityList.pop()

        return

    def callback_pose(self, msg):
        """ update queue which tracks recent pose data """
        # if running in simulation, pose with covariance stamped is not used
        self.PoseList.insert(0, msg)
        if len(self.PoseList) > self.MaxPoseListSize:
            self.PoseList.pop()

        if ( (rospy.get_time() - self.prev_publish_time) > self.time_between_waypoints):
            self.find_and_publish_next_waypoint()
            self.prev_publish_time = rospy.get_time()

        return
    

    def init_subscribers(self):
        """ initialize ROS subscribers and publishers """
        # subscribers
        rospy.Subscriber(
            self.topic_range_img, DepthMap_msg, self.RangeImageCallback,
        )

        rospy.Subscriber(
            self.topic_odom, Odometry, self.callback_velocity, 
        )

        rospy.Subscriber(
            self.topic_pose, Pose, self.callback_pose, 
        )


        rospy.spin() # spin() simply keeps python from exiting until this node is stopped
        
        return

    def publish_waypoint(self, pose_stamped_msg):
        
        rate = rospy.Rate(50) # 1hz

        if not rospy.is_shutdown():
            print('Publishing Waypoint')
            self.PosePublisher.publish(pose_stamped_msg)
            # rate.sleep()

        return 


    def get_current_pose_xyz(self):
        """
        Returns:
            most recently-received pose as x,y,z coordinates in world frame
        """
        return pose_2_np_arr_xyz(self.PoseList[0])


    def RF_to_WF(self, pose_stamped_rf):
        """ 
        This function transfroms a pose_stamped point in the robot frame to the world frame.
        """
        rate = rospy.Rate(10.0)

        transform_found = False
        while not transform_found:
            try:
                tf_r2w = self.tf_buffer.lookup_transform('Elios/base_link', 'world', rospy.Time(0))
                transform_found = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('NO TRANSFORM FOUND')
                rate.sleep()
                continue
            
        pose_stamped_wf = tf2_geometry_msgs.do_transform_pose(pose_stamped_rf, tf_r2w)

        return pose_stamped_wf

    def WF_to_RF(self, pose_stamped_wf):
        """ 
        This function transfroms a pose_stamped point in the world frame to the robot frame.
        """
        rate = rospy.Rate(10.0)

        transform_found = False
        while not transform_found:
            try:
                tf_w2r = self.tf_buffer.lookup_transform('world', 'Elios/base_link', rospy.Time(0))
                transform_found = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print('NO TRANSFORM FOUND')
                rate.sleep()
                continue
            
        pose_stamped_rf = tf2_geometry_msgs.do_transform_pose(pose_stamped_wf, tf_w2r)

        return pose_stamped_rf


    def getBestActionRF(self, start_point):
        BestAction = self.PossibleActions[0]
        initial_velocity = self.VelocityList[0] # Note that this is 3d current velocity
        best_end_point_rf = get_pose_after_action(start_point, initial_velocity, BestAction)
        

        loss_input_list = []

        action_index = 0
        for action in self.PossibleActions:
            
            # First find velocity of robot after taking current action (given the current velocty  ):
            vel_after_action = get_vel_after_action(initial_velocity, action)
            # Now find endpoint in robot frame after taking current action:
            end_point_rf = get_pose_after_action(start_point, initial_velocity, action)
            # Now get distance to the goal:
            dist_to_goal = euclidean_dist(end_point_rf, self.GoalPoint_rf)
            # Now that we have the velocity after action and distance to goal terms, we add to the loss input list:
            loss_input_list.append([action_index, dist_to_goal, vel_after_action])

            action_index+=1

            # if (euclidean_dist(end_point_rf, self.GoalPoint_rf) < euclidean_dist(best_end_point_rf, self.GoalPoint_rf)):
            #     BestAction = action
            #     best_end_point_rf = end_point_rf

        
        best_action_index = self.GetBestLossIndex(loss_input_list)
        BestAction = self.PossibleActions[best_action_index]
        best_end_point_rf = get_pose_after_action(start_point, initial_velocity, BestAction)


        return BestAction, best_end_point_rf # Note tjat we return the best end point in terms of the robot frame, so we can publish it.

    def GetBestLossIndex(self, loss_input_list):
        """ 
        This function finds the action index that gives the best loss, 
        i.e. returns the index in PossibleActions of the action that gives the best loss at the current step.
        
        """

        sorted_loss_input_list = sorted(loss_input_list, key=lambda x: x[1]) # Here the loss input list is sorted by 
        # distance to goal, in decreasing order.
        # sorted_loss_input_list.reverse()
        sorted_loss_input_list = np.array(sorted_loss_input_list)
        # We will now calculate the loss for the first five elements in the sorted list. For the closest five distances to 
        # the goal, we assign a reward value of 1 to 0.5 in increments of 0.1.
        best_reward = 0.0
        best_reward_action_index = 0 # This is action [0, 0, 0], i.e. don't move. If there are no safe actions to take in the next
        # part of code, we will take this action.

        for i in range (5):

            curr_reward = 1.0 - float(i)/10
            curr_end_vel = sorted_loss_input_list[i, 2]

            if (curr_end_vel > self.VelocityThreshold):
                curr_reward = curr_reward - ( math.exp(curr_end_vel) / (2 * math.exp(self.MaxAllowedVelocity)) )
            
            if (curr_reward > best_reward):
                best_reward = curr_reward
                best_reward_action_index = sorted_loss_input_list[i, 0]
            # Right now we only consider the action if it produces an end velocity that's less than the threshold.
        
        print('\n\n\n\nREWARD: ', best_reward, '\n\n\n\n')
        return int(best_reward_action_index)


    def getBestActionWF(self, start_point=[0, 0, 0]):
        BestAction = self.PossibleActions[0]
        best_end_point_rf = get_pose_after_action(start_point, BestAction)
        # Now convert best_end_point_rf from robot frame to world frame.
        # We convert this endpoint to pose stamped message and transform it to world frame:
        best_end_point_rf_ps = xyz_2_PoseStamped(best_end_point_rf)
        best_end_point_wf_ps = self.RF_to_WF(best_end_point_rf_ps)
        # Now convert pose stamped message to xyz array (just position in world frame now)
        best_end_point_wf = pose_2_np_arr_xyz(best_end_point_wf_ps.pose)

        for action in self.PossibleActions:
            
            # first find endpoint in robot frame after taking current action:
            end_point_rf = get_pose_after_action(start_point, action)
            # Next convert this endpoint to pose stamped message 