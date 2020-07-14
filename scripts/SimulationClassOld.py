#!/usr/bin/env python

from utils import *
from depthmapper.msg import DepthMap_msg
from sensor_msgs.msg import PointCloud2
import tf2_ros
import tf2_geometry_msgs

import sys

sys.path.insert(1, '/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder')

from AutoEncoder import *
from Reward import *


class SimulationClass():

    def __init__(self, GoalPoint, Mode='Train', Network=None): # Here mode is either Train or Test, where train uses the loss function to move and collects
        # training data along the way, and test uses a trained CNN to move.

        self.GoalPoint_rf = GoalPoint
        self.Mode = Mode
        self.Network = Network
        self.AutoEncoder = AutoEncoder()
        self.Reward = Reward(self.GoalPoint_rf)
        self.distance_to_goal_thresh = 0.9 # meters
        self.PointCloudList = []
        self.RangeImageList = []
        self.TrainingDataRIs = [] # This will be the list of state values collected during simulation, 
        # where state is our range image and action is the selected 3d accleration
        self.TrainingDataEncodedRIs = []
        self.TrainingDataLabels = [] # And this will be the list of respective action values collected during simulation, 
        # where action is the selected 3d accleration

        self.PoseList = []
        self.MaxPoseListSize = 5
        self.MaxPointCLoudListSize = 5
        self.MaxRangeImageListSize = 5
        self.VelocityList = [] # To store most recent velocity 
        self.PossibleActions = PossibleActions

        # topics:
        self.topic_range_img = "/depth_map"
        self.topic_pose = '/elios_VLP16/ground_truth/pose'
        self.topic_odom = '/elios_VLP16/ground_truth/odometry'
        # publishers:
        self.PosePublisher = rospy.Publisher(
            '/elios_VLP16/goal', Pose, queue_size=45,
        )
        self.RateThrustPublisher = rospy.Publisher(
            '/elios_VLP16/command/rate_thrust', RateThrust, queue_size=45
        )

        # create transforms:
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        ### Now convert goal point to world frame:
        self.GoalPoint_rf_ps = xyz_2_PoseStamped(self.GoalPoint_rf)

        self.GoalPoint_wf_ps = self.RF_to_WF(self.GoalPoint_rf_ps)
        self.GoalPoint_wf = pose_2_np_arr_xyz(self.GoalPoint_wf_ps.pose)
        print('Distance to goal per dimension: ', self.GoalPoint_wf)


        self.prev_publish_time = rospy.get_time()
        self.inference_time = 0
        self.time_between_publishes = ActionTime # The rate at which we publish the next waypoint (seconds)
        # I.e. every time_between_publishes seconds we publish a waypoint.

        self.save_dir = '/home/satchel/ctrl_ws/src/ctrl_planner/training_data/' # Dir to save the collected training data files to

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


    def find_and_publish_next_action(self):

        curr_pose = self.get_current_pose_xyz()
        ## Note that start_poit_rf is the point the robot is currently at in terms of the acual origin.
        # action_wf, end_point_wf = self.getBestActionWF(curr_pose)
        if(self.Mode == 'Train'):

            action_rf, end_point_rf = self.getBestActionRF(curr_pose)

            # print('\n\nstart point: ', curr_pose)
            distance_to_goal = euclidean_dist(curr_pose, self.GoalPoint_rf)
            print('Goal point: ', self.GoalPoint_rf)
            print('The current distance to the goal is ', distance_to_goal)
            print('Taking action ', action_rf)
            self.inference_time = rospy.get_time() - self.prev_publish_time
            print('TOTAL TIME SINCE LAST PUBLISH (TRAIN MODE): ', self.inference_time)

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
            end_point_rf = get_pose_after_action(curr_pose, self.VelocityList[0], action_rf)

            print('start point: ', curr_pose)
            print('Taking action ', action_rf)
            print('This takes us to point ', end_point_rf)

            distance_to_goal = euclidean_dist(end_point_rf, self.GoalPoint_rf)
            print('The optimal distance to the goal was ', distance_to_goal)

            self.inference_time = rospy.get_time() - self.prev_publish_time
            print('TOTAL TIME SINCE LAST PUBLISH (TEST MODE): ', self.inference_time)


        # if (distance_to_goal < self.distance_to_goal_thresh):
        #     print('\n\nWe have reached the goal (within distance to goal threshold)\n\n')
        #     print(self.TrainingData)  
        #     rospy.signal_shutdown('Shutdown')

        if (distance_to_goal < self.distance_to_goal_thresh): # I.e. the best action is to not move (we have reached the goal), 
        # or we are simply trying to slow down a little bit. For the latter case, we don't want to do anything in this if statement, just continue the sim. So we 
        # also check in this if statement if the goal has actually been reached.

            # while(True):
            #     self.publish_action(acc_to_RateThrust([0, 0, 0])) # Stop for a little bit
            #     rospy.sleep(0.5)

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

            self.Reward.ResetPlot()

            if (self.GoalPoint_rf == [0, 0, 2]):

                x = np.random.uniform(low=0, high=28)
                y = np.random.uniform(low=-1, high=0.5)
                z = np.random.uniform(low=0.8, high=2.8)

                self.GoalPoint_rf = [x, y, z]
                self.Reward.SetGoalPoint(self.GoalPoint_rf)
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
        # end_point_rf[2] = end_point_rf[2] + 0.137
        ## Now we convert endpoint to PoseStamped message:
        # end_point_wf_ps = xyz_2_PoseStamped(end_point_wf)
        action_rf_rt = acc_to_RateThrust(action_rf)
        end_point_rf_ps = xyz_2_PoseStamped(end_point_rf)




        if (self.save_data == 'y'):
            self.add_to_training_data(action_rf)

        if (euclidean_dist(curr_pose, [0.0, 0.0, 0.0]) < 0.2): # This is for the purpose of goin up a little bit before starting the real sim
            point_rf = [0, 0, 1]
            print('\nInstead, moving to point ', point_rf)
            self.PosePublisher(xyz_2_Pose(point_rf))
        else:
            self.publish_action(action_rf_rt)


        self.prev_publish_time = rospy.get_time()

        return


    def PointCloudCallback(self, data):

        pcl = data
        self.PointCloudList.insert(0, pcl)

        if len(self.PointCloudList) > self.MaxPointCLoudListSize:
            self.PointCloudList.pop()

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
    
    def callback_odom(self, msg):
        """ update queue which tracks recent velocity and pose data """
        # if running in simulation, pose with covariance stamped is not used
        Vector3message = msg.twist.twist.linear
        LinVel = []
        LinVel.append(float(Vector3message.x))
        LinVel.append(float(Vector3message.y))
        LinVel.append(float(Vector3message.z))

        self.VelocityList.insert(0, LinVel)
        if len(self.VelocityList) > self.MaxPoseListSize: # We'll just use same size as the pose list
            self.VelocityList.pop()

        # if running in simulation, pose with covariance stamped is not used
        self.PoseList.insert(0, msg.pose.pose)

        if len(self.PoseList) > self.MaxPoseListSize:
            self.PoseList.pop()

        
        if ( (rospy.get_time() - self.prev_publish_time) > self.time_between_publishes):
            self.find_and_publish_next_action()
            self.prev_publish_time = rospy.get_time()

        
        return

    def callback_pose(self, msg):
        """ update queue which tracks recent pose data """
        # if running in simulation, pose with covariance stamped is not used
        self.PoseList.insert(0, msg)
        if len(self.PoseList) > self.MaxPoseListSize:
            self.PoseList.pop()

        if ( (rospy.get_time() - self.prev_publish_time) > self.time_between_publishes):
            self.find_and_publish_next_action()
            self.prev_publish_time = rospy.get_time()

        return
    

    def init_subscribers(self):
        """ initialize ROS subscribers and publishers """
        # subscribers
        # rospy.Subscriber(
        #     self.topic_point_cloud, PointCloud2, self.PointCloudCallback,
        # )

        rospy.Subscriber(
            self.topic_range_img, DepthMap_msg, self.RangeImageCallback,
        )

        rospy.Subscriber(
            self.topic_odom, Odometry, self.callback_odom, 
        )

        # rospy.Subscriber(
        #     self.topic_pose, Pose, self.callback_pose, 
        # )

        rospy.spin() # spin() simply keeps python from exiting until this node is stopped

        
        return

    def publish_waypoint(self, pose_stamped_msg):
        
        # rate = rospy.Rate(50) # 1hz

        if not rospy.is_shutdown():
            print('Publishing Waypoint')
            self.PosePublisher.publish(pose_stamped_msg)

        return 

    def publish_action(self, rate_thrust_msg):
        
        # rate = rospy.Rate(50) # 1hz

        if not rospy.is_shutdown():
            print('Publishing Acceleration')
            self.RateThrustPublisher.publish(rate_thrust_msg)

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
                tf_r2w = self.tf_buffer.lookup_transform('elios_VLP16/base_link', 'world', rospy.Time(0))
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
                tf_w2r = self.tf_buffer.lookup_transform('world', 'elios_VLP16/base_link', rospy.Time(0))
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
        
        self.Reward.SetLookaheadDepth(0)
        reward_input_list = self.Reward.CreateRewardInputList(start_point, initial_velocity)

        # self.Reward.SetRewardInputList(reward_input_list)
        best_action_index, best_reward = self.Reward.GetBestRewardIndex(
            reward_input_list, LookaheadDepth=self.Reward.LookaheadDepth, gamma=0.9, plot_rewards=True
        )

        BestAction = self.PossibleActions[best_action_index]
        best_end_point_rf = get_pose_after_action(start_point, initial_velocity, BestAction)


        return BestAction, best_end_point_rf # Note that we return the best end point in terms of the robot frame, so we can publish it.


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