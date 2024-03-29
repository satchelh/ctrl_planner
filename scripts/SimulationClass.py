from ActorCritic import *
from Reward import *
from utils import *

from depthmapper.msg import DepthMap_msg
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import ContactsState, ModelState
from std_srvs.srv import Empty, EmptyRequest

from tensorflow.keras import backend as K

import sys
import csv

sys.path.insert(1, '/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder')

from AutoEncoder import *

tf.compat.v1.disable_eager_execution()

class SimulationClass():

    def __init__(self, GoalPoint, Mode='Train', StartWithWeights=False):

        self.GoalPoint_rf = GoalPoint
        self.StartPoint_rf = [2, 0, 2]
        self.Mode = Mode
        self.StartWithWeights = StartWithWeights
        self.AutoEncoder = AutoEncoder()
        self.AutoEncoder.load_model()

        self.PointCloudList = []
        self.RangeImageList = []

        self.TrainingData = []
        self.TrainingDataRIs = [] # This will be the list of state values collected during simulation, 
        # where state is our range image and action is the selected 3d accleration
        self.EncodedRIs = []
        self.save_dir = '/home/satchel/ctrl_ws/src/ctrl_planner/training_data/'
        self.TrainingDataLabels = [] 

        self.PoseList = []

        self.MaxPointCLoudListSize = 3
        self.MaxRangeImageListSize = 3
        self.MaxPoseListSize = 1
        self.VelocityList = [] # To store most recent velocity 
        self.PossibleActions = PossibleActions

        self.model_name = 'elios_VLP16'
        # topics:
        self.topic_point_cloud = '/velodyne_points'
        self.topic_range_img = "/depth_map"
        self.topic_pose = '/' + self.model_name + '/ground_truth/pose'
        self.topic_odom = '/' + self.model_name + '/ground_truth/odometry'
        self.topic_contact = '/' + self.model_name + '/' + self.model_name + '_contact'
        # frames:
        self.robot_collision_frame = rospy.get_param(
            'robot_collision_frame',
            'elios::elios/base_link::elios/base_link_fixed_joint_lump__elios_collision_collision_1'
        )
        self.ground_collision_frame = rospy.get_param(
            'ground_collision_frame', 'ground_plane::link::collision')
        # publishers:
        self.PosePublisher = rospy.Publisher(
            '/elios_VLP16/goal', Pose, queue_size=1,
        )
        self.RateThrustPublisher = rospy.Publisher(
            '/elios_VLP16/command/rate_thrust', RateThrust, queue_size=1
        )
        self.ResetPublisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=1
        )

        self.session = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(self.session)
        self.ActorCritic = ActorCritic(self.session)

        self.curr_iteration = 0
        self.curr_episode = 0
        self.new_episode = False
        self.MAX_ITERATIONS_PER_EPISODE = 2
        self.NUM_EPISODES_BEFORE_SWITCH = 16
        self.NUM_EPISODES_BEFORE_DECAY = self.NUM_EPISODES_BEFORE_SWITCH - 8
        self.epsilon = 1.0
        self.epsilon_decay = 0.90

        self.Reward = Reward(self.StartPoint_rf, self.GoalPoint_rf)
        self.Top5RewardActions = []
        self.collision = False

        self.ResetSim()

        if self.Mode == 'Test' or self.StartWithWeights:
            self.ActorCritic.load_model()

        self.StepCounterForPlot = 0

        self.init_subscribers()
    

    def init_subscribers(self):
        """ initialize ROS subscribers and publishers """

        rospy.sleep(0.5)

        while not rospy.is_shutdown():

            rospy.Subscriber(
                self.topic_point_cloud, PointCloud2, self.PointCloudCallback,
            )

            rospy.Subscriber(
                self.topic_range_img, DepthMap_msg, self.RangeImageCallback,
            )

            rospy.Subscriber(
                self.topic_odom, Odometry, self.callback_odom, 
            )

            # rospy.Subscriber(
            #     self.topic_contact, ContactsState, self.contact_callback
            # )

            # rospy.Subscriber(
            #     self.topic_pose, Pose, self.callback_pose, 
            # )

            rospy.sleep(0.2)
            self.find_and_publish_next_action()

        # rospy.spin() # spin() simply keeps python from exiting until this node is stopped

        
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

        # if (self.Mode == 'Test'):
        #     range_image = np.reshape(range_image, (16, 90))
        
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
        
        # print(self.PoseList[0])

        # self.find_and_publish_next_action()
        
        return

    def contact_callback(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot_collision_frame):
                print('FUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCK')
                rospy.logdebug('Contact found!')
                self.collision = True
                # if (msg.states[i].collision2_name == self.ground_collision_frame):
                #     rospy.logdebug('Robot colliding with the ground')
                #     self.reset_sim()
                # else:
                #     rospy.logdebug(
                #         'Robot colliding with something else (not ground)')
                #     self.reset_sim()
            else:
                rospy.logdebug('Contact not found yet ...')


    def encode_RI(self, range_image):
        """ Accepts as input a flattened range image, direct from the topic,
        and ouputs the encoded version (1D) with the current Goalpoint added to the end.
        """
        RI = np.reshape(range_image, (16, 90))

        RI = np.expand_dims(RI, -1)
        RI = np.expand_dims(RI, 0)

        encoded_RI = self.AutoEncoder.encoder.predict(RI)

        encoded_RI_with_GP = np.append(encoded_RI[0], self.GoalPoint_rf, axis=0)
        encoded_RI_with_GP_and_vel = np.append(encoded_RI_with_GP, self.VelocityList[0], axis=0)

        return encoded_RI_with_GP_and_vel

    
    def add_to_training_data(self, prev_RI, action, reward, curr_RI=None):

        self.TrainingDataRIs.append(curr_RI)

        encoded_prev_RI = self.encode_RI(prev_RI)

        if (curr_RI == None):
            curr_RI = self.RangeImageList[0]
        
        encoded_curr_RI = self.encode_RI(curr_RI)

        self.TrainingData.append([encoded_prev_RI, action, reward, encoded_curr_RI]) # This is our main result

        self.EncodedRIs.append(encoded_prev_RI)


    def publish_action(self, rate_thrust_msg):
        
        # rate = rospy.Rate(50) # 1hz

        if not rospy.is_shutdown():
            # print('Publishing Acceleration')
            self.RateThrustPublisher.publish(rate_thrust_msg)

        return 
    
    def get_current_pose_xyz(self):
        """
        Returns:
            most recently-received pose as x,y,z coordinates in world frame
        """
        return pose_2_np_arr_xyz(self.PoseList[0])
    

    def find_and_publish_next_action(self):

        curr_RI = self.RangeImageList[0]
        ## Note that start_poit_rf is the point the robot is currently at in terms of the acual origin.
        # action_wf, end_point_wf = self.getBestActionWF(curr_pose)
        if(self.Mode == 'Train'):

            with self.session.as_default():

                with self.session.graph.as_default():

                    print('CURRENT EPISODE: ' + str(self.curr_episode + 1))

                    if self.new_episode and self.curr_episode % self.NUM_EPISODES_BEFORE_SWITCH >= self.NUM_EPISODES_BEFORE_DECAY: # Retrain when we start a new episode.
            
                        # self.curr_episode += 1

                        self.epsilon *= self.epsilon_decay

                        curr_pose = self.get_current_pose_xyz()

                        if (self.curr_episode + 1) % self.NUM_EPISODES_BEFORE_SWITCH == 0:
                            self.StartPoint_rf = curr_pose
                            # self.Reward.SetStartPoint(self.StartPoint_rf)
                            self.epsilon = 1.0
                            self.Top5RewardActions *= 0
                            self.ResetSim()
                        
                        # Stop the robot:
                        self.ResetSim()
                        temp_action = [0, 0, 0]
                        temp_action_msg = acc_to_RateThrust(temp_action)
                        self.RateThrustPublisher.publish(temp_action_msg)
                        # curr_pose = self.get_current_pose_xyz()
                        # curr_pose_msg = xyz_2_Pose(curr_pose)
                        # self.PosePublisher.publish(curr_pose_msg)
                        

                        print('\nRetraining with ' + str(len(self.TrainingData)) + ' samples.\n\n')
                        rospy.sleep(1)

                        if self.epsilon == 1.0:
                            self.ActorCritic.train(self.TrainingData, iterations=100)
                        elif self.epsilon < 1.0:
                            self.ActorCritic.train(self.TrainingData, iterations=10)

                        while self.ActorCritic.DoneTraining == False:
                            
                            continue
                            
                        
                        # rospy.sleep(5)
                        self.ActorCritic.DoneTraining = False

                        with open (self.save_dir + 'TrainingData.csv', 'ab') as csvfile:
                            writer = csv.writer(csvfile)
                            for data_sample in self.TrainingData:
                                writer.writerow(data_sample)
                        # with open (self.save_dir + 'EncodedRIs.txt', 'ab') as f:
                        #     np.savetxt(f, self.EncodedRIs, fmt='%.5f')


                        self.TrainingData *= 0

                        # if self.curr_episode >= self.NUM_EPISODES_BEFORE_DECAY: # i.e. only after NUM_EPISODES_BEFORE_DECAY
                        # # epochs of training with random actions do we start using predicitons from our actor net.
                        #     self.epsilon *= self.epsilon_decay
                        
                        self.ActorCritic.save_model()
                        
                    else:       

                        self.ResetSim()

                        curr_state = self.encode_RI(curr_RI)
                
                        curr_state = np.expand_dims(curr_state, 0)
                        
                        action_rf = self.ActorCritic.act(curr_state, self.epsilon)
                        action_rf = action_rf[0]
                        action_rf[2] = 0.0
                        action_rf[1] = 0.0

                        action_rf_rt = acc_to_RateThrust(action_rf)
                        self.publish_action(action_rf_rt) # Here publish action
                        # start_time = rospy.get_time()

                        # print('Taking action ', action_rf)
                        # while (rospy.get_time() - start_time) < ActionTime:
                        #     continue
                        rospy.sleep(ActionTime)
                
                        new_state = self.encode_RI(self.RangeImageList[0])
                        new_state = np.expand_dims(new_state, 0)
                        curr_pose = self.get_current_pose_xyz()
                        curr_pcl = self.PointCloudList[0]
                        curr_lin_vel = self.VelocityList[0]
                        # print(self.collision)
                        reward = self.Reward.GetReward(curr_pose, curr_pcl, curr_lin_vel, self.collision)
                        # print(curr_pose)

                        self.collision = False

                        action_rf = np.expand_dims(action_rf, 0)


                        self.TrainingData.append([curr_state, action_rf, reward, new_state])
                    

                    self.curr_iteration += 1
                    if self.curr_iteration % self.MAX_ITERATIONS_PER_EPISODE == 0:
                        self.new_episode = True
                        self.curr_episode +=1
                        if self.epsilon < 0.8:
                            self.StepCounterForPlot += 1
                            self.PlotRewardRealTime(self.StepCounterForPlot, reward)
                    else:
                        self.new_episode = False

        elif (self.Mode == 'Test'):

            curr_state = self.encode_RI(curr_RI)
            curr_state = np.expand_dims(curr_state, 0)
            action_rf = self.ActorCritic.act(curr_state, 0.0)
            action_rf = action_rf[0]
            action_rf[2] = 0.0
            action_rf[1] = 0.0

            action_rf_rt = acc_to_RateThrust(action_rf)
            self.publish_action(action_rf_rt) # Here publish action

            rospy.sleep(0.5)

            temp_action = [0, 0, 0]
            temp_action_msg = acc_to_RateThrust(temp_action)
            self.RateThrustPublisher.publish(temp_action_msg)
            curr_pose = self.get_current_pose_xyz()
            curr_pose_msg = xyz_2_Pose(curr_pose)
            self.PosePublisher.publish(curr_pose_msg)
            self.RateThrustPublisher.publish(temp_action_msg)
            # self.RateThrustPublisher.publish(temp_action_msg)
            rospy.sleep(1)



    def AddToTop5(self, action, reward):

        # Note that this assumes that the Top5RewardActions is in order least (loc 0) to greatest (loc 4)
        Added = False

        if len(self.Top5RewardActions < 5):
            self.Top5RewardActions.append([action, reward])
            Added = True
        elif len(self.Top5RewardActions == 5) and reward > self.Top5RewardActions[0][1]:
            self.Top5RewardActions[0] = [action, reward]
            Added = True
        
        if Added:
            self.Top5RewardActions.sort(key=lambda pair: pair[1])

        return Added


    
    def PlotRewardRealTime(self, epoch, reward):

        plt.plot(epoch, reward, 'ro')

        plt.title('1 Step Lookahead Weighted Real Time Reward')

        plt.xlabel('Epoch')
        plt.ylabel('Reward')

        plt.xlim([0, epoch])
        plt.ylim([-200, 550])

        plt.draw()
        plt.pause(0.2)


        # self.reset_timer()

        # rospy.loginfo('Unpausing physics')
        # self.unpause_physics_proxy(EmptyRequest())


            # action_rf = self.ActorCritic.act(self.encode_RI(curr_state))
            # start_time = rospy.get_time()

            # action_rf_rt = acc_to_RateThrust(action_rf)
            # self.publish_action(action_rf_rt) # Here publish action

            # while (rospy.get_time - start_time) < ActionTime:
            #     print('Taking action ', action_rf)
            
            # new_state = self.RangeImageList[0]
            # curr_pose = self.get_current_pose_xyz()
            # curr_pcl = self.PointCloudList[0]
            # curr_lin_vel = self.VelocityList[0]
            # reward = self.Reward.GetReward(curr_pose, curr_pcl, curr_lin_vel)

            # self.add_to_training_data(curr_state, action_rf, reward, new_state)

            

            # self.ActorCritic.train(self.TrainingData)

            # tf.compat.v1.keras.backend.clear_session()
            # sess = tf.compat.v1.Session()
            # tf.compat.v1.keras.backend.set_session(sess)


        # elif(self.Mode == 'Test'):
        #     # First reshape range image:
        #     range_image = self.RangeImageList[0]
        #     # range_image = np.expand_dims(range_image, -1)
        #     # range_image = np.expand_dims(range_image, 0)
        #     encoded_RI_with_GP = self.encode_RI(range_image)
        #     encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, -1)
        #     encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, 0)

        #     action_index = np.argmax(self.Network.predict(encoded_RI_with_GP)) # action_index = np.argmax(self.Network.predict(range_image))
        #     action_rf = self.PossibleActions[action_index]
        #     end_point_rf = get_pose_after_action(curr_pose, self.VelocityList[0], action_rf)

        #     print('start point: ', curr_pose)
        #     print('Taking action ', action_rf)
        #     print('This takes us to point ', end_point_rf)

        #     distance_to_goal = euclidean_dist(end_point_rf, self.GoalPoint_rf)
        #     print('The optimal distance to the goal was ', distance_to_goal)

        #     self.inference_time = rospy.get_time() - self.prev_publish_time
        #     print('TOTAL TIME SINCE LAST PUBLISH (TEST MODE): ', self.inference_time)


        # # if (distance_to_goal < self.distance_to_goal_thresh):
        # #     print('\n\nWe have reached the goal (within distance to goal threshold)\n\n')
        # #     print(self.TrainingData)  
        # #     rospy.signal_shutdown('Shutdown')

        # if (distance_to_goal < self.distance_to_goal_thresh): # I.e. the best action is to not move (we have reached the goal), 
        # # or we are simply trying to slow down a little bit. For the latter case, we don't want to do anything in this if statement, just continue the sim. So we 
        # # also check in this if statement if the goal has actually been reached.

        #     # while(True):
        #     #     self.publish_action(acc_to_RateThrust([0, 0, 0])) # Stop for a little bit
        #     #     rospy.sleep(0.5)

        #     if (self.Mode == 'Train'):
        #         with open (self.save_dir + 'TrainingDataRIs.txt', 'ab') as f:
        #             np.savetxt(f, self.TrainingDataRIs, fmt='%.5f')
        #         with open (self.save_dir + 'TrainingDataEncodedRIs.txt', 'ab') as f:
        #             np.savetxt(f, self.TrainingDataEncodedRIs, fmt='%.5f')
        #         with open (self.save_dir + 'TrainingDataLabels.txt', 'ab') as f:
        #             np.savetxt(f, self.TrainingDataLabels, fmt='%.5f')
            
        #     self.TrainingDataRIs *= 0
        #     self.TrainingDataEncodedRIs *= 0
        #     self.TrainingDataLabels *= 0

        #     self.Reward.ResetPlot()

        #     if (self.GoalPoint_rf == [0, 0, 2]):

        #         x = np.random.uniform(low=0, high=28)
        #         y = np.random.uniform(low=-1, high=0.5)
        #         z = np.random.uniform(low=0.8, high=2.8)

        #         self.GoalPoint_rf = [x, y, z]
        #         self.Reward.SetGoalPoint(self.GoalPoint_rf)
        #         self.save_data = 'y'
        
        #     elif (self.GoalPoint_rf != [0, 0, 2]):
        #         self.GoalPoint_rf = [0, 0, 2]
        #         self.save_data = 'n'

        
        # elif (self.Mode=='Test' and distance_to_goal < self.distance_to_goal_thresh): # This is how we stop the robot in test mode (using the net)
        # # when it has reached it's goal
        #     print('\nEnter new goal point in format [x, y, z]: ')
        #     self.GoalPoint_rf = input()
        #     print('\nEnter whether or not to save collected training information unitl the next step, y or n: ')
        #     self.save_data = input()

            
        # ## At this point we must account for offset in the low-level controller (z axis):
        # # The offset is roughly 0.16 m:
        # # end_point_rf[2] = end_point_rf[2] + 0.137
        # ## Now we convert endpoint to PoseStamped message:
        # # end_point_wf_ps = xyz_2_PoseStamped(end_point_wf)
        # action_rf_rt = acc_to_RateThrust(action_rf)
        # end_point_rf_ps = xyz_2_PoseStamped(end_point_rf)




        # if (self.save_data == 'y'):
        #     self.add_to_training_data(action_rf)

        # if (euclidean_dist(curr_pose, [0.0, 0.0, 0.0]) < 0.2): # This is for the purpose of goin up a little bit before starting the real sim
        #     point_rf = [0, 0, 1]
        #     print('\nInstead, moving to point ', point_rf)
        #     self.PosePublisher(xyz_2_Pose(point_rf))
        # else:
        #     self.publish_action(action_rf_rt)


        # self.prev_publish_time = rospy.get_time()

        return

    def ResetSim(self):
        # rospy.loginfo('Pausing physics')
        # self.pause_physics_proxy(EmptyRequest())

        # Fill in the new position of the robot
        new_position = ModelState()
        new_position.model_name = self.model_name
        new_position.reference_frame = 'world'
        new_position.pose.position.x = self.StartPoint_rf[0]
        new_position.pose.position.y = self.StartPoint_rf[1]
        new_position.pose.position.z = self.StartPoint_rf[2]
        new_position.pose.orientation.x = 0
        new_position.pose.orientation.y = 0
        new_position.pose.orientation.z = 0
        new_position.pose.orientation.w = 1
        new_position.twist.linear.x = 0
        new_position.twist.linear.y = 0
        new_position.twist.linear.z = 0
        new_position.twist.angular.x = 0
        new_position.twist.angular.y = 0
        new_position.twist.angular.z = 0
        # rospy.loginfo('Placing robot')
        self.ResetPublisher.publish(new_position)

        return 
