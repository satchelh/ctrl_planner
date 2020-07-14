#!/usr/bin/env python

import rospy
import time
import numpy as np
import random
from gazebo_msgs.msg import ContactsState, ModelState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty, EmptyRequest
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# start quadrotor
# os.system("timeout 1s rostopic pub /hummingbird/bridge/arm std_msgs/Bool 'True'")
# os.system("timeout 1s rostopic pub /hummingbird/autopilot/start std_msgs/Empty")


class DataGenerator:
    def __init__(self):
        rospy.init_node("data_generation_node")
        self.goal_pub = rospy.Publisher('goal_topic', Pose, queue_size=1)
        self.model_state_pub = rospy.Publisher('/gazebo/set_model_state',
                                               ModelState,
                                               queue_size=1)
        self.sphere_marker_pub = rospy.Publisher('sphere_marker_topic',
                                                 MarkerArray,
                                                 queue_size=1)
        self.current_pose = None  # In world frame
        # In world frame, used to check if the robot has reached its intended position
        self.current_goal = None
        self.get_params()
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics',
                                                      Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)
        rospy.Subscriber('contact_topic', ContactsState, self.contact_cb)
        rospy.Subscriber('odom_topic', Odometry, self.odom_cb)
        self.timeout_timer = rospy.Timer(
            rospy.Duration(self.goal_generation_radius * 5), self.timer_cb)

    def get_params(self):
        self.goal_generation_radius = rospy.get_param('goal_generation_radius',
                                                      2)
        self.waypoint_radius = rospy.get_param('waypoint_radius', 0.1)
        self.robot_collision_frame = rospy.get_param(
            'robot_collision_frame',
            'delta::delta/base_link::delta/base_link_fixed_joint_lump__delta_collision_collision'
        )
        self.ground_collision_frame = rospy.get_param(
            'ground_collision_frame', 'ground_plane::link::collision')

    # def transform_pose_to_world(self, p):
    #     # Convert this goal into the world frame using the current_pose

    #     return p
    def draw_new_goal(self, p):
        markerArray = MarkerArray()
        count = 0
        MARKERS_MAX = 100
        marker = Marker()
        marker.header.frame_id = "/delta/base_link"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = p.position.x
        marker.pose.position.y = p.position.y
        marker.pose.position.z = p.position.z

        # We add the new marker to the MarkerArray, removing the oldest
        # marker from it when necessary
        if (count > MARKERS_MAX):
            markerArray.markers.pop(0)

        markerArray.markers.append(marker)
        # Renumber the marker IDs
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        # Publish the MarkerArray
        self.sphere_marker_pub.publish(markerArray)

        count += 1

    def generate_new_goal(self):
        # Generate and return a pose in the sphere centered at the robot frame with radius as the goal_generation_radius

        # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/5838055#5838055
        goal = Pose()
        # sphere_marker_array = MarkerArray()
        u = random.random()
        v = random.random()
        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        while np.isnan(phi):
            phi = np.arccos(2.0 * v - 1.0)
        r = self.goal_generation_radius * np.cbrt(random.random())
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        rospy.loginfo_throttle(2, 'New Goal: (%.3f , %.3f , %.3f)', x, y, z)
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.orientation.x = 0
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 1
        # Convert this goal into the world frame and set it as the current goal
        # self.current_goal = self.transform_pose_to_world(goal)
        self.draw_new_goal(goal)
        return goal

    def reset_sim(self):
        rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        # Fill in the new position of the robot
        new_position = ModelState()
        new_position.model_name = 'delta'
        new_position.reference_frame = 'world'
        new_position.pose.position.x = 0
        new_position.pose.position.y = 0
        new_position.pose.position.z = 0
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
        rospy.loginfo('Placing robot')
        self.model_state_pub.publish(new_position)

        self.reset_timer()

        rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

    def reset_timer(self):
        rospy.loginfo('Resetting the timeout timer')
        self.timeout_timer.shutdown()
        self.timeout_timer = rospy.Timer(
            rospy.Duration(self.goal_generation_radius * 5), self.timer_cb)

    def reset_model(self):
        # To be used later to reset a specific model. This should not be used right now. Huan is checking how to reset a single model.
        pass

    def timer_cb(self, event):
        self.goal_pub.publish(self.generate_new_goal())


    def get_pose_diff(self, p1, p2):
        # At the moment, only return the translation difference. Maybe we should be sending the yaw error also
        position_1 = np.array([p1.position.x, p1.position.y, p1.position.z])
        position_2 = np.array([p2.position.x, p2.position.y, p2.position.z])
        return np.linalg.norm(position_1 - position_2)

    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose
        if self.current_goal is None:
            self.goal_pub.publish(self.generate_new_goal())
            rospy.sleep(10.0)
            return
        if self.get_pose_diff(self.current_pose,
                              self.current_goal) < self.waypoint_radius:
            # If the robot has reached the given goal pose, send the next waypoint and reset the timer
            self.reset_timer()
            self.goal_pub.publish(self.generate_new_goal())
            rospy.sleep(10.0)

    def contact_cb(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot_collision_frame):
                rospy.logdebug('Contact found!')
                if (msg.states[i].collision2_name ==
                        self.ground_collision_frame):
                    rospy.logdebug('Robot colliding with the ground')
                else:
                    rospy.logdebug(
                        'Robot colliding with something else (not ground)')
                    self.reset_sim()
            else:
                rospy.logdebug('Contact not found yet ...')


if __name__ == '__main__':
    dg = DataGenerator()
    rospy.loginfo('Ready')
    rospy.spin()