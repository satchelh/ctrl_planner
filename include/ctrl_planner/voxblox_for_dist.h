#ifndef VOX_FOR_DIST
#define VOX_FOR_DIST
#include "reward/map_manager_voxblox_impl.h"

#include <iostream>
#include <ros/ros.h>
#include <vector>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <cmath>

#include <geometry_msgs/PoseStamped.h>

class CollisionChecker
{
    public:

        CollisionChecker(ros::NodeHandle& nmain);

        MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>* _map_manager;

        // Eigen::Vector3d _current_xyz;
        Eigen::Vector3d _robot_size;

        void callback_pose(const geometry_msgs::PoseStamped::ConstPtr & msg);
        void check_collision_status()
        void is_colliding();

    private:

        bool _collision_free
        ros::ServiceServer _srv_collision_check;

        std::string _model_name;
        std::string _topic_pose;

        std::deque<Eigen::Vector3d> _pose_Q;
        int _num_pose;
        int _pose_Qsize;
        
        ros::Subscriber _sub_pose;
};


#endif