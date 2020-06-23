#ifndef VOX_FOR_DIST
#define VOX_FOR_DIST
#include "reward/map_manager_voxblox_impl.h"
#include <vector>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <cmath>

#include <ros/ros.h>

#include <iostream>

class GetDistances
{
    public:
        MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>* _map_manager;

        Eigen::Vector3d _target_xyz;
        Eigen::Vector3d _robot_size;

    private:
        ros::ServiceServer srv_closest_occ_dist;
};


#endif