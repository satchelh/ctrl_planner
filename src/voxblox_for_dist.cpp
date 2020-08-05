# include "ctrl_planner/voxblox_for_dist.h"


CollisionChecker::CollisionChecker(
    ros::NodeHandle& nmain
    ros::NodeHandle & n, 
    ros::NodeHandle & nh_private)
{
    
    _map_manager = new MapManagerVoxblox<MapManagerVoxbloxServer,MapManagerVoxbloxVoxel>(n, nh_private);

    _pose_Qsize = 3;
    _num_pose = 0;
    _collision_free = false;

    _model_name = 'elios_VLP16';
    _topic_pose = '/' + _model_name + '/ground_truth/pose';

    _sub_pose = nmain.subscribe(
        _topic_pose, 1, & CollisionChecker::callback_pose, this);

    
    return;
}

void CollisionChecker::callback_pose(
    const geometry_msgs::PoseStamped::ConstPtr& msg
)
{
    geometry_msgs::PoseStamped _msg_in = std::remove_const<std::remove_reference<decltype(*msg)>::type>::type(*msg);
    double x = _msg_in.pose.position.x;
    double y = _msg_in.pose.position.y;
    double z = _msg_in.pose.position.z;
    
    // add most recent pose to pose Q
    Eigen::Vector3d current_xyz;
    current_xyz << x, y, z;
    _pose_Q.push_front(current_xyz);
    _num_pose += 1;
    if( _num_pose >= _pose_Qsize )
    {
        _pose_Q.pop_back();
        _num_pose -= 1;
    }

    return;
}

void CollisionChecker::check_collision_status()
{
    Eigen::Vector3d curr_pose = _pose_Q[0]
    
}

void CollisionChecker::is_colliding()
{
    return _collision_free;
}