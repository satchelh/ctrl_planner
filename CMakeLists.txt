cmake_minimum_required(VERSION 2.8.3)
project(ctrl_planner)

add_definitions(-std=c++11)

find_package(catkin_simple REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  mav_msgs
)


# ## System dependencies are found with CMake's conventions
find_package(Boost 1.58 REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )


catkin_package(
  CATKIN_DEPENDS
    mav_msgs
    roscpp
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/ctrl_planner.cpp
# )

add_executable(test_voxblox src/test_voxblox.cpp)
target_link_libraries(test_voxblox ${catkin_LIBRARIES})

add_executable(wait_for_gazebo src/wait_for_gazebo.cpp)
target_link_libraries(wait_for_gazebo ${catkin_LIBRARIES})
# add_dependencies(hovering_example ${catkin_EXPORTED_TARGETS})
# add_dependencies(test_voxblox
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )



