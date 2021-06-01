# Feature match odom

This repository implements an ORB feature based visual odometry.

1. [Dependencies](#dependencies)
2. [Running](#running)

## Dependencies
1. Install Ceres.
2. Download other dependent ROS packages.
3. Prepare the ROS bag file.

## Running
1. Launch demo.launch file:
```
roslaunch feature_match_odom demo.launch
```

2. Play the ROS bag file:
```
rosbag play --clock /path/to/bag/file.bag
```