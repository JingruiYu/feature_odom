cmake_minimum_required(VERSION 2.8.3)
project(feature_match_odom)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  roslib
  sensor_msgs
  tf
  state_machine_msgs
  vehicle_msgs
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_libraries(
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

add_library(orb_feature_match
    src/frame.cpp
    src/orbFeatureMatcher.cpp
    src/featureMatchOdom.cpp
    src/lkFeatureTracker.cpp
    src/optimization.cpp
)

add_executable(feature_match_odom_node src/feature_match_odom_node.cpp)
target_link_libraries(feature_match_odom_node orb_feature_match)

# test nodes
add_executable(test_subscribe_compressed_image_node test/test_subscribe_compressed_image_node.cpp)
target_link_libraries(test_subscribe_compressed_image_node)

add_executable(test_orb_feature_match_node test/test_orb_feature_match_node.cpp)
target_link_libraries(test_orb_feature_match_node)

add_executable(prepare_images_with_odom test/prepare_images_with_odom.cpp)
target_link_libraries(prepare_images_with_odom)

add_executable(test_orb_feature_match_lib_node test/test_orb_feature_match_lib_node.cpp)
target_link_libraries(test_orb_feature_match_lib_node orb_feature_match)

add_executable(test_performance test/test_performance.cpp)
target_link_libraries(test_performance)

add_executable(test_lk test/test_lk.cpp)
target_link_libraries(test_lk)

add_executable(prepare_test_data_for_ba test/prepare_test_data_for_ba.cpp)
target_link_libraries(prepare_test_data_for_ba orb_feature_match)

add_executable(prepare_images_for_labling test/prepare_images_for_labling.cpp)
target_link_libraries(prepare_images_for_labling)
