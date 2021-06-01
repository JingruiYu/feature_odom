#ifndef FEATUREMATCHODOM_H
#define FEATUREMATCHODOM_H

#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "param.h"
#include "frame.h"
#include "orbFeatureMatcher.h"
#include "lkFeatureTracker.h"
#include "optimization.h"

namespace feature_match_odom
{

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;

class featureMatchOdom
{

public:
  featureMatchOdom(ros::NodeHandle nh, ros::NodeHandle pnh);
  ~featureMatchOdom();

  void startLocalBa();

protected:
  // update frame
  void updateFrame(const sensor_msgs::ImageConstPtr& msg);

  // update mask
  void updateMask(const sensor_msgs::ImageConstPtr& msg);

  // update vehicle odom
  void updateVehicleOdom(const nav_msgs::OdometryConstPtr& msg);

  // process frame
  virtual void processFrame(const ros::TimerEvent& event);

  // calculate result and show in ref frame
  void calcResultRefFrame(cv::Point2f& base_origin_of_test_in_ref,
                          cv::Point2f& base_origin_of_ref_in_ref);

  // check to update ref frame
  virtual void checkUpdateRefFrame();

  // prepare data for optimizer
  virtual void prepareDataForRefOptimizer(
      std::vector<Pose2d>& poses, std::vector<Eigen::Vector2d>& points,
      std::vector<std::vector<Observation>>& observations);

  // transform a point in birdview image to base_footprint
  void transformToFootprint(const cv::Point2f& p_src, cv::Point2f& p_dst);

  // calculate odom msg and publish it
  void calcAndPublishOdom(cv::Point2f base_origin_of_ref_in_ref,
                          cv::Point2f base_origin_of_test_in_ref);

  // check homography validity by vehicle footprint shape
  bool checkHomographyVehicleFootprint();

  // check relative pose validity by vehicle odom and update it
  void checkWithVehicleOdom(double& dx_base, double& dy_base, double& dth_base,
                            bool direct_replace);

  // check if matched points is enough with a threshold
  virtual bool checkEnoughMatchedPoint(int min_num);

  // draw footprint
  void drawFootprint(cv::Mat dst, std::vector<cv::Point2f> footprint,
                     cv::Scalar color);

  virtual void featurePointToCloud(std::vector<tf::Point>& feature_pts,
                                   sensor_msgs::PointCloud2& cloud,
                                   bool is_local, double intensity);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;

  ros::Timer process_timer_;

  image_transport::Subscriber birdview_img_sub_;
  image_transport::Subscriber birdview_mask_sub_;
  image_transport::Publisher matched_img_pub_;
  image_transport::Publisher result_img_pub_;
  ros::Subscriber vehicle_odom_sub_;
  ros::Publisher feature_match_odom_pub_;
  ros::Publisher total_feature_cloud_pub_;
  ros::Publisher test_frame_feature_cloud_pub_;
  ros::Publisher ref_frame_feature_cloud_pub_;

  tf::TransformBroadcaster odom_feature_broadcaster_;

  cv::Mat current_frame_, current_mask_;
  std::shared_ptr<frame> ref_frame_ptr_, test_frame_ptr_;
  std::vector<frame> frame_seq_;
  std::vector<std::vector<cv::DMatch>> matches_seq_;

  int total_frame_num_;
  int valid_frame_num_;

  nav_msgs::Odometry current_vehicle_odom_msg_;
  nav_msgs::Odometry feature_match_odom_msg_;
  nav_msgs::Odometry ref_vehicle_odom_msg_;
  nav_msgs::Odometry test_vehicle_odom_msg_;

  ros::Time current_frame_time_, current_mask_time_;

  std::shared_ptr<orbFeatureMatcher> matcher_ptr_;
  lkFeatureTracker tracker_;

  bool initialized_;
  bool frame_updated_, mask_updated_;
  bool vehicle_odom_updated_;

  std::vector<tf::Point> total_feature_points_;

  cv::Mat homography_;

  std::thread thread_local_ba_;
  std::mutex mtx_local_ba_;
};
}
#endif // FEATUREMATCHODOM_H
