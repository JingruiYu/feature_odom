#ifndef FRAME_H
#define FRAME_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

namespace feature_match_odom
{
class frame
{
public:
  frame();
  frame(ros::Time stamp, cv::Mat& frame_image, cv::Mat& mask_image);

  // detect orb keypoints
  void detectKeypoints();

  // compute orb features
  void computeFeature();

  // visualize orb features
  void visualizeFeature();

  // clone
  frame clone();

  ros::Time timestamp;

  cv::Mat frame_image;
  cv::Mat mask_image;

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  double odom_x, odom_y, odom_th;

private:
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor;
};
}

#endif // FRAME_H
