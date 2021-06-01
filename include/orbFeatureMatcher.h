#ifndef ORBFEATUREMATCHER_H
#define ORBFEATUREMATCHER_H

#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>

#include "param.h"
#include "frame.h"
#include "optimization.h"

namespace feature_match_odom
{
class orbFeatureMatcher
{
public:
  orbFeatureMatcher(std::shared_ptr<frame> ref_ptr,
                    std::shared_ptr<frame> test_ptr);

  // calculate knn matches
  void calcKnnMatches();

  // calculate direct matches
  void calcDirectMatches(const nav_msgs::Odometry& ref_odom_msg,
                         const nav_msgs::Odometry& test_odom_msg);

  // get matches
  std::vector<cv::DMatch> getMatches();

  // get number of matches
  int getMatchesNum();

  // visualize matches
  cv::Mat visualizeMatches();

  // get homography
  cv::Mat getHomography();

private:
  cv::Ptr<cv::DescriptorMatcher> matcher_;

  std::shared_ptr<frame> frame_ref_ptr_;
  std::shared_ptr<frame> frame_test_ptr_;

  std::vector<cv::DMatch> matches_;
};
}

#endif // ORBFEATUREMATCHER_H
