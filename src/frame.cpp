#include "frame.h"

namespace feature_match_odom
{
frame::frame() {}

frame::frame(ros::Time stamp, cv::Mat& image, cv::Mat& mask)
    : timestamp(stamp), odom_x(0.0), odom_y(0.0), odom_th(0.0)
{
  detector = cv::ORB::create(500);
  descriptor = cv::ORB::create(500);

  frame_image = image.clone();
  mask_image = mask.clone();

  // process mask
  if (!mask_image.empty())
  {
    if (mask_image.channels() != 1)
      cv::cvtColor(mask_image, mask_image, cv::COLOR_BGR2GRAY);

    cv::threshold(mask_image, mask_image, 100, 255, cv::THRESH_BINARY);
  }
}
void frame::detectKeypoints()
{
  detector->detect(frame_image, keypoints, mask_image);
}

void frame::computeFeature()
{
  descriptor->compute(frame_image, keypoints, descriptors);
}

void frame::visualizeFeature()
{
  cv::Mat result;
  cv::drawKeypoints(frame_image, keypoints, result, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DEFAULT);
  cv::imshow("orb feature points", result);
  cv::waitKey(1);
}

frame frame::clone()
{
  frame tmp;
  tmp.timestamp = timestamp;
  tmp.frame_image = frame_image.clone();
  tmp.mask_image = mask_image.clone();
  tmp.keypoints = keypoints;
  tmp.descriptors = descriptors;
  tmp.odom_x = odom_x;
  tmp.odom_y = odom_y;
  tmp.odom_th = odom_th;
  return tmp;
}
}
