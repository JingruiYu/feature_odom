#include "lkFeatureTracker.h"

namespace feature_match_odom
{
lkFeatureTracker::lkFeatureTracker() : initialized_(false) {}

void lkFeatureTracker::initializeTracker(std::shared_ptr<frame> frame_ptr)
{
  prev_frame_ptr_ = frame_ptr;
  initialized_ = true;
}

bool lkFeatureTracker::trackKeypoints(std::shared_ptr<frame> frame_ptr)
{
  if (!initialized_)
  {
    ROS_ERROR(
        "trackKeypoints: Tracking failed! Tracker has not been initialized!");
    return false;
  }

  if (prev_frame_ptr_->keypoints.size() < minKeypointCount)
  {
    ROS_WARN("trackKeypoints: Tracking failed! Too few keypoints (%lu) in "
             "previous frame!",
             prev_frame_ptr_->keypoints.size());
    prev_frame_ptr_ = frame_ptr;
    return false;
  }

  // prepare gray images
  cv::Mat current_frame = frame_ptr->frame_image.clone();
  cv::Mat prev_frame = prev_frame_ptr_->frame_image.clone();
  cv::Mat current_frame_gray, prev_frame_gray;
  cv::cvtColor(current_frame, current_frame_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(prev_frame, prev_frame_gray, cv::COLOR_BGR2GRAY);

  // prepare points constainers
  std::vector<cv::Point2f> prevPts, curPts;
  cv::KeyPoint::convert(prev_frame_ptr_->keypoints, prevPts);

  // calculate tracked keypoints
  std::vector<uchar> status;
  std::vector<float> err;

  cv::calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, prevPts, curPts,
                           status, err, winSize, 3, termcrit, 0, 0.001);

  std::vector<cv::Point2f> curPts_valid;
  for (int i = 0; i < status.size(); i++)
  {
    if (status[i])
      curPts_valid.push_back(curPts[i]);
  }

  ROS_INFO("Prev points num: %lu, Tracked points num: %lu, valid num: %lu",
           prevPts.size(), curPts.size(), curPts_valid.size());

  if (curPts_valid.size() < minKeypointCount)
  {
    // detect keypoints when too few tracked keypoints
    ROS_WARN("trackKeypoints: Tracking failed! Not enough tracking successful "
             "points!");
    frame_ptr->detectKeypoints();
  }
  else
    // update keypoints to input frame
    cv::KeyPoint::convert(curPts_valid, frame_ptr->keypoints);

  // update to previous frame
  prev_frame_ptr_ = frame_ptr;

  return true;
}
}
