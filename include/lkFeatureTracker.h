#ifndef LKFEATURETRACKER_H
#define LKFEATURETRACKER_H

#include "frame.h"

namespace feature_match_odom
{

static cv::TermCriteria
    termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
static cv::Size winSize(21, 21);

static const int minKeypointCount = 200;

class lkFeatureTracker
{
public:
  lkFeatureTracker();

  // initialize tracker
  void initializeTracker(std::shared_ptr<frame> frame_ptr);

  // track keypoints
  bool trackKeypoints(std::shared_ptr<frame> frame_ptr);

private:
  std::shared_ptr<frame> prev_frame_ptr_;

  bool initialized_;
};
}

#endif // LKFEATURETRACKER_H
