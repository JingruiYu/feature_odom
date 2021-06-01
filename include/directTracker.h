#ifndef DIRECTTRACKER_H
#define DIRECTTRACKER_H

#include <tf/tf.h>

#include "param.h"
#include "frame.h"

namespace feature_match_odom
{
static const int success_track_thresh = 20;

class directTracker
{
public:
  directTracker();

  void initializeTracker(std::shared_ptr<frame> frame_ptr);

  bool trackKeypoints(std::shared_ptr<frame> frame_ptr,
                      std::vector<bool>& tracked_result_ref,
                      std::vector<bool>& tracked_reslut_cur);

private:
  std::shared_ptr<frame> ref_frame_ptr_;

  bool initialized_;
};
}

#endif // DIRECTTRACKER_H
