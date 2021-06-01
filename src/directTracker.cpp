#include "directTracker.h"

namespace feature_match_odom
{
directTracker::directTracker() : initialized_(false) {}

void directTracker::initializeTracker(std::shared_ptr<frame> frame_ptr)
{
  ref_frame_ptr_ = frame_ptr;
  initialized_ = true;
}

bool directTracker::trackKeypoints(std::shared_ptr<frame> frame_ptr,
                                   std::vector<bool>& tracked_result_ref,
                                   std::vector<bool>& tracked_reslut_cur)
{
  // get keypoints in current frame and transform to base_footprint
  int frame_width = frame_ptr->frame_image.cols;
  int frame_height = frame_ptr->frame_image.rows;

  std::vector<tf::Point> cur_kpts;
  for (auto kp : frame_ptr->keypoints)
  {
    tf::Point tmp;
    tmp.setX((frame_height / 2 - kp.pt.x) * pixel2meter + rear_axle_to_center);
    tmp.setY((frame_width / 2 - kp.pt.y) * pixel2meter);
    cur_kpts.push_back(tmp);
  }

  // get transform: current base_footprint -> ref base_footprint
  tf::Transform tf_cur2ref;

  // transform to ref frame and get tracked result
  double dist_thresh = 0.1; // m
  double dist_thresh_pixel = dist_thresh / pixel2meter;

  tracked_result_ref.resize(ref_frame_ptr_->keypoints.size(), false);
  tracked_reslut_cur.resize(frame_ptr->keypoints.size(), false);

  for (int i = 0; i < cur_kpts.size(); i++)
  {
    tf::Point pt_ref = tf_cur2ref * cur_kpts[i];
    cv::Point2f pt_pixel_ref(
        frame_height / 2 - (pt_ref.x() - rear_axle_to_center) / pixel2meter,
        frame_width / 2 - pt_ref.y() / pixel2meter);

    // find the closest corresponding points in ref frame
    double min_dist = 1e3;
    int min_dist_index = -1;
    for (int j = 0; j < ref_frame_ptr_->keypoints.size(); j++)
    {
      double dist =
          std::hypot(pt_pixel_ref.x - ref_frame_ptr_->keypoints[j].pt.x,
                     pt_pixel_ref.y - ref_frame_ptr_->keypoints[j].pt.y);
      if (dist < dist_thresh_pixel && dist < min_dist)
      {
        min_dist = dist;
        min_dist_index = j;
      }
    }

    // record in results
    if (min_dist_index >= 0)
    {
      tracked_reslut_cur[i] = true;
      tracked_result_ref[min_dist_index] = true;
    }
  }

  // show number of tracked points
  int tracked_cur_cnt = 0, tracked_ref_cnt = 0;
  for (auto s : tracked_reslut_cur)
  {
    if (s)
      tracked_cur_cnt++;
  }
  for (auto s : tracked_result_ref)
  {
    if (s)
      tracked_ref_cnt++;
  }
  ROS_INFO("Tracked points num: current - %d, ref - %d", tracked_cur_cnt,
           tracked_ref_cnt);

}
}
