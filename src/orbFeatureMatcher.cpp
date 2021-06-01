#include "orbFeatureMatcher.h"

namespace feature_match_odom
{
orbFeatureMatcher::orbFeatureMatcher(std::shared_ptr<frame> ref_ptr,
                                     std::shared_ptr<frame> test_ptr)
{
  frame_ref_ptr_ = ref_ptr;
  frame_test_ptr_ = test_ptr;

  matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

void orbFeatureMatcher::calcKnnMatches()
{
  // get feature matches using knn-matcher
  const float max_distance = 60.0;
  const float minRatio = 1.0f / 1.2f;
  const int k = 2;

  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher_->knnMatch(frame_ref_ptr_->descriptors, frame_test_ptr_->descriptors,
                     knnMatches, k);

  for (size_t i = 0; i < knnMatches.size(); i++)
  {
    const cv::DMatch& bestMatch = knnMatches[i][0];
    const cv::DMatch& betterMatch = knnMatches[i][1];
    float distanceRatio = bestMatch.distance / betterMatch.distance;
    if (bestMatch.distance < max_distance && distanceRatio < minRatio)
      matches_.push_back(bestMatch);
  }
}

void orbFeatureMatcher::calcDirectMatches(
    const nav_msgs::Odometry& ref_odom_msg,
    const nav_msgs::Odometry& test_odom_msg)
{
  ros::Time tic, toc;
  tic = ros::Time::now();

  matches_.clear();

  // get transform: test base_footprint -> ref base_footprint
  tf::Transform tf_base_test2odom;
  tf_base_test2odom.setOrigin(tf::Vector3(test_odom_msg.pose.pose.position.x,
                                          test_odom_msg.pose.pose.position.y,
                                          0.0));
  tf_base_test2odom.setRotation(tf::createQuaternionFromYaw(
      tf::getYaw(test_odom_msg.pose.pose.orientation)));

  tf::Transform tf_base_ref2odom;
  tf_base_ref2odom.setOrigin(tf::Vector3(ref_odom_msg.pose.pose.position.x,
                                         ref_odom_msg.pose.pose.position.y,
                                         0.0));
  tf_base_ref2odom.setRotation(tf::createQuaternionFromYaw(
      tf::getYaw(ref_odom_msg.pose.pose.orientation)));

  tf::Transform tf_test2ref = tf_base_ref2odom.inverse() * tf_base_test2odom;

  ROS_DEBUG("tf_test2ref: (%f, %f, %f)", tf_test2ref.getOrigin().x(),
            tf_test2ref.getOrigin().y(), tf::getYaw(tf_test2ref.getRotation()));

  // get transform: ref base_footprint -> test base_footprint
  tf::Transform tf_ref2test = tf_test2ref.inverse();

  ROS_DEBUG("tf_ref2test: (%f, %f, %f)", tf_ref2test.getOrigin().x(),
            tf_ref2test.getOrigin().y(), tf::getYaw(tf_ref2test.getRotation()));

  toc = ros::Time::now();
  ROS_DEBUG("get transform: %.6f s", (toc - tic).toSec());

  tic = ros::Time::now();
  // Forward transform to test frame and get matched result
  float max_euclidean_dist_thresh = 0.1; // m
  float max_euclidean_dist_thresh_pixel =
      max_euclidean_dist_thresh / pixel2meter;

  float max_hamming_dist_thresh = 180;

  int frame_width = frame_ref_ptr_->frame_image.cols;
  int frame_height = frame_ref_ptr_->frame_image.rows;

  for (int i = 0; i < frame_ref_ptr_->keypoints.size(); i++)
  {
    tf::Point pt_ref;
    pt_ref.setX((frame_height / 2 - frame_ref_ptr_->keypoints[i].pt.y) *
                    pixel2meter +
                rear_axle_to_center);
    pt_ref.setY((frame_width / 2 - frame_ref_ptr_->keypoints[i].pt.x) *
                pixel2meter);

    tf::Point pt_test = tf_ref2test * pt_ref;

    cv::Point2f pt_pixel_test(
        frame_width / 2 - pt_test.y() / pixel2meter,
        frame_height / 2 - (pt_test.x() - rear_axle_to_center) / pixel2meter);

    ROS_DEBUG("Pair pt: ref (%f, %f), test (%f, %f)", pt_ref.x(), pt_ref.y(),
              pt_test.x(), pt_test.y());
    ROS_DEBUG("Pair pixel: ref (%f, %f), test (%f, %f)",
              frame_ref_ptr_->keypoints[i].pt.x,
              frame_ref_ptr_->keypoints[i].pt.y, pt_pixel_test.x,
              pt_pixel_test.y);

    // find the closest corresponding points in test frame
    for (int j = 0; j < frame_test_ptr_->keypoints.size(); j++)
    {
      // calculate euclidean distance
      float dist_e =
          std::hypot(pt_pixel_test.x - frame_test_ptr_->keypoints[j].pt.x,
                     pt_pixel_test.y - frame_test_ptr_->keypoints[j].pt.y);
      if (dist_e < max_euclidean_dist_thresh_pixel)
      {
        // calculate hamming distance
        float dist_h =
            cv::norm(frame_ref_ptr_->descriptors.row(i),
                     frame_test_ptr_->descriptors.row(j), cv::NORM_HAMMING);

        ROS_DEBUG("ref #%d and test #%d dist_e: %f dist_h: %f", i, j, dist_e,
                  dist_h);

        if (dist_h < max_hamming_dist_thresh)
        {
          cv::DMatch m(i, j, dist_h);
          matches_.push_back(m);
        }
      }
    }
  }

  toc = ros::Time::now();
  ROS_DEBUG("Forward transform: %.6f s", (toc - tic).toSec());

  //  tic = ros::Time::now();
  //  // Backward transform to ref frame and get matched result
  //  for (int i = 0; i < frame_test_ptr_->keypoints.size(); i++)
  //  {
  //    tf::Point pt_test;
  //    pt_test.setX((frame_height / 2 - frame_test_ptr_->keypoints[i].pt.y) *
  //                     pixel2meter +
  //                 rear_axle_to_center);
  //    pt_test.setY((frame_width / 2 - frame_test_ptr_->keypoints[i].pt.x) *
  //                 pixel2meter);

  //    tf::Point pt_ref = tf_test2ref * pt_test;
  //    cv::Point2f pt_pixel_ref(
  //        frame_width / 2 - pt_ref.y() / pixel2meter,
  //        frame_height / 2 - (pt_ref.x() - rear_axle_to_center) /
  //        pixel2meter);

  //    // find the closest corresponding points in ref frame
  //    for (int j = 0; j < frame_ref_ptr_->keypoints.size(); j++)
  //    {
  //      float dist_e =
  //          std::hypot(pt_pixel_ref.x - frame_ref_ptr_->keypoints[j].pt.x,
  //                     pt_pixel_ref.y - frame_ref_ptr_->keypoints[j].pt.y);
  //      if (dist_e < max_euclidean_dist_thresh_pixel)
  //      {
  //        // calculate hamming distance
  //        float dist_h =
  //            cv::norm(frame_test_ptr_->descriptors.row(i),
  //                     frame_ref_ptr_->descriptors.row(j), cv::NORM_HAMMING);

  //        if (dist_h < max_hamming_dist_thresh)
  //        {
  //          cv::DMatch m(j, i, dist_h);
  //          matches_.push_back(m);
  //        }
  //      }
  //    }
  //  }
  //  toc = ros::Time::now();
  //  ROS_INFO("Backward transform: %.6f s", (toc - tic).toSec());

  //  tic = ros::Time::now();
  //  ROS_DEBUG("matches num before removing duplicate: %lu", matches_.size());

  //  // remove duplicate matches
  //  for (int i = 0; i < matches_.size(); i++)
  //  {
  //    for (int j = i + 1; j < matches_.size(); j++)
  //    {
  //      if (matches_[i].queryIdx == matches_[j].queryIdx &&
  //          matches_[i].trainIdx == matches_[j].trainIdx)
  //      {
  //        matches_.erase(matches_.begin() + j);
  //        j--;
  //      }
  //    }
  //  }
  //  ROS_DEBUG("matches num after removing duplicate: %lu", matches_.size());
  //  toc = ros::Time::now();
  //  ROS_INFO("remove dup: %.6f s", (toc - tic).toSec());
}

std::vector<cv::DMatch> orbFeatureMatcher::getMatches() { return matches_; }

int orbFeatureMatcher::getMatchesNum() { return matches_.size(); }

cv::Mat orbFeatureMatcher::visualizeMatches()
{
  cv::Mat img_match;
  cv::drawMatches(frame_ref_ptr_->frame_image, frame_ref_ptr_->keypoints,
                  frame_test_ptr_->frame_image, frame_test_ptr_->keypoints,
                  matches_, img_match, cv::Scalar(0, 255, 0),
                  cv::Scalar(125, 125, 125));

  cv::imshow("matched key points", img_match);
  cv::waitKey(1);

  return img_match;
}

cv::Mat orbFeatureMatcher::getHomography()
{
  const int minNumbermatchesAllowed = 8;
  if (matches_.size() < minNumbermatchesAllowed)
  {
    ROS_ERROR("too few matches: %lu < %d", matches_.size(),
              minNumbermatchesAllowed);
    return cv::Mat();
  }

  // Prepare data for findHomography
  std::vector<cv::Point2f> refPoints(matches_.size());
  std::vector<cv::Point2f> testPoints(matches_.size());

  for (size_t i = 0; i < matches_.size(); i++)
  {
    refPoints[i] = frame_ref_ptr_->keypoints[matches_[i].queryIdx].pt;
    testPoints[i] = frame_test_ptr_->keypoints[matches_[i].trainIdx].pt;
  }

  // find homography matrix and get inliers mask
  const double reprojectionThreshold = 3.0;
  std::vector<uchar> inliersMask(refPoints.size());
  cv::Mat homography =
      cv::findHomography(refPoints, testPoints, CV_FM_RANSAC,
                         reprojectionThreshold, inliersMask, 5000);

  // refine matches
  std::vector<cv::DMatch> inliers;
  for (size_t i = 0; i < inliersMask.size(); i++)
  {
    if (inliersMask[i])
    {
      inliers.push_back(matches_[i]);

      cv::Point2f refPoint = frame_ref_ptr_->keypoints[matches_[i].queryIdx].pt;
      cv::Point2f testPoint =
          frame_test_ptr_->keypoints[matches_[i].trainIdx].pt;

      ROS_DEBUG("ref pt: (%.1f, %.1f), test pt: (%.1f, %.1f)", refPoint.x,
                refPoint.y, testPoint.x, testPoint.y);
    }
  }
  matches_.swap(inliers);

  return homography;
}
}
