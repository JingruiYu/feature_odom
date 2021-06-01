#define ROS_ASSERT_ENABLED

#include <iostream>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/opencv.hpp>

static const double vehicle_length = 4.63;
static const double vehicle_width = 1.901;
static const double rear_axle_to_center = 1.393;

static const double pixel2meter = 0.03984;

// draw footprint
void drawFootprint(cv::Mat dst, std::vector<cv::Point2f> footprint,
                   cv::Scalar color)
{
  cv::Point points[1][4];
  points[0][0] = footprint[0];
  points[0][1] = footprint[1];
  points[0][2] = footprint[2];
  points[0][3] = footprint[3];

  const cv::Point* pts[] = {points[0]};
  int npts[] = {4};
  cv::polylines(dst, pts, npts, 1, true, color, 1, 8, 0);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_orb_feature_match_node");
  ros::NodeHandle nh;

  std::string ref_frame_path, test_frame_path;
  std::string ref_frame_mask_path, test_frame_mask_path;

  if (argc == 5)
  {
    ref_frame_path = argv[1];
    ref_frame_mask_path = argv[2];

    test_frame_path = argv[3];
    test_frame_mask_path = argv[4];
  }
  else
  {
    ROS_ERROR("give ref frame and test frame as arguments!");
    return 1;
  }

  cv::Mat ref_frame = cv::imread(ref_frame_path);
  cv::Mat test_frame = cv::imread(test_frame_path);

  if (ref_frame.empty() || test_frame.empty())
  {
    ROS_ERROR("Empty ref frame and/or test frame!");
    return 2;
  }

  // show input images
  //  cv::imshow("ref frame", ref_frame);
  //  cv::imshow("test frame", test_frame);
  //  cv::waitKey(1);

  // get mask
  cv::Mat ref_frame_mask = cv::imread(ref_frame_mask_path);
  cv::Mat test_frame_mask = cv::imread(test_frame_mask_path);

  if (ref_frame_mask.empty() || test_frame_mask.empty())
  {
    ROS_ERROR("Empty mask of ref frame and/or test frame!");
    return 3;
  }

  cv::cvtColor(ref_frame_mask, ref_frame_mask, cv::COLOR_BGR2GRAY);
  cv::cvtColor(test_frame_mask, test_frame_mask, cv::COLOR_BGR2GRAY);

  cv::threshold(ref_frame_mask, ref_frame_mask, 100, 255, cv::THRESH_BINARY);
  cv::threshold(test_frame_mask, test_frame_mask, 100, 255, cv::THRESH_BINARY);

  // show mask images
  //  cv::imshow("ref frame mask", ref_frame_mask);
  //  cv::imshow("test frame mask", test_frame_mask);
  //  cv::waitKey(1);

  // initialization
  std::vector<cv::KeyPoint> keypoints_ref, keypoints_test;
  cv::Mat descriptors_ref, descriptors_test;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1500,1.1,8,31,0,2,cv::ORB::FAST_SCORE,31,5);
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(1500,1.1,8,31,0,2,cv::ORB::FAST_SCORE,31,5);
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");

  // get orb feature points with mask
  detector->detect(ref_frame, keypoints_ref, ref_frame_mask);
  detector->detect(test_frame, keypoints_test, test_frame_mask);
  descriptor->compute(ref_frame, keypoints_ref, descriptors_ref);
  descriptor->compute(test_frame, keypoints_test, descriptors_test);

  cv::Mat outimg1, outimg2;
  cv::drawKeypoints(ref_frame, keypoints_ref, outimg1, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DEFAULT);
  cv::drawKeypoints(test_frame, keypoints_test, outimg2, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DEFAULT);
  cv::imshow("key orb feature points", outimg1);
  cv::imshow("test orb feature points", outimg2);
  cv::waitKey(1);

  // get feature matches
  std::vector<cv::DMatch> matches;

  // knn
  const float minRatio = 1.0f / 1.5f;
  const int k = 2;

  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher->knnMatch(descriptors_ref, descriptors_test, knnMatches, k);

  for (size_t i = 0; i < knnMatches.size(); i++)
  {
    const cv::DMatch& bestMatch = knnMatches[i][0];
    const cv::DMatch& betterMatch = knnMatches[i][1];
    float distanceRatio = bestMatch.distance / betterMatch.distance;
    if (bestMatch.distance < 30 && distanceRatio < minRatio)
      matches.push_back(bestMatch);
  }

  cv::Mat img_match_before_ransac;
  cv::drawMatches(ref_frame, keypoints_ref, test_frame, keypoints_test, matches,
                  img_match_before_ransac);

  cv::imshow("all matched key points before ransac", img_match_before_ransac);
  cv::waitKey(1);

  // RANSAC
  const int minNumbermatchesAllowed = 8;
  if (matches.size() < minNumbermatchesAllowed)
  {
    ROS_ERROR("too few matches: %lu < %d", matches.size(),
              minNumbermatchesAllowed);
    return 4;
  }

  // Prepare data for findHomography
  std::vector<cv::Point2f> refPoints(matches.size());
  std::vector<cv::Point2f> testPoints(matches.size());

  for (size_t i = 0; i < matches.size(); i++)
  {
    refPoints[i] = keypoints_ref[matches[i].queryIdx].pt;
    testPoints[i] = keypoints_test[matches[i].trainIdx].pt;
  }

  // find homography matrix and get inliers mask
  const double reprojectionThreshold = 3.0;
  std::vector<uchar> inliersMask(refPoints.size());
  cv::Mat homography = cv::findHomography(refPoints, testPoints, CV_FM_RANSAC,
                                          reprojectionThreshold, inliersMask);

  if (homography.empty())
  {
    ROS_ERROR("can not get a valid homography matrix!");
    return 5;
  }

  std::cout << "homography: \n" << homography << std::endl;

  std::vector<cv::DMatch> inliers;
  for (size_t i = 0; i < inliersMask.size(); i++)
  {
    if (inliersMask[i])
    {
      inliers.push_back(matches[i]);

      cv::Point2f refPoint = keypoints_ref[matches[i].queryIdx].pt;
      cv::Point2f testPoint = keypoints_test[matches[i].trainIdx].pt;

      ROS_INFO("ref pt: (%.1f, %.1f), test pt: (%.1f, %.1f)", refPoint.x,
               refPoint.y, testPoint.x, testPoint.y);
    }
  }
  matches.swap(inliers);

  // draw results
  cv::Mat img_match;
  cv::drawMatches(ref_frame, keypoints_ref, test_frame, keypoints_test, matches,
                  img_match);

  cv::imshow("final matched key points", img_match);
  cv::waitKey(1);

  // draw footprints to test frame
  int frame_width = test_frame.cols;
  int frame_height = test_frame.rows;

  cv::Point2f p1, p2, p3, p4;
  p1.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p1.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p2.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p2.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p3.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p3.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  p4.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p4.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  std::vector<cv::Point2f> footprint_test, footprint_ref;
  footprint_test.push_back(p1);
  footprint_test.push_back(p2);
  footprint_test.push_back(p3);
  footprint_test.push_back(p4);

  cv::perspectiveTransform(footprint_test, footprint_ref, homography);

  cv::Mat result_img_test = test_frame.clone();
  drawFootprint(result_img_test, footprint_test, cv::Scalar(0, 255, 0));
  drawFootprint(result_img_test, footprint_ref, cv::Scalar(255, 255, 0));

  // draw origin of base_link to test frame
  std::vector<cv::Point2f> base_origin_of_ref_in_test,
      base_origin_of_test_in_test;
  base_origin_of_test_in_test.push_back(cv::Point2f(
      frame_width / 2, frame_height / 2 + rear_axle_to_center / pixel2meter));

  cv::perspectiveTransform(base_origin_of_test_in_test,
                           base_origin_of_ref_in_test, homography);
  std::cout << "base_origin_of_ref_in_test: \n" << base_origin_of_ref_in_test
            << std::endl;

  cv::circle(result_img_test, cv::Point(base_origin_of_test_in_test[0].x,
                                        base_origin_of_test_in_test[0].y),
             2, cv::Scalar(0, 255, 0), -1);
  cv::circle(result_img_test, cv::Point(base_origin_of_ref_in_test[0].x,
                                        base_origin_of_ref_in_test[0].y),
             2, cv::Scalar(255, 255, 0), -1);

  // get homography from test frame to ref frame
  cv::Mat inv_h = homography.inv();
  std::cout << "inv_h: \n" << inv_h << std::endl;

  // draw footprint to ref frame
  footprint_test.clear();
  footprint_ref.clear();

  footprint_ref.push_back(p1);
  footprint_ref.push_back(p2);
  footprint_ref.push_back(p3);
  footprint_ref.push_back(p4);

  cv::perspectiveTransform(footprint_ref, footprint_test, inv_h);

  cv::Mat result_img_ref = ref_frame.clone();
  drawFootprint(result_img_ref, footprint_ref, cv::Scalar(255, 255, 0));
  drawFootprint(result_img_ref, footprint_test, cv::Scalar(0, 255, 0));

  // draw origin of base_link to ref frame
  std::vector<cv::Point2f> base_origin_of_ref_in_ref,
      base_origin_of_test_in_ref;
  base_origin_of_ref_in_ref = base_origin_of_test_in_test;
  cv::perspectiveTransform(base_origin_of_ref_in_ref,
                           base_origin_of_test_in_ref, inv_h);
  std::cout << "base_origin_of_test_in_ref: \n" << base_origin_of_test_in_ref
            << std::endl;

  cv::circle(result_img_ref, cv::Point(base_origin_of_ref_in_ref[0].x,
                                       base_origin_of_ref_in_ref[0].y),
             2, cv::Scalar(255, 255, 0), -1);
  cv::circle(result_img_ref, cv::Point(base_origin_of_test_in_ref[0].x,
                                       base_origin_of_test_in_ref[0].y),
             2, cv::Scalar(0, 255, 0), -1);

  // get delta odom
  double odom_dx =
      (base_origin_of_ref_in_ref[0].y - base_origin_of_test_in_ref[0].y) *
      pixel2meter;
  double odom_dy =
      (base_origin_of_ref_in_ref[0].x - base_origin_of_test_in_ref[0].x) *
      pixel2meter;
  double odom_th = atan2(inv_h.at<double>(0, 1), inv_h.at<double>(0, 0));

  ROS_INFO("Delta odom from test to ref: (%f m, %f m, %f rad)", odom_dx, odom_dy,
           odom_th);

  cv::imshow("result in test frame", result_img_test);
  cv::imshow("result in ref frame", result_img_ref);
  cv::waitKey(1);

  cv::waitKey(0);

  return 0;
}
