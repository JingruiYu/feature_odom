#define ROS_ASSERT_ENABLED

#include <iostream>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>

#include <opencv2/opencv.hpp>

#include "frame.h"
#include "orbFeatureMatcher.h"

using namespace feature_match_odom;

// static const double vehicle_length = 4.63;
// static const double vehicle_width = 1.901;
// static const double rear_axle_to_center = 1.393;

// static const double pixel2meter = 0.03984;

std::vector<ros::Duration> timing;
std::vector<std::string> timing_descript;

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
  ros::init(argc, argv, "test_orb_feature_match_lib_node");
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
    return 1;
  }

  // get mask
  cv::Mat ref_frame_mask = cv::imread(ref_frame_mask_path);
  cv::Mat test_frame_mask = cv::imread(test_frame_mask_path);

  if (ref_frame_mask.empty() || test_frame_mask.empty())
  {
    ROS_ERROR("Empty mask of ref frame and/or test frame!");
    return 1;
  }

  cv::cvtColor(ref_frame_mask, ref_frame_mask, cv::COLOR_BGR2GRAY);
  cv::threshold(ref_frame_mask, ref_frame_mask, 100, 255, cv::THRESH_BINARY);
  cv::cvtColor(test_frame_mask, test_frame_mask, cv::COLOR_BGR2GRAY);
  cv::threshold(test_frame_mask, test_frame_mask, 100, 255, cv::THRESH_BINARY);

  // ignore footprint
  double boundary = 20.0;
  int frame_width = ref_frame.cols;
  int frame_height = ref_frame.rows;

  double x = frame_width / 2 - (vehicle_width / 2 / pixel2meter) - boundary;
  double y = frame_height / 2 - (vehicle_length / 2 / pixel2meter) - boundary;
  double width = vehicle_width / pixel2meter + 2 * boundary;
  double height = vehicle_length / pixel2meter + 2 * boundary;

  cv::rectangle(ref_frame_mask, cv::Rect(x, y, width, height), cv::Scalar(0),
                -1);
  cv::rectangle(test_frame_mask, cv::Rect(x, y, width, height), cv::Scalar(0),
                -1);

  // initialization
  ros::Time tic = ros::Time::now();

  frame ref(ros::Time::now(), ref_frame, ref_frame_mask);

  ros::Time toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("frame construction");

  frame test(ros::Time::now(), test_frame, test_frame_mask);

  // extract feature
  tic = ros::Time::now();
  ref.detectKeypoints();
  ref.computeFeature();
  toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("feature extraction");

  ref.visualizeFeature();

  test.detectKeypoints();
  test.computeFeature();
  test.visualizeFeature();

  // calculate feature matches
  tic = ros::Time::now();
  orbFeatureMatcher matcher(std::make_shared<frame>(ref),
                            std::make_shared<frame>(test));

  matcher.calcKnnMatches();

  toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("calculate knn-matches");

  matcher.visualizeMatches();
  cv::waitKey(0);

  tic = ros::Time::now();
  nav_msgs::Odometry ref_odom_msg, test_odom_msg;

  ref_odom_msg.pose.pose.position.x = 0.958597;
  ref_odom_msg.pose.pose.position.y = -0.0146898;
  ref_odom_msg.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(-0.048469);

  test_odom_msg.pose.pose.position.x = 1.98602;
  test_odom_msg.pose.pose.position.y = -0.130937;
  test_odom_msg.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(-0.190162);

  matcher.calcDirectMatches(ref_odom_msg, test_odom_msg);

  toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("calculate direct matches");

  matcher.visualizeMatches();
  cv::waitKey(0);

  // calculate homography
  tic = ros::Time::now();
  cv::Mat homography = matcher.getHomography();
  toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("calculate homography");

  matcher.visualizeMatches();

  if (homography.empty())
  {
    ROS_ERROR("can not get a valid homography matrix!");
    return 1;
  }

  std::cout << "homography: \n" << homography << std::endl;

  // draw footprints to test frame
  tic = ros::Time::now();
  //  int frame_width = test_frame.cols;
  //  int frame_height = test_frame.rows;

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
  toc = ros::Time::now();
  timing.push_back(toc - tic);
  timing_descript.push_back("transform footprint");

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

  ROS_INFO("Delta odom from test to ref: (%f m, %f m, %f rad)", odom_dx,
           odom_dy, odom_th);

  cv::imshow("result in test frame", result_img_test);
  cv::imshow("result in ref frame", result_img_ref);
  cv::waitKey(1);

  // output timing result
  ros::Duration total;
  for (auto iter = timing.begin(); iter != timing.end(); iter++)
  {
    ROS_INFO("time step: %f (%s)", (*iter).toSec(),
             timing_descript[iter - timing.begin()].c_str());
    total += *iter;
  }
  ROS_INFO("total time: %f", total.toSec());

  cv::waitKey(0);

  return 0;
}
