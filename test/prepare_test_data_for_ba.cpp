#include <iostream>
#include <fstream>
#include <algorithm>

#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include "frame.h"
#include "orbFeatureMatcher.h"

using namespace feature_match_odom;

struct odom
{
  double x = 0.0;
  double y = 0.0;
  double th = 0.0;
};

struct observation
{
  int pose_ind;
  int keypoint_ind;
  cv::Point2f keypoint;
};

struct trans
{
  double x = 0.0;
  double y = 0.0;
  double th = 0.0;
};

std::vector<frame> frame_seq;
std::vector<odom> odom_seq;

std::shared_ptr<frame> ref_frame_ptr;
std::vector<cv::Point2f> ref_keypoints;

std::vector<observation> observation_seq;

std::vector<trans> estimate_trans_seq;

int once_matched_pts_num = 0;

bool loadDataFromFile(std::string input_file_path)
{

  std::ifstream data_file;
  data_file.open(input_file_path);

  while (!data_file.eof())
  {
    uint64_t timestamp;
    std::string image_path;
    cv::Mat image;
    odom vehicle_odom;

    data_file >> timestamp;
    data_file >> image_path;
    data_file >> vehicle_odom.x;
    data_file >> vehicle_odom.y;
    data_file >> vehicle_odom.th;

    image = cv::imread(image_path);

    if (image.empty())
    {
      ROS_WARN("Cannot find image: %s", image_path.c_str());
      continue;
    }

    // generate mask
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    double boundary = 20.0;
    int frame_width = image.cols;
    int frame_height = image.rows;

    double x = frame_width / 2 - (vehicle_width / 2 / pixel2meter) - boundary;
    double y = frame_height / 2 - (vehicle_length / 2 / pixel2meter) - boundary;
    double width = vehicle_width / pixel2meter + 2 * boundary;
    double height = vehicle_length / pixel2meter + 2 * boundary;

    cv::rectangle(mask, cv::Rect(x, y, width, height), cv::Scalar(0), -1);

    ros::Time t;
    t = t.fromNSec(timestamp);

    frame_seq.push_back(frame(t, image, mask));
    odom_seq.push_back(vehicle_odom);
  }

  data_file.close();

  ROS_INFO("Load data Finished! Total frames: %lu", frame_seq.size());

  return !frame_seq.empty();
}

void getEstimateTrans(const cv::Mat& homography, trans& estimate_trans)
{
  // get homography from test frame to ref frame
  cv::Mat inv_h = homography.inv();

  // get delta feature odom
  int frame_width = ref_frame_ptr->frame_image.cols;
  int frame_height = ref_frame_ptr->frame_image.rows;

  std::vector<cv::Point2f> base_origin_of_test_in_test,
      base_origin_of_test_in_ref, base_origin_of_ref_in_ref;

  base_origin_of_test_in_test.push_back(cv::Point2f(
      frame_width / 2, frame_height / 2 + rear_axle_to_center / pixel2meter));

  cv::perspectiveTransform(base_origin_of_test_in_test,
                           base_origin_of_test_in_ref, inv_h);
  base_origin_of_ref_in_ref = base_origin_of_test_in_test;

  // calculate delta pose in base_footprint
  estimate_trans.x =
      (base_origin_of_ref_in_ref[0].y - base_origin_of_test_in_ref[0].y) *
      pixel2meter;
  estimate_trans.y =
      (base_origin_of_ref_in_ref[0].x - base_origin_of_test_in_ref[0].x) *
      pixel2meter;
  estimate_trans.th =
      std::atan2(inv_h.at<double>(0, 1), inv_h.at<double>(0, 0));
}

void transformToFootprint(const cv::Point2f& p_src, cv::Point2f& p_dst)
{ // pixel in image -> m in base_footprint

  int frame_width = ref_frame_ptr->frame_image.cols;
  int frame_height = ref_frame_ptr->frame_image.rows;

  p_dst.x = (frame_height / 2 - p_src.y) * pixel2meter + rear_axle_to_center;
  p_dst.y = (frame_width / 2 - p_src.x) * pixel2meter;
}

void findCommonMatchedIndex(std::vector<std::vector<int>>& matched_pts_index,
                            std::vector<int>& common_matched_pts_index)
{
  std::vector<int> cur_ind(matched_pts_index.size(), 0); // save current index

  bool terminate = false;

  for (int ind = 0; ind < ref_frame_ptr->keypoints.size(); ind++)
  { // iterate each keypoint index in ref frame

    int common_cnt = 0;

    for (int j = 0; j < matched_pts_index.size(); j++)
    { // iterate matched index for each test frame

      auto iter = std::find_if(matched_pts_index[j].begin() + cur_ind[j],
                               matched_pts_index[j].end(),
                               std::bind2nd(std::greater_equal<int>(), ind));

      if (iter == matched_pts_index[j].end())
        terminate = true;
      else
        cur_ind[j] = iter - matched_pts_index[j].begin();

      if (*iter == ind)
        common_cnt++;
    }

    if (common_cnt == matched_pts_index.size())
      common_matched_pts_index.push_back(ind);

    if (terminate)
      break;
  }
}

void findOnceMatchedIndex(std::vector<std::vector<int>>& matched_pts_index,
                          std::vector<int>& once_matched_pts_index)
{
  std::set<int> unique_index_set;
  for (auto index : matched_pts_index)
    for (auto i : index)
      unique_index_set.insert(i);

  once_matched_pts_index.assign(unique_index_set.begin(),
                                unique_index_set.end());
}

void saveBaDataToFile(std::string output_file_path)
{
  std::ofstream output_file;
  output_file.open(output_file_path);

  // row 0: poses_num, points_num, observations_num
  output_file << frame_seq.size() << ' ';
  output_file << ref_keypoints.size() << ' ';
  output_file << observation_seq.size() << '\n';

  // row 1~observations_num: pose_id, point_id, obsevation_x, obsevation_y
  for (auto ob : observation_seq)
  {
    output_file << ob.pose_ind << ' ';
    output_file << ob.keypoint_ind << ' ';
    output_file << ob.keypoint.x << ' ';
    output_file << ob.keypoint.y << '\n';
  }

  output_file.close();
}

void saveInitialDataToFile(std::string output_file_path)
{
  std::ofstream output_file;
  output_file.open(output_file_path);

  // row 0: poses_num, points_num
  output_file << estimate_trans_seq.size() << ' ';
  output_file << ref_keypoints.size() << '\n';

  // row 1~poses_num: pose_x, pose_y, pose_th
  for (auto t : estimate_trans_seq)
  {
    output_file << t.x << ' ';
    output_file << t.y << ' ';
    output_file << t.th << '\n';
  }

  // row following: point_x, point_y
  for (auto p : ref_keypoints)
  {
    output_file << p.x << ' ';
    output_file << p.y << '\n';
  }

  output_file.close();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "prepare_test_data_for_ba");
  ros::NodeHandle nh;

  // load data from file
  if (argc == 2)
  {
    if (!loadDataFromFile(argv[1]))
    {
      ROS_ERROR("Failed to load data!");
      return 1;
    }
  }
  else
  {
    ROS_ERROR("Give a data file path as a argument!");
    return 1;
  }

  // process data sequence
  // -- set ref frame
  ref_frame_ptr = std::make_shared<frame>(frame_seq[0]);
  ref_frame_ptr->detectKeypoints();
  ref_frame_ptr->computeFeature();

  for (auto kp : ref_frame_ptr->keypoints)
  {
    cv::Point2f p;
    transformToFootprint(kp.pt, p);
    ref_keypoints.push_back(p);
  }

  ROS_INFO("Total ref keypoints: %lu", ref_frame_ptr->keypoints.size());

  nav_msgs::Odometry ref_odom_msg;
  ref_odom_msg.pose.pose.position.x = odom_seq[0].x;
  ref_odom_msg.pose.pose.position.y = odom_seq[0].y;
  ref_odom_msg.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(odom_seq[0].th);

  frame_seq.erase(frame_seq.begin()); // remove ref frame

  // -- get matched feature points between ref frame and every test frames
  std::vector<std::vector<int>>
      matched_pts_index; // matched keypoints' index in ref frame
  for (int i = 0; i < frame_seq.size(); i++)
  {
    frame_seq[i].detectKeypoints();
    frame_seq[i].computeFeature();

    nav_msgs::Odometry test_odom_msg;
    test_odom_msg.pose.pose.position.x = odom_seq[i].x;
    test_odom_msg.pose.pose.position.y = odom_seq[i].y;
    test_odom_msg.pose.pose.orientation =
        tf::createQuaternionMsgFromYaw(odom_seq[i].th);

    orbFeatureMatcher matcher(ref_frame_ptr,
                              std::make_shared<frame>(frame_seq[i]));
    matcher.calcDirectMatches(ref_odom_msg, test_odom_msg);

    cv::Mat homography = matcher.getHomography();
    ROS_INFO_STREAM("homography:\n" << homography);

    // get estimated trans with homography
    trans estimate_trans;
    getEstimateTrans(homography, estimate_trans);
    estimate_trans_seq.push_back(estimate_trans);

    matcher.visualizeMatches();

    std::vector<cv::DMatch> matches = matcher.getMatches();

    std::vector<int> ind_vec;
    for (auto iter = matches.begin(); iter != matches.end(); iter++)
    {
      if (iter != matches.begin() && (*(iter - 1)).queryIdx == (*iter).queryIdx)
        continue; // ignore duplicated index in ref frame

      ind_vec.push_back((*iter).queryIdx); // keypoint index in ref frame

      observation ob_tmp;
      ob_tmp.pose_ind = i;
      ob_tmp.keypoint_ind = (*iter).queryIdx;
      transformToFootprint(frame_seq[i].keypoints[(*iter).trainIdx].pt,
                           ob_tmp.keypoint);

      observation_seq.push_back(ob_tmp);
    }

    std::sort(ind_vec.begin(), ind_vec.end(), std::less<int>());

    matched_pts_index.push_back(ind_vec);

    cv::waitKey(1);
  }

  // -- get common matched feature points in all frames (need to be optimized
  // along with poses of each test frame by BA)
  std::vector<int> common_matched_pts_index;

  findCommonMatchedIndex(matched_pts_index, common_matched_pts_index);

  ROS_INFO("common_matched_pts_index num: %lu",
           common_matched_pts_index.size());

  // -- get once matched feature points in all frames (need to be optimized
  // along with poses of each test frame by BA)
  std::vector<int> once_matched_pts_index;

  findOnceMatchedIndex(matched_pts_index, once_matched_pts_index);

  once_matched_pts_num = once_matched_pts_index.size();

  ROS_INFO("once_matched_pts_index num: %lu", once_matched_pts_index.size());

  // save data for BA to file
  std::string output_file_path = argv[1];
  auto pos = output_file_path.find_last_of(".");
  if (pos != output_file_path.npos)
  {
    output_file_path.insert(pos, "_output");
    ROS_INFO_STREAM("output file: " << output_file_path);

    saveBaDataToFile(output_file_path);
  }
  else
    ROS_WARN("Failed to save to ba data file!");

  // save initial data for BA to file
  std::string initial_file_path = argv[1];
  pos = initial_file_path.find_last_of(".");
  if (pos != initial_file_path.npos)
  {
    initial_file_path.insert(pos, "_initial");
    ROS_INFO_STREAM("initial file: " << initial_file_path);

    saveInitialDataToFile(initial_file_path);
  }
  else
    ROS_WARN("Failed to save to initial data file!");

  return 0;
}
