#include "featureMatchOdom.h"

namespace feature_match_odom
{

featureMatchOdom::featureMatchOdom(ros::NodeHandle nh, ros::NodeHandle pnh)
    : nh_(nh), pnh_(pnh), it_(nh), initialized_(false), frame_updated_(false),
      mask_updated_(false), vehicle_odom_updated_(false)
{
  birdview_img_sub_ = it_.subscribe(
      "/camera/image/birdview_seg", 1, &featureMatchOdom::updateFrame, this,
      image_transport::TransportHints("compressed"));
  birdview_mask_sub_ = it_.subscribe(
      "/freespace/freespace_image", 1, &featureMatchOdom::updateMask, this,
      image_transport::TransportHints("compressed"));
  matched_img_pub_ = it_.advertise("/feature_match_odom/matched_image", 1);
  result_img_pub_ = it_.advertise("/feature_match_odom/result_image", 1);

  vehicle_odom_sub_ = nh_.subscribe("ackermann_odom", 100,
                                    &featureMatchOdom::updateVehicleOdom, this);

  feature_match_odom_pub_ =
      nh.advertise<nav_msgs::Odometry>("odom_feature", 100);

  total_feature_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/total_feature_cloud", 10);
  test_frame_feature_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/test_feature_cloud", 30);
  ref_frame_feature_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/ref_feature_cloud", 30);

  process_timer_ = nh_.createTimer(ros::Duration(0.01),
                                   &featureMatchOdom::processFrame, this);

  // initialize odom_feature msg
  feature_match_odom_msg_.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(0.0);
  current_vehicle_odom_msg_.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(0.0);
  ref_vehicle_odom_msg_.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(0.0);
  test_vehicle_odom_msg_.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(0.0);

  total_frame_num_ = 0.0;
  valid_frame_num_ = 0.0;
}

featureMatchOdom::~featureMatchOdom()
{
  if (thread_local_ba_.joinable())
    thread_local_ba_.join();
}

void featureMatchOdom::startLocalBa()
{
  thread_local_ba_ = std::thread([this]()
                                 {
                                   this->checkUpdateRefFrame();
                                 });
}

void featureMatchOdom::updateFrame(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    // Get the msg image
    current_frame_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    current_frame_time_ = ros::Time::now();
    test_vehicle_odom_msg_ = current_vehicle_odom_msg_;
    frame_updated_ = true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

void featureMatchOdom::updateMask(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    // Get the msg image
    current_mask_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    current_mask_time_ = msg->header.stamp;
    mask_updated_ = true;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

void featureMatchOdom::updateVehicleOdom(const nav_msgs::OdometryConstPtr& msg)
{
  current_vehicle_odom_msg_.header = msg->header;
  current_vehicle_odom_msg_.pose = msg->pose;
  current_vehicle_odom_msg_.twist = msg->twist;
  vehicle_odom_updated_ = true;
}

void featureMatchOdom::processFrame(const ros::TimerEvent& event)
{

  if (frame_updated_ && vehicle_odom_updated_)
  {
    total_frame_num_++;

    ROS_INFO("Processing a new frame");

    cv::Mat mask_image;

    bool reserve_mask = true;

    if (reserve_mask || mask_updated_)
      mask_image = current_mask_;
    else
    {
      // ignore footprint
      mask_image = cv::Mat(current_frame_.rows, current_frame_.cols, CV_8UC1,
                           cv::Scalar(255));

      double boundary = 15.0;
      int frame_width = current_frame_.cols;
      int frame_height = current_frame_.rows;

      double x = frame_width / 2 - (vehicle_width / 2 / pixel2meter) - boundary;
      double y =
          frame_height / 2 - (vehicle_length / 2 / pixel2meter) - boundary;
      double width = vehicle_width / pixel2meter + 2 * boundary;
      double height = vehicle_length / pixel2meter + 2 * boundary;

      cv::rectangle(mask_image, cv::Rect(x, y, width, height), cv::Scalar(0),
                    -1);
    }
    ROS_INFO("Watiting to lock ba");

    std::lock_guard<std::mutex> guard(mtx_local_ba_);

    ROS_INFO("Finish to lock ba");

    test_frame_ptr_ = std::make_shared<frame>(
        frame(current_frame_time_, current_frame_, mask_image));

    if (!initialized_)
    {
      test_frame_ptr_->detectKeypoints();
      test_frame_ptr_->computeFeature();

      ref_frame_ptr_ = std::make_shared<frame>(test_frame_ptr_->clone());
      ref_vehicle_odom_msg_ = test_vehicle_odom_msg_;
      initialized_ = true;
    }
    else
    {
      test_frame_ptr_->detectKeypoints();
      test_frame_ptr_->computeFeature();

      test_frame_ptr_->visualizeFeature();

      matcher_ptr_ = std::make_shared<orbFeatureMatcher>(
          orbFeatureMatcher(ref_frame_ptr_, test_frame_ptr_));

      // calculate knn-matches
      //      matcher_ptr_->calcKnnMatches();

      // calculate direct matches
      matcher_ptr_->calcDirectMatches(ref_vehicle_odom_msg_,
                                      test_vehicle_odom_msg_);

      // calculate homography
      homography_ = matcher_ptr_->getHomography();

      // save matches to sequence
      matches_seq_.push_back(matcher_ptr_->getMatches());

      cv::Mat matched_img = matcher_ptr_->visualizeMatches();

      sensor_msgs::ImagePtr msg_matched_img =
          cv_bridge::CvImage(std_msgs::Header(), "bgr8", matched_img)
              .toImageMsg();
      matched_img_pub_.publish(msg_matched_img);

      // publish ref feature points as ros msg
      double x = ref_frame_ptr_->odom_x;
      double y = ref_frame_ptr_->odom_y;
      double th = ref_frame_ptr_->odom_th;

      tf::Transform tf_base_ref2odom_feature;
      tf_base_ref2odom_feature.setOrigin(tf::Vector3(x, y, 0.0));
      tf_base_ref2odom_feature.setRotation(tf::createQuaternionFromYaw(th));

      std::vector<cv::DMatch> feature_matches = matcher_ptr_->getMatches();
      std::vector<tf::Point> matched_ref_feature_pts;
      for (auto m : feature_matches)
      {
        cv::Point2f p_in_footprint;
        transformToFootprint(ref_frame_ptr_->keypoints[m.queryIdx].pt,
                             p_in_footprint);

        tf::Point p_tmp;
        p_tmp.setX(p_in_footprint.x);
        p_tmp.setY(p_in_footprint.y);

        matched_ref_feature_pts.push_back(tf_base_ref2odom_feature * p_tmp);
      }

      sensor_msgs::PointCloud2 ref_cloud_msg;
      featurePointToCloud(matched_ref_feature_pts, ref_cloud_msg, false, 100.0);
      ref_frame_feature_cloud_pub_.publish(ref_cloud_msg);

      // check result
      bool valid_pose = false;
      if (!homography_.empty())
      {
        valid_pose = checkHomographyVehicleFootprint();

        // calculate relative pose of test frame in ref frame
        cv::Point2f base_of_test_in_ref, base_of_ref_in_ref;
        calcResultRefFrame(base_of_ref_in_ref, base_of_test_in_ref);

        // calculate odom msg
        calcAndPublishOdom(base_of_ref_in_ref, base_of_test_in_ref);

        // add test frame to sequence for ref frame optimization
        frame_seq_.push_back(test_frame_ptr_->clone());
      }
      else
      {
        ROS_WARN("processFrame: homography is invalid!");
      }

      // publish test feature points as ros topic
      std::vector<tf::Point> matched_test_feature_pts;
      for (auto m : feature_matches)
      {
        cv::Point2f p_in_footprint;
        transformToFootprint(test_frame_ptr_->keypoints[m.trainIdx].pt,
                             p_in_footprint);

        tf::Point p_tmp;
        p_tmp.setX(p_in_footprint.x);
        p_tmp.setY(p_in_footprint.y);

        matched_test_feature_pts.push_back(p_tmp);
      }

      sensor_msgs::PointCloud2 test_cloud_msg;
      double intensity = valid_pose ? 100.0 : 0.0;
      featurePointToCloud(matched_test_feature_pts, test_cloud_msg, true,
                          intensity);
      test_frame_feature_cloud_pub_.publish(test_cloud_msg);

      ROS_INFO("valid frame num: %d/%d", valid_frame_num_, total_frame_num_);
    }

    frame_updated_ = false;
    mask_updated_ = false;
    vehicle_odom_updated_ = false;
  }
}

// draw result on ref frame
void featureMatchOdom::calcResultRefFrame(cv::Point2f& base_of_ref_in_ref,
                                          cv::Point2f& base_of_test_in_ref)
{
  int frame_width = current_frame_.cols;
  int frame_height = current_frame_.rows;

  // generate footprint coordinates to the center of frame
  cv::Point2f p1, p2, p3, p4;
  p1.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p1.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p2.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p2.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p3.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p3.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  p4.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p4.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  // get homography from test frame to ref frame
  cv::Mat inv_h = homography_.inv();

  // draw footprint to ref frame
  std::vector<cv::Point2f> footprint_test_in_ref, footprint_test_in_test,
      footprint_ref_in_ref;
  footprint_test_in_test.push_back(p1);
  footprint_test_in_test.push_back(p2);
  footprint_test_in_test.push_back(p3);
  footprint_test_in_test.push_back(p4);

  cv::perspectiveTransform(footprint_test_in_test, footprint_test_in_ref,
                           inv_h);
  footprint_ref_in_ref = footprint_test_in_test;

  cv::Mat result_img_ref = ref_frame_ptr_->frame_image.clone();
  drawFootprint(result_img_ref, footprint_ref_in_ref, cv::Scalar(255, 255, 0));
  drawFootprint(result_img_ref, footprint_test_in_ref, cv::Scalar(0, 255, 0));

  // draw origin of base_footprint to ref frame
  std::vector<cv::Point2f> base_origin_of_test_in_test,
      base_origin_of_test_in_ref, base_origin_of_ref_in_ref;

  base_origin_of_test_in_test.push_back(cv::Point2f(
      frame_width / 2, frame_height / 2 + rear_axle_to_center / pixel2meter));

  cv::perspectiveTransform(base_origin_of_test_in_test,
                           base_origin_of_test_in_ref, inv_h);
  base_origin_of_ref_in_ref = base_origin_of_test_in_test;

  base_of_ref_in_ref = base_origin_of_ref_in_ref[0];
  base_of_test_in_ref = base_origin_of_test_in_ref[0];

  cv::circle(result_img_ref,
             cv::Point(base_of_ref_in_ref.x, base_of_ref_in_ref.y), 2,
             cv::Scalar(255, 255, 0), -1);
  cv::circle(result_img_ref,
             cv::Point(base_of_test_in_ref.x, base_of_test_in_ref.y), 2,
             cv::Scalar(0, 255, 0), -1);

  cv::imshow("result in ref frame", result_img_ref);
  cv::waitKey(1);

  sensor_msgs::ImagePtr msg_result_img =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", result_img_ref)
          .toImageMsg();
  result_img_pub_.publish(msg_result_img);
}

void featureMatchOdom::calcAndPublishOdom(
    cv::Point2f base_origin_of_ref_in_ref,
    cv::Point2f base_origin_of_test_in_ref)
{
  // get homography from test frame to ref frame
  cv::Mat inv_h = homography_.inv();

  // calculate delta pose in base_footprint
  double dx_base =
      (base_origin_of_ref_in_ref.y - base_origin_of_test_in_ref.y) *
      pixel2meter;
  double dy_base =
      (base_origin_of_ref_in_ref.x - base_origin_of_test_in_ref.x) *
      pixel2meter;
  double dth_base = atan2(inv_h.at<double>(0, 1), inv_h.at<double>(0, 0));

  // check homography validity with vehicle footprint constraints
  bool valid_footprint = checkHomographyVehicleFootprint();

  if (valid_footprint)
    valid_frame_num_++;

  // check homography validity, replace with odom's if bad one occurs
  checkWithVehicleOdom(dx_base, dy_base, dth_base, !valid_footprint);

  // calculate tf: base_footprint_test -> base_footprint_ref
  tf::Transform tf_base_test2base_ref;
  tf_base_test2base_ref.setOrigin(tf::Vector3(dx_base, dy_base, 0.0));
  tf_base_test2base_ref.setRotation(tf::createQuaternionFromYaw(dth_base));

  // calculate tf: base_footprint_ref -> odom_feature
  double x = ref_frame_ptr_->odom_x;
  double y = ref_frame_ptr_->odom_y;
  double th = ref_frame_ptr_->odom_th;

  tf::Transform tf_base_ref2odom_feature;
  tf_base_ref2odom_feature.setOrigin(tf::Vector3(x, y, 0.0));
  tf_base_ref2odom_feature.setRotation(tf::createQuaternionFromYaw(th));

  // calculate tf: base_footprint_test -> odom_feature
  tf::Transform tf_base_test2odom_feature =
      tf_base_ref2odom_feature * tf_base_test2base_ref;

  // update odom_feature msg
  feature_match_odom_msg_.header.stamp = ros::Time::now();
  feature_match_odom_msg_.header.frame_id = "odom_feature";

  tf::Vector3 origin_test = tf_base_test2odom_feature.getOrigin();
  feature_match_odom_msg_.pose.pose.position.x = origin_test.x();
  feature_match_odom_msg_.pose.pose.position.y = origin_test.y();

  double th_test = tf::getYaw(tf_base_test2odom_feature.getRotation());
  feature_match_odom_msg_.pose.pose.orientation =
      tf::createQuaternionMsgFromYaw(th_test);

  // publish the odom_feature msg
  feature_match_odom_pub_.publish(feature_match_odom_msg_);

  // update odom to frame
  test_frame_ptr_->odom_x = origin_test.x();
  test_frame_ptr_->odom_y = origin_test.y();
  test_frame_ptr_->odom_th = th_test;

  ROS_DEBUG("==========test frame odom: (%.3f, %.3f, %.3f)",
            test_frame_ptr_->odom_x, test_frame_ptr_->odom_y,
            test_frame_ptr_->odom_th);

  // publish tf: base_footprint -> odom_feature
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = ros::Time::now();
  odom_trans.header.frame_id = "odom_feature";
  odom_trans.child_frame_id = "base_footprint_feature";

  odom_trans.transform.translation.x = origin_test.x();
  odom_trans.transform.translation.y = origin_test.y();
  odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(th_test);

  // send the transform
  odom_feature_broadcaster_.sendTransform(odom_trans);
}

// check to update ref frame
void featureMatchOdom::checkUpdateRefFrame()
{
  while (ros::ok())
  {
    ROS_INFO_THROTTLE(1.0,
                      "***********checkUpdateRefFrame in feature match odom");
    // prepare data containers
    bool update_ref_frame = false;
    std::string condition_str;
    std::shared_ptr<frame> cached_current_frame, cached_previous_ref_frame;

    // thresholds
    double vehicle_pos_thresh = 2.0 /*1.5*/;         // m
    double vehicle_ori_thresh = 0.8 /*0.5*/;         // rad
    double feature_pos_thresh = 1.5 /*1.2*/ /*1.0*/; // m
    double feature_ori_thresh = 0.6 /*0.6*/ /*0.5*/; // rad
    double time_thresh = 3.0 /*3.0*/;                // s
    double min_time_thresh = 0.5;                    // s

    // calculate spatial and temporal variation
    {
      std::lock_guard<std::mutex> guard(mtx_local_ba_);

      // check initialized
      if (!initialized_)
      {
        ROS_INFO_THROTTLE(1.0, "Local BA: waiting for initialization...");
        continue;
      }

      // -- 1. vehicle
      double delta_vehicle_pos_x = test_vehicle_odom_msg_.pose.pose.position.x -
                                   ref_vehicle_odom_msg_.pose.pose.position.x;
      double delta_vehicle_pos_y = test_vehicle_odom_msg_.pose.pose.position.y -
                                   ref_vehicle_odom_msg_.pose.pose.position.y;
      double delta_vehicle_pos =
          std::hypot(delta_vehicle_pos_x, delta_vehicle_pos_y);

      double delta_vehicle_ori =
          std::fabs(tf::getYaw(test_vehicle_odom_msg_.pose.pose.orientation) -
                    tf::getYaw(ref_vehicle_odom_msg_.pose.pose.orientation));

      // -- 2. feature
      double delta_feature_pos =
          std::hypot(test_frame_ptr_->odom_x - ref_frame_ptr_->odom_x,
                     test_frame_ptr_->odom_y - ref_frame_ptr_->odom_y);
      double delta_feature_ori =
          std::fabs(test_frame_ptr_->odom_th - ref_frame_ptr_->odom_th);

      // -- 3. time
      double delta_time =
          (test_frame_ptr_->timestamp - ref_frame_ptr_->timestamp).toSec();

      // check condition
      if (delta_vehicle_pos > vehicle_pos_thresh ||
          delta_vehicle_ori > vehicle_ori_thresh)
      {
        update_ref_frame = true;
        condition_str = "vehicle movement!";
      }
      else if (delta_feature_pos > feature_pos_thresh ||
               delta_feature_ori > feature_ori_thresh)
      {
        update_ref_frame = true;
        condition_str = "feature movement!";
      }
      else if (delta_time > time_thresh)
      {
        update_ref_frame = true;
        condition_str = "time out!";
      }

      if (delta_time < min_time_thresh)
        update_ref_frame = false;
    }

    // update ref frame
    if (update_ref_frame)
    {
      ROS_INFO("Start to update ref frame");

      // prepare data for optimizer
      std::vector<Pose2d> poses;
      std::vector<Eigen::Vector2d> points;
      std::vector<std::vector<Observation>> observations;

      {
        std::lock_guard<std::mutex> guard(mtx_local_ba_);

        // cache current frame
        cached_current_frame =
            std::make_shared<frame>(test_frame_ptr_->clone());

        prepareDataForRefOptimizer(poses, points, observations);

        // clear frame sequence
        frame_seq_.clear();
        matches_seq_.clear();

        // cache previous ref frame
        cached_previous_ref_frame = ref_frame_ptr_;

        // update ref frame
        ref_frame_ptr_ = cached_current_frame;
        ref_vehicle_odom_msg_ = test_vehicle_odom_msg_;
      }

      // construct a optimizer
      optimization op(poses, points, observations);

      // solve problem
      op.solveProblem();

      // optimize poses in frame_seq and keypoints in last ref frame
      Pose2d pose = poses.back();

      // calculate feature odom of ref frame
      // calculate tf: current ref -> previous ref
      tf::Transform tf_cur_ref2pre_ref;
      tf_cur_ref2pre_ref.setOrigin(tf::Vector3(pose.x, pose.y, 0.0));
      tf_cur_ref2pre_ref.setRotation(
          tf::createQuaternionFromYaw(pose.yaw_radians));

      std::lock_guard<std::mutex> guard(mtx_local_ba_);

      // calculate tf: pre_ref -> odom_feature
      double x = cached_previous_ref_frame->odom_x;
      double y = cached_previous_ref_frame->odom_y;
      double th = cached_previous_ref_frame->odom_th;

      tf::Transform tf_prev_ref2odom_feature;
      tf_prev_ref2odom_feature.setOrigin(tf::Vector3(x, y, 0.0));
      tf_prev_ref2odom_feature.setRotation(tf::createQuaternionFromYaw(th));

      // calculate tf: cur_ref -> odom_feature
      tf::Transform tf_cur_ref2odom_feature =
          tf_prev_ref2odom_feature * tf_cur_ref2pre_ref;

      // update odom of ref frame
      ref_frame_ptr_->odom_x = tf_cur_ref2odom_feature.getOrigin().x();
      ref_frame_ptr_->odom_y = tf_cur_ref2odom_feature.getOrigin().y();
      ref_frame_ptr_->odom_th =
          tf::getYaw(tf_cur_ref2odom_feature.getRotation());

      ROS_INFO("Update ref frame: %s", condition_str.c_str());
      ROS_INFO("===================== current odom: (%.3f, %.3f, %.3f)",
               ref_frame_ptr_->odom_x, ref_frame_ptr_->odom_y,
               ref_frame_ptr_->odom_th);

      // publish all ref feature points and total feature points
      for (auto kp : ref_frame_ptr_->keypoints)
      {
        cv::Point2f p_in_footprint;
        transformToFootprint(kp.pt, p_in_footprint);

        tf::Point p_tmp;
        p_tmp.setX(p_in_footprint.x);
        p_tmp.setY(p_in_footprint.y);

        total_feature_points_.push_back(tf_cur_ref2odom_feature * p_tmp);
      }

      sensor_msgs::PointCloud2 total_cloud_msg;
      featurePointToCloud(total_feature_points_, total_cloud_msg, false, 100.0);
      total_feature_cloud_pub_.publish(total_cloud_msg);
    }
    else
      ros::Duration(0.1).sleep();
  }
}

void featureMatchOdom::prepareDataForRefOptimizer(
    std::vector<Pose2d>& poses, std::vector<Eigen::Vector2d>& points,
    std::vector<std::vector<Observation>>& observations)
{
  std::vector<std::vector<int>>
      matched_pts_index; // matched keypoints' index in ref frame
  for (int i = 0; i < frame_seq_.size(); i++)
  {
    // get pose of each frame (test -> ref)
    // calculate tf: ref -> odom_feature
    tf::Transform tf_ref2odom_feature;
    tf_ref2odom_feature.setOrigin(
        tf::Vector3(ref_frame_ptr_->odom_x, ref_frame_ptr_->odom_y, 0.0));
    tf_ref2odom_feature.setRotation(
        tf::createQuaternionFromYaw(ref_frame_ptr_->odom_th));

    // calculate tf: test -> odom_feature
    tf::Transform tf_test2odom_feature;
    tf_test2odom_feature.setOrigin(
        tf::Vector3(frame_seq_[i].odom_x, frame_seq_[i].odom_y, 0.0));
    tf_test2odom_feature.setRotation(
        tf::createQuaternionFromYaw(frame_seq_[i].odom_th));

    // calculate tf: test -> ref
    tf::Transform tf_test2ref =
        tf_ref2odom_feature.inverse() * tf_test2odom_feature;

    Pose2d pose;
    pose.x = tf_test2ref.getOrigin().x();
    pose.y = tf_test2ref.getOrigin().y();
    pose.yaw_radians = tf::getYaw(tf_test2ref.getRotation());
    poses.push_back(pose);

    std::vector<Observation> ob_vec;
    std::vector<int> ind_vec;
    for (auto iter = matches_seq_[i].begin(); iter != matches_seq_[i].end();
         iter++)
    {
      if ((*(iter - 1)).queryIdx == (*iter).queryIdx)
        continue; // ignore duplicated index in ref frame

      ind_vec.push_back((*iter).queryIdx); // keypoint index in ref frame

      Observation ob_tmp;
      ob_tmp.pose_index = i;
      ob_tmp.point_index = (*iter).queryIdx;

      cv::Point2f p_in_footprint;
      transformToFootprint(frame_seq_[i].keypoints[(*iter).trainIdx].pt,
                           p_in_footprint);
      ob_tmp.point[0] = p_in_footprint.x;
      ob_tmp.point[1] = p_in_footprint.y;

      ob_vec.push_back(ob_tmp);
    }

    observations.push_back(ob_vec);
    matched_pts_index.push_back(ind_vec);
  }

  // get all points in ref frame
  for (auto kp : ref_frame_ptr_->keypoints)
  {
    cv::Point2f tmp_p;
    transformToFootprint(kp.pt, tmp_p);

    Eigen::Vector2d p;
    p[0] = tmp_p.x;
    p[1] = tmp_p.y;
    points.push_back(p);
  }
}

// transform a point in birdview image to base_footprint
void featureMatchOdom::transformToFootprint(const cv::Point2f& p_src,
                                            cv::Point2f& p_dst)
{
  int frame_width = ref_frame_ptr_->frame_image.cols;
  int frame_height = ref_frame_ptr_->frame_image.rows;

  p_dst.x = (frame_height / 2 - p_src.y) * pixel2meter + rear_axle_to_center;
  p_dst.y = (frame_width / 2 - p_src.x) * pixel2meter;
}

// check homography validity by vehicle footprint shape constraints
bool featureMatchOdom::checkHomographyVehicleFootprint()
{
  int frame_width = current_frame_.cols;
  int frame_height = current_frame_.rows;

  // generate footprint coordinates to the center of frame
  cv::Point2f p1, p2, p3, p4;
  p1.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p1.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p2.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p2.y = frame_height / 2 - (vehicle_length / 2 / pixel2meter);

  p3.x = frame_width / 2 - (vehicle_width / 2 / pixel2meter);
  p3.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  p4.x = frame_width / 2 + (vehicle_width / 2 / pixel2meter);
  p4.y = frame_height / 2 + (vehicle_length / 2 / pixel2meter);

  // get homography from test frame to ref frame
  cv::Mat inv_h = homography_.inv();

  // get footprint in ref frame
  std::vector<cv::Point2f> footprint_test_in_ref, footprint_test_in_test;
  footprint_test_in_test.push_back(p1);
  footprint_test_in_test.push_back(p2);
  footprint_test_in_test.push_back(p3);
  footprint_test_in_test.push_back(p4);

  cv::perspectiveTransform(footprint_test_in_test, footprint_test_in_ref,
                           inv_h);

  cv::Point2f p1_t = footprint_test_in_ref[0];
  cv::Point2f p2_t = footprint_test_in_ref[1];
  cv::Point2f p3_t = footprint_test_in_ref[2];
  cv::Point2f p4_t = footprint_test_in_ref[3];

  // check vehicle width and length
  double p1_p2_dist_pixel = std::hypot(p1_t.x - p2_t.x, p1_t.y - p2_t.y);
  double p2_p3_dist_pixel = std::hypot(p2_t.x - p3_t.x, p2_t.y - p3_t.y);
  double p3_p4_dist_pixel = std::hypot(p3_t.x - p4_t.x, p3_t.y - p4_t.y);
  double p4_p1_dist_pixel = std::hypot(p4_t.x - p1_t.x, p4_t.y - p1_t.y);

  double size_thresh = 0.2; // m

  if (std::abs(p1_p2_dist_pixel * pixel2meter - vehicle_width) > size_thresh ||
      std::abs(p3_p4_dist_pixel * pixel2meter - vehicle_width) > size_thresh ||
      std::abs(p2_p3_dist_pixel * pixel2meter - vehicle_length) > size_thresh ||
      std::abs(p4_p1_dist_pixel * pixel2meter - vehicle_length) > size_thresh)
  {
    ROS_INFO("Invalid homography: vehicle size constraints failed!");
    return false;
  }

  // check angle of adjacent edges of footprint
  double cos_p12_p23 =
      (p1_t - p2_t).dot(p2_t - p3_t) / p1_p2_dist_pixel / p2_p3_dist_pixel;
  double cos_p23_p34 =
      (p2_t - p3_t).dot(p3_t - p4_t) / p2_p3_dist_pixel / p3_p4_dist_pixel;
  double cos_p34_p41 =
      (p3_t - p4_t).dot(p4_t - p1_t) / p3_p4_dist_pixel / p4_p1_dist_pixel;
  double cos_p41_p12 =
      (p4_t - p1_t).dot(p1_t - p2_t) / p4_p1_dist_pixel / p1_p2_dist_pixel;

  double angle_cos_thresh = 0.08; // cos(80 degree)

  if (std::fabs(cos_p12_p23) > angle_cos_thresh ||
      std::fabs(cos_p23_p34) > angle_cos_thresh ||
      std::fabs(cos_p34_p41) > angle_cos_thresh ||
      std::fabs(cos_p41_p12) > angle_cos_thresh)
  {
    ROS_INFO("Invalid homography: angle of edges failed!");
    return false;
  }

  return true;
}

// check relative pose validity by vehicle odom and update it
void featureMatchOdom::checkWithVehicleOdom(double& dx_base, double& dy_base,
                                            double& dth_base,
                                            bool direct_replace)
{
  tf::Transform tf_base_test2odom;
  tf_base_test2odom.setOrigin(
      tf::Vector3(test_vehicle_odom_msg_.pose.pose.position.x,
                  test_vehicle_odom_msg_.pose.pose.position.y, 0.0));
  tf_base_test2odom.setRotation(tf::createQuaternionFromYaw(
      tf::getYaw(test_vehicle_odom_msg_.pose.pose.orientation)));

  tf::Transform tf_base_ref2odom;
  tf_base_ref2odom.setOrigin(
      tf::Vector3(ref_vehicle_odom_msg_.pose.pose.position.x,
                  ref_vehicle_odom_msg_.pose.pose.position.y, 0.0));
  tf_base_ref2odom.setRotation(tf::createQuaternionFromYaw(
      tf::getYaw(ref_vehicle_odom_msg_.pose.pose.orientation)));

  // calculate tf: base_footprint_test -> base_footprint_ref according to
  // vehicle odom
  tf::Transform tf_base_test2base_ref =
      tf_base_ref2odom.inverse() * tf_base_test2odom;

  double pos_thresh = 0.3/*0.2*/;    // m
  double angle_thresh = 0.1/*0.06*/; // rad

  int matched_num_thresh = 80;

  double dx_odom = tf_base_test2base_ref.getOrigin().x();
  double dy_odom = tf_base_test2base_ref.getOrigin().y();
  double dth_odom = tf::getYaw(tf_base_test2base_ref.getRotation());

  ROS_WARN("delta feature: (%f, %f, %f)  delta odom: (%f, %f, %f)", dx_base,
           dy_base, dth_base, dx_odom, dy_odom, dth_odom);

  bool update_with_odom = false;

  if (direct_replace)
  {
    ROS_INFO("update with vehicle odom: direct replace");
    update_with_odom = true;
  }
  else if (!checkEnoughMatchedPoint(matched_num_thresh))
  {
    ROS_ERROR(
        "update with vehicle odom: not enough tracked points (%d)*********",
        matcher_ptr_->getMatchesNum());
    update_with_odom = true;
  }
  else if (std::fabs(dx_base - dx_odom) > pos_thresh ||
           std::fabs(dy_base - dy_odom) > pos_thresh)
  {
    ROS_INFO(
        "update with vehicle odom: too much position movement (%.3f, %.3f)",
        std::fabs(dx_base - dx_odom), std::fabs(dy_base - dy_odom));
    update_with_odom = true;
  }
  else if (std::fabs(dth_base - dth_odom) > angle_thresh)
  {
    ROS_ERROR("update with vehicle odom: too much angle movement (%.3f)",
              std::fabs(dth_base - dth_odom));
    update_with_odom = true;
  }

  if (update_with_odom)
  {
    dx_base = dx_odom;
    dy_base = dy_odom;
    dth_base = dth_odom;
  }
}

// check if matched points is enough with a threshold
bool featureMatchOdom::checkEnoughMatchedPoint(int min_num)
{
  return matcher_ptr_->getMatchesNum() > min_num;
}

// draw footprint
void featureMatchOdom::drawFootprint(cv::Mat dst,
                                     std::vector<cv::Point2f> footprint,
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

void featureMatchOdom::featurePointToCloud(std::vector<tf::Point>& feature_pts,
                                           sensor_msgs::PointCloud2& cloud,
                                           bool is_local, double intensity)
{
  PointCloud cloud_pcl;

  for (auto fp : feature_pts)
  {
    pcl::PointXYZI p;
    p.x = fp.x();
    p.y = fp.y();
    p.z = fp.z();
    p.intensity = intensity;
    cloud_pcl.points.push_back(p);
  }

  cloud_pcl.width = feature_pts.size();
  cloud_pcl.height = 1;

  // transfer to ros msg
  pcl::toROSMsg(cloud_pcl, cloud);

  // update header
  cloud.header.frame_id = is_local ? "base_footprint_feature" : "odom_feature";
  cloud.header.stamp = ros::Time::now();
}
}
