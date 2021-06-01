#include <fstream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/tf.h>

#include <opencv2/highgui/highgui.hpp>

int cnt = 0;

std::string folder_path_mask;
std::string folder_path_fish;

cv::Mat src_img, mask_img, output_img;
cv::Mat fisheye_img;

ros::Time last_save_time;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    src_img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

void maskCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    mask_img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

    if (!src_img.empty())
    {
      if ((ros::Time::now() - last_save_time).toSec() < 0.2)
        return;

      cv::Mat merged_img;
      cv::addWeighted(src_img, 0.8, mask_img, 0.2, 0.2, merged_img);

      std::string src_img_name = std::to_string(cnt) + "_src.jpg";
      std::string mask_img_name = std::to_string(cnt) + "_mask.jpg";
      std::string merged_img_name = std::to_string(cnt) + "_merged.jpg";

      cv::imwrite(folder_path_mask + src_img_name, src_img);
      cv::imwrite(folder_path_mask + mask_img_name, mask_img);
      cv::imwrite(folder_path_mask + merged_img_name, merged_img);

      cnt++;

      last_save_time = ros::Time::now();
    }
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

void fisheyeCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    if ((ros::Time::now() - last_save_time).toSec() < 0.2)
      return;

    fisheye_img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

    std::string fisheye_img_name = std::to_string(cnt) + "_src.jpg";
    cv::imwrite(folder_path_fish + fisheye_img_name, fisheye_img);

    cnt++;

    last_save_time = ros::Time::now();
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "prepare_images_for_labling");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // Subscriber to image
  image_transport::Subscriber sub = it.subscribe(
      "/camera/image/birdview_seg", 1, imageCallback, ros::VoidPtr(),
      image_transport::TransportHints("compressed"));

  image_transport::Subscriber mask_sub = it.subscribe(
      "/freespace/freespace_image", 1, maskCallback, ros::VoidPtr(),
      image_transport::TransportHints("compressed"));

  image_transport::Subscriber fisheye_sub =
      it.subscribe("/camera/0/0/image", 1, fisheyeCallback, ros::VoidPtr(),
                   image_transport::TransportHints("compressed"));

  folder_path_mask = "/home/yujr/catkin_ws/src/feature_match_odom-master/res/save/mask/";
  folder_path_fish = "/home/yujr/catkin_ws/src/feature_match_odom-master/res/save/fish/";
  /* if (argc == 2)
  {
    folder_path = std::string(argv[1]) + "/";
  }
  else
  {
    ROS_ERROR("Need a folder path!");
    return 1;
  }
 */
  last_save_time = ros::Time::now();

  ros::spin();

  return 0;
}
