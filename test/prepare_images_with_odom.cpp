#include <fstream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <vehicle_msgs/VehicleInfoStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>

#include <opencv2/highgui/highgui.hpp>

static const double wheelbase = 2.80;
static const double speed_thresh = 10.0;
static const std::string file_path =
    "/home/ros/saic_ws/src/saic_odom/feature_match_odom/res/"
    "images_with_odom2/";

int cnt = 0;

double odom_x = 0.0;
double odom_y = 0.0;
double odom_th = 0.0;

ros::Time last_time;
bool init = true;

cv::Mat mask_img;

std::fstream data_file;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if (!init)
    return;

  try
  {
    // Get the msg image
    cv::Mat img;
    img = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    cv::imshow("image", img);
    cv::waitKey(1);

    std::string img_name = file_path + "img" + std::to_string(cnt) + ".jpg";
    cv::imwrite(img_name, img);

    std::string mask_name =
        file_path + "mask/img" + std::to_string(cnt) + ".jpg";
    cv::imwrite(mask_name, mask_img);

    // record to txt file
    data_file << msg->header.stamp.toNSec() << ' ' << img_name << ' '
              << mask_name << ' ' << odom_x << ' ' << odom_y << ' ' << odom_th
              << '\n';
    cnt++;
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
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

void vehicleInfoCallback(const vehicle_msgs::VehicleInfoStampedConstPtr& msg)
{
  if (!init)
  {
    last_time = msg->header.stamp;
    init = true;
    return;
  }

  // calculate odom
  double speed = msg->speed;
  double steering_angle =
      msg->steeringAngle * 0.91; // correct steering angle by a factor

  if (std::fabs(speed) > speed_thresh)
    return;

  ros::Time current_time = msg->header.stamp;
  double dt = (current_time - last_time).toSec();
  last_time = current_time;

  // calculate ackermann odom
  double vx = 0.0, vy = 0.0, vth = 0.0;

  if (steering_angle == 0.0)
    vth = 0.0;
  else
  {
    double turning_radius = wheelbase / tan(steering_angle);
    vth = speed / turning_radius;
  }

  vx = speed * cos(odom_th);
  vy = speed * sin(odom_th);

  odom_x += vx * dt;
  odom_y += vy * dt;
  odom_th += vth * dt;
}

void odomCallback(const nav_msgs::OdometryConstPtr& msg)
{
  odom_x = msg->pose.pose.position.x;
  odom_y = msg->pose.pose.position.y;
  odom_th = tf::getYaw(msg->pose.pose.orientation);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "prepare_images_with_odom");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // Subscriber to image
  image_transport::Subscriber sub = it.subscribe(
      "/camera/image/birdview_seg", 1, imageCallback, ros::VoidPtr(),
      image_transport::TransportHints("compressed"));

  image_transport::Subscriber mask_sub = it.subscribe(
      "/freespace/freespace_image", 1, maskCallback, ros::VoidPtr(),
      image_transport::TransportHints("compressed"));

  // Subsciber to vehicle_info
  ros::Subscriber vehicle_info_sub =
      nh.subscribe("/deepps/vehicle_info", 100, vehicleInfoCallback);

  ros::Subscriber odom_sub = nh.subscribe("odom", 100, odomCallback);

  std::string data_file_path = file_path + "data.txt";
  ROS_INFO_STREAM("data file path: " << data_file_path);

  data_file.open(data_file_path);

  ros::spin();

  data_file.close();

  return 0;
}
