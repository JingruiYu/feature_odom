#include <fstream>
#include <iomanip>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Imu.h>
#include <vehicle_msgs/VehicleInfoStamped.h>

static const double wheelbase = 2.80;
static const double track = 1.616;

static const double speed_thresh = 10.0;
static const double gyro_thresh = 5.0;

double x_ackermann = 0.0, y_ackermann = 0.0, th_ackermann = 0.0;
double vx_ackermann = 0.0, vy_ackermann = 0.0, vth_ackermann = 0.0;
double x_imu = 0.0, y_imu = 0.0, th_imu = 0.0;
double vx_imu = 0.0, vy_imu = 0.0, vth_imu = 0.0;
double gyro_z = 0.0;
ros::Time last_time, current_time;

ros::Publisher ackermann_odom_pub;
ros::Publisher imu_odom_pub;

bool init = false;
bool vel_updated = false;

int ground_truth_ind = 0;
int vehicle_odom_ind = 0;
int feature_odom_ind = 0;

std::ofstream ground_truth_file, vehicle_odom_file, feature_odom_file,
    feature_odom_cam_file;

void velCB(const vehicle_msgs::VehicleInfoStampedConstPtr fdbk_vel_msg)
{
  if (!init)
  {
    last_time = ros::Time::now() /*fdbk_vel_msg->header.stamp*/;
    init = true;
    return;
  }

  double speed = fdbk_vel_msg->speed;
  double speed_left = fdbk_vel_msg->speed_left;
  double speed_right = fdbk_vel_msg->speed_right;
  double steering_angle = fdbk_vel_msg->steeringAngle * 0.91;

  if (std::fabs(speed) > speed_thresh || std::fabs(speed_left) > speed_thresh ||
      std::fabs(speed_right) > speed_thresh)
    return;

  current_time = ros::Time::now() /*fdbk_vel_msg->header.stamp*/;
  double dt = (current_time - last_time).toSec();
  last_time = current_time;

  // calculate ackermann odom
  if (steering_angle == 0.0)
    vth_ackermann = 0.0;
  else
  {
    double turning_radius = wheelbase / tan(steering_angle);
    vth_ackermann = speed / turning_radius;
  }

  vx_ackermann = speed * cos(th_ackermann);
  vy_ackermann = speed * sin(th_ackermann);

  x_ackermann += vx_ackermann * dt;
  y_ackermann += vy_ackermann * dt;
  th_ackermann += vth_ackermann * dt;

  // calculate imu odom
  vx_imu = speed * cos(th_imu);
  vy_imu = speed * sin(th_imu);
  vth_imu = -gyro_z;

  x_imu += vx_imu * dt;
  y_imu += vy_imu * dt;
  th_imu += vth_imu * dt;

  // ************* publish ackermann odom *************
  // since all odometry is 6DOF we'll need a quaternion created from yaw
  geometry_msgs::Quaternion ackermann_odom_quat =
      tf::createQuaternionMsgFromYaw(th_ackermann);

  // next, we'll publish the odometry message over ROS
  nav_msgs::Odometry ackermann_odom;
  ackermann_odom.header.stamp = current_time;
  ackermann_odom.header.frame_id = "odom_feature";
  ackermann_odom.child_frame_id = "base_footprint";

  // set the position
  ackermann_odom.pose.pose.position.x = x_ackermann;
  ackermann_odom.pose.pose.position.y = y_ackermann;
  ackermann_odom.pose.pose.position.z = 0.0;
  ackermann_odom.pose.pose.orientation = ackermann_odom_quat;

  // set the velocity
  ackermann_odom.twist.twist.linear.x = vx_ackermann;
  ackermann_odom.twist.twist.linear.y = vy_ackermann;
  ackermann_odom.twist.twist.angular.z = vth_ackermann;

  // publish the message
  ackermann_odom_pub.publish(ackermann_odom);

  // write to file
  vehicle_odom_file << std::fixed << std::setprecision(9)
                    << current_time.toSec() << " " << x_ackermann << " "
                    << y_ackermann << " " << 0.0 << " " << ackermann_odom_quat.x
                    << " " << ackermann_odom_quat.y << " "
                    << ackermann_odom_quat.z << " " << ackermann_odom_quat.w
                    << "\n";

  // publish tf
  static tf::TransformBroadcaster odom_feature_broadcaster;

  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = current_time;
  odom_trans.header.frame_id = "odom_feature";
  odom_trans.child_frame_id = "base_footprint";

  odom_trans.transform.translation.x = ackermann_odom.pose.pose.position.x;
  odom_trans.transform.translation.y = ackermann_odom.pose.pose.position.y;
  odom_trans.transform.rotation = ackermann_odom.pose.pose.orientation;

  odom_feature_broadcaster.sendTransform(odom_trans);

  // ************* publish imu odom *************
  // since all odometry is 6DOF we'll need a quaternion created from yaw
  geometry_msgs::Quaternion imu_odom_quat =
      tf::createQuaternionMsgFromYaw(th_imu);

  // next, we'll publish the odometry message over ROS
  nav_msgs::Odometry imu_odom;
  imu_odom.header.stamp = current_time;
  imu_odom.header.frame_id = "odom_feature";
  imu_odom.child_frame_id = "base_footprint_imu";

  // set the position
  imu_odom.pose.pose.position.x = x_imu;
  imu_odom.pose.pose.position.y = y_imu;
  imu_odom.pose.pose.position.z = 0.0;
  imu_odom.pose.pose.orientation = imu_odom_quat;

  // set the velocity
  imu_odom.twist.twist.linear.x = vx_imu;
  imu_odom.twist.twist.linear.y = vy_imu;
  imu_odom.twist.twist.angular.z = vth_imu;

  // publish the message
  imu_odom_pub.publish(imu_odom);

  // write to file
  ground_truth_file << std::fixed << std::setprecision(9)
                    << current_time.toSec() << " " << x_imu << " " << y_imu
                    << " " << 0.0 << " " << imu_odom_quat.x << " "
                    << imu_odom_quat.y << " " << imu_odom_quat.z << " "
                    << imu_odom_quat.w << "\n";

  // publish tf
  static tf::TransformBroadcaster odom_imu_broadcaster;

  geometry_msgs::TransformStamped odom_trans_imu;
  odom_trans_imu.header.stamp = current_time;
  odom_trans_imu.header.frame_id = "odom_feature";
  odom_trans_imu.child_frame_id = "base_footprint_imu";

  odom_trans_imu.transform.translation.x = imu_odom.pose.pose.position.x;
  odom_trans_imu.transform.translation.y = imu_odom.pose.pose.position.y;
  odom_trans_imu.transform.rotation = imu_odom.pose.pose.orientation;

  odom_imu_broadcaster.sendTransform(odom_trans_imu);

  ROS_INFO_THROTTLE(0.5, "Publish odom!");

  ground_truth_ind++;
  vehicle_odom_ind++;
}

void imuCB(const sensor_msgs::ImuConstPtr fdbk_imu_msg)
{
  double tmp_gyro_z = fdbk_imu_msg->angular_velocity.z;
  if (tmp_gyro_z < gyro_thresh)
    gyro_z = tmp_gyro_z;
}

void featureOdomCB(const nav_msgs::OdometryConstPtr feature_odom_msg)
{
  ros::Time current_time = ros::Time::now();

  // write to file
  double x = feature_odom_msg->pose.pose.position.x;
  double y = feature_odom_msg->pose.pose.position.y;
  double z = feature_odom_msg->pose.pose.position.z;
  double qx = feature_odom_msg->pose.pose.orientation.x;
  double qy = feature_odom_msg->pose.pose.orientation.y;
  double qz = feature_odom_msg->pose.pose.orientation.z;
  double qw = feature_odom_msg->pose.pose.orientation.w;

  feature_odom_file << std::fixed << std::setprecision(9)
                    << current_time.toSec() << " " << x << " " << y << " " << z
                    << " " << qx << " " << qy << " " << qz << " " << qw << "\n";

  static tf::TransformListener tf_listener;
  if (tf_listener.waitForTransform("base_footprint", "cam_front_optical",
                                   ros::Time(0), ros::Duration(0.1)))
  {
    try
    {
      tf::StampedTransform tf_cam2base_stamp;
      tf::Transform tf_cam2base;
      tf_listener.lookupTransform("base_footprint", "cam_front_optical",
                                  ros::Time(0), tf_cam2base_stamp);
      tf_cam2base.setBasis(tf_cam2base_stamp.getBasis());
      tf_cam2base.setOrigin(tf_cam2base_stamp.getOrigin());

      tf::Transform tf_base2odom;
      tf_base2odom.setOrigin(tf::Vector3(x, y, z));
      tf_base2odom.setRotation(tf::createQuaternionFromYaw(
          tf::getYaw(feature_odom_msg->pose.pose.orientation)));

      tf::Transform tf_cam2_odom = tf_base2odom * tf_cam2base;

      feature_odom_cam_file
          << std::fixed << std::setprecision(9) << current_time.toSec() << " "
          << tf_cam2_odom.getOrigin().x() << " " << tf_cam2_odom.getOrigin().y()
          << " " << tf_cam2_odom.getOrigin().z() << " "
          << tf_cam2_odom.getRotation().x() << " "
          << tf_cam2_odom.getRotation().y() << " "
          << tf_cam2_odom.getRotation().z() << " "
          << tf_cam2_odom.getRotation().w() << "\n";
    }
    catch (tf::TransformException ex)
    {
      ROS_ERROR("%s", ex.what());
    }
  }
  else
  {
    ROS_WARN_THROTTLE(1.0, "No camera pose is recorded");
  }

  feature_odom_ind++;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "test_performance_feature");
  ros::NodeHandle nh;
  
  ros::Subscriber fdbk_vel_sub =
      nh.subscribe("/deepps/vehicle_info", 100, &velCB);
  ros::Subscriber fdbk_imu_sub = nh.subscribe("/imu", 100, &imuCB);
  ros::Subscriber feature_odom_sub =
      nh.subscribe("/odom_feature", 100, &featureOdomCB);

  ackermann_odom_pub = nh.advertise<nav_msgs::Odometry>("ackermann_odom", 100);
  imu_odom_pub = nh.advertise<nav_msgs::Odometry>("imu_odom", 100);

  ground_truth_file.open("/home/yujr/catkin_ws/src/feature_match_odom-master/res/res/ground_truth.txt");
  vehicle_odom_file.open("/home/yujr/catkin_ws/src/feature_match_odom-master/res/res/vehicle_odom.txt");
  feature_odom_file.open("/home/yujr/catkin_ws/src/feature_match_odom-master/res/res/feature_odom.txt");
  feature_odom_cam_file.open("/home/yujr/catkin_ws/src/feature_match_odom-master/res/res/feature_odom_cam.txt");

  const std::string file_head = "# time x y z qx qy qz qw\n";
  ground_truth_file << file_head;
  vehicle_odom_file << file_head;
  feature_odom_file << file_head;
  feature_odom_cam_file << file_head;

  ros::spin();

  ground_truth_file.close();
  vehicle_odom_file.close();
  feature_odom_file.close();
  feature_odom_cam_file.close();

  return 0;
}
