#include <ros/ros.h>
#include "featureMatchOdom.h"

using namespace feature_match_odom;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "feature_match_odom_node");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  featureMatchOdom fmo(nh, nh_private);
  fmo.startLocalBa();

  ros::spin();

  return 0;
}
