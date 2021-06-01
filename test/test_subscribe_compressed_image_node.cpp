#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int cnt = 0;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    // Get the msg image
    cv::Mat img;
    img = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::imshow("image", img);
    cv::waitKey(10);

    std::string img_name = "/home/ros/img" + std::to_string(cnt) + ".jpg";
//    cv::imwrite(img_name, img);
    cnt++;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'jpg'.", msg->encoding.c_str());
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // Subscriber to image
  image_transport::Subscriber sub =
      it.subscribe("/camera/image/birdview_seg", 1, imageCallback, ros::VoidPtr(),
                   image_transport::TransportHints("compressed"));

  ros::spin();

  return 0;
}
