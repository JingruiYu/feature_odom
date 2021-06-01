#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

#include <ros/ros.h>

#include "eigen3/Eigen/Eigen"
#include "ceres/ceres.h"

namespace feature_match_odom
{

struct Pose2d
{
  double x;
  double y;
  double yaw_radians;
};

struct Observation
{
  int pose_index;
  int point_index;
  Eigen::Vector2d point;
};

// Convert yaw angle to rotation matrix.
template <typename T> Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians)
{
  const T cos_yaw = ceres::cos(yaw_radians);
  const T sin_yaw = ceres::sin(yaw_radians);

  Eigen::Matrix<T, 2, 2> rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}

// Normalizes the angle in radians between [-pi and pi).
template <typename T> inline T NormalizeAngle(const T& angle_radians)
{
  // Use ceres::floor because it is specialized for double and Jet types.
  T two_pi(2.0 * M_PI);
  return angle_radians -
         two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}

// Defines a local parameterization for updating the angle to be constrained in
// [-pi to pi).
class AngleLocalParameterization
{
public:
  template <typename T>
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const
  {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  static ceres::LocalParameterization* Create()
  {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

struct BirdviewReprojectionError
{
  BirdviewReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y)
  {
  }

  template <typename T>
  bool operator()(const T* const pose_x, const T* const pose_y,
                  const T* const pose_yaw, const T* const point_x,
                  const T* const point_y, T* residuals) const
  {

    // get translation vector
    const Eigen::Matrix<T, 2, 1> trans(*pose_x, *pose_y);
    // get rotation matrix: observe -> reference
    const Eigen::Matrix<T, 2, 2> rotation = RotationMatrix2D(*pose_yaw);

    // get point in ref frame
    const Eigen::Matrix<T, 2, 1> p_ref(*point_x, *point_y);

    // get point in test frame
    const Eigen::Matrix<T, 2, 1> p_test(static_cast<T>(observed_x),
                                        static_cast<T>(observed_y));

    // calculate reprojected point
    Eigen::Matrix<T, 2, 1> p_proj = rotation * p_test + trans;

    // calculate residual
    residuals[0] = p_ref(0, 0) - p_proj(0, 0);
    residuals[1] = p_ref(1, 0) - p_proj(1, 0);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y)
  {
    return (new ceres::AutoDiffCostFunction<BirdviewReprojectionError, 2, 1, 1,
                                            1, 1, 1>(
        new BirdviewReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

class optimization
{
public:
  optimization(std::vector<Pose2d>& poses, std::vector<Eigen::Vector2d>& points,
               std::vector<std::vector<Observation>>& observations);

  void solveProblem();

private:
  void buildProblem();

  void setOptions(ceres::Solver::Options& options);

  // parameters
  std::vector<Pose2d>& poses_;
  std::vector<Eigen::Vector2d>& points_;

  // input
  std::vector<std::vector<Observation>>& observations_;

  ceres::Problem problem_;
};
}

#endif // OPTIMIZATION_H
