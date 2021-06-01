#include "optimization.h"

namespace feature_match_odom
{
optimization::optimization(std::vector<Pose2d>& poses,
                           std::vector<Eigen::Vector2d>& points,
                           std::vector<std::vector<Observation>>& observations)
    : poses_(poses), points_(points), observations_(observations)
{
}

void optimization::solveProblem()
{
  buildProblem();

  ceres::Solver::Options options;
  setOptions(options);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem_, &summary);
//  ROS_INFO_STREAM(summary.BriefReport());
}

void optimization::buildProblem()
{
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0)/*NULL*/;
  ceres::LocalParameterization* angle_local_parameterization =
      AngleLocalParameterization::Create();

  for (auto ob_pose : observations_)
  {
    for (auto ob : ob_pose)
    {
      // get pointers to poses
      double& pose_x = poses_[ob.pose_index].x;
      double& pose_y = poses_[ob.pose_index].y;
      double& pose_yaw = poses_[ob.pose_index].yaw_radians;

      // get pointers to points
      double& point_x = points_[ob.point_index][0];
      double& point_y = points_[ob.point_index][1];

      // create cost function with observations
      ceres::CostFunction* cost_function =
          BirdviewReprojectionError::Create(ob.point[0], ob.point[1]);

      problem_.AddResidualBlock(cost_function, loss_function, &pose_x, &pose_y,
                                &pose_yaw, &point_x, &point_y);

      problem_.SetParameterization(&pose_yaw, angle_local_parameterization);
    }
  }
}

void optimization::setOptions(ceres::Solver::Options& options)
{
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 10;
  options.num_threads = 3;
  options.max_solver_time_in_seconds = 1.0;
  options.minimizer_progress_to_stdout = true;
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
}
}
