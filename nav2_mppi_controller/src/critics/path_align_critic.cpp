// Copyright (c) 2023 Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/critics/path_align_critic.hpp"

namespace mppi::critics
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

void PathAlignCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 10.0f);

  getParam(max_path_occupancy_ratio_, "max_path_occupancy_ratio", 0.07f);
  getParam(offset_from_furthest_, "offset_from_furthest", 20);
  getParam(trajectory_point_step_, "trajectory_point_step", 4);
  getParam(threshold_to_consider_, "threshold_to_consider", 0.5f);
  getParam(use_path_orientations_, "use_path_orientations", false);
  getParam(cusp_terminal_heading_weight_, "cusp_terminal_heading_weight", 0.0f);
  if (cusp_terminal_heading_weight_ < 0.0f) {
    throw std::invalid_argument("Cusp terminal heading weight must be nonnegative");
  }

  RCLCPP_INFO(
    logger_,
    "ReferenceTrajectoryCritic instantiated with %d power and %f weight",
    power_, weight_);
}

void PathAlignCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  const size_t total_path_pts = data.path.x.shape(0);
  if (total_path_pts == 0) {
    return;
  }

  // If the robot is already within the goal checker's position AND yaw tolerances,
  // do not penalize residual yaw error — let the robot decelerate cleanly to a stop.
  // Final-goal proximity must not disable a required intermediate cusp.
  if (data.active_cusp_direction == 0 && data.goal_checker) {
    geometry_msgs::msg::Pose pose_tolerance;
    geometry_msgs::msg::Twist velocity_tolerance;
    if (data.goal_checker->getTolerances(pose_tolerance, velocity_tolerance)) {
      const double dx = data.state.pose.pose.position.x - data.goal.position.x;
      const double dy = data.state.pose.pose.position.y - data.goal.position.y;
      const double dist_sq = dx * dx + dy * dy;
      const double max_early_exit_dist = static_cast<double>(pose_tolerance.position.x);
      if (dist_sq <= max_early_exit_dist * max_early_exit_dist) {
        const double yaw_tol = tf2::getYaw(pose_tolerance.orientation);
        const double robot_yaw = tf2::getYaw(data.state.pose.pose.orientation);
        const double goal_yaw = tf2::getYaw(data.goal.orientation);
        const double dyaw = std::abs(angles::shortest_angular_distance(robot_yaw, goal_yaw));
        if (dyaw <= yaw_tol) {
          auto node = parent_.lock();
          if (node) {
            RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 1000,
              "[PAC] SKIP: within goal checker tolerances (dist=%.3f <= %.3f, dyaw=%.3f <= %.3f)",
              std::sqrt(dist_sq), pose_tolerance.position.x, dyaw, yaw_tol);
          }
          return;
        }
      }
    }
  }

  // Averaged tracking cost can prefer an exact-position, wrong-yaw stop over
  // the transient motion needed to satisfy an intermediate cusp. Apply this
  // separate terminal objective only to a short real prefix. A distant cusp
  // must not pull the heading through intervening path curvature; final goals
  // keep their existing goal critic. Default zero preserves previous scoring.
  const size_t real_cusp_pts = std::min(
    data.real_path_point_count.value_or(data.path.yaws.size()), data.path.yaws.size());
  if (cusp_terminal_heading_weight_ > 0.0f && data.active_cusp_direction != 0 &&
    real_cusp_pts > 0 && real_cusp_pts <= 4 && data.trajectories.yaws.shape(1) > 0)
  {
    auto error = xt::fabs(utils::shortest_angular_distance(
      xt::view(data.trajectories.yaws, xt::all(), -1),
      data.path.yaws(real_cusp_pts - 1)));
    data.costs += cusp_terminal_heading_weight_ * error;
  }

  if (total_path_pts == 1) {
    if (!use_path_orientations_) {
      return;
    }
    const size_t batch_size = data.trajectories.x.shape(0);
    auto && cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    const float goal_yaw = (data.path.yaws.shape(0) > 0) ?
      data.path.yaws(0) : static_cast<float>(tf2::getYaw(data.goal.orientation));
    const auto T_yaw = xt::view(
      data.trajectories.yaws, xt::all(),
      xt::range(0, _, trajectory_point_step_));
    const auto traj_sampled_size = T_yaw.shape(1);
    for (size_t t = 0; t < batch_size; ++t) {
      float summed_yaw_err = 0.0f;
      for (size_t p = 0; p < traj_sampled_size; ++p) {
        float dyaw = angles::shortest_angular_distance(goal_yaw, T_yaw(t, p));
        summed_yaw_err += fabsf(dyaw);
      }
      cost[t] = summed_yaw_err / static_cast<float>(traj_sampled_size);
    }
    if (power_ > 1u) {
      data.costs += xt::pow(std::move(cost) * weight_, power_);
    } else {
      data.costs += std::move(cost) * weight_;
    }
    return;
  }

  // Don't apply close to goal, let the goal critics take over (final segment only)
  if (data.active_cusp_direction == 0 && utils::withinPositionGoalTolerance(
      threshold_to_consider_, data.state.pose.pose, data.goal))
  {
    auto node = parent_.lock();
    if (node) {
      RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 1000,
        "[PAC] SKIP: within goal tolerance (threshold=%.3f)", threshold_to_consider_);
    }
    return;
  }

  // Don't apply when first getting bearing w.r.t. the path, unless on terminal approach
  utils::setPathFurthestPointIfNotSet(data);
  const size_t total_segments = total_path_pts - 1;
  size_t path_segments_count =
    std::min(*data.furthest_reached_path_point, total_segments);

  const bool active_real_cusp = (data.active_cusp_direction != 0 && real_cusp_pts > 0);
  if (active_real_cusp) {
    path_segments_count = real_cusp_pts;
  }

  const bool short_segment = (total_path_pts <= 4);
  if (path_segments_count == 0) {
    if (!short_segment) {
      return;
    }
    path_segments_count = 1;
  }

  const bool is_terminal_path =
    (path_segments_count >= total_segments) || short_segment;

  if (!is_terminal_path && path_segments_count < offset_from_furthest_) {
    auto node = parent_.lock();
    if (node) {
      RCLCPP_DEBUG_THROTTLE(
        logger_, *node->get_clock(), 1000,
        "[PAC] SKIP: furthest_reached=%zu < offset=%zu",
        path_segments_count, static_cast<size_t>(offset_from_furthest_));
    }
    return;
  }

  // Number of points to include in local path vector.
  // On terminal approach, retain all points to include the final goal pose.
  const size_t num_pts = is_terminal_path ?
    total_path_pts : std::min(*data.furthest_reached_path_point + 1, total_path_pts);

  // Don't apply when dynamic obstacles are blocking significant proportions of the local path
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  std::vector<bool> & path_pts_valid = *data.path_pts_valid;
  float invalid_ctr = 0.0f;
  const size_t check_pts = std::min(num_pts, path_pts_valid.size());
  for (size_t i = 0; i < check_pts; i++) {
    if (!path_pts_valid[i]) {invalid_ctr += 1.0f;}
  }
  float check_pts_flt = static_cast<float>(check_pts);
  if (check_pts_flt > 0.0f && invalid_ctr / check_pts_flt > max_path_occupancy_ratio_ &&
    invalid_ctr > 2.0f)
  {
    auto node = parent_.lock();
    if (node) {
      RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 1000,
        "[PAC] SKIP: path_occupancy %.1f%% > max %.1f%% (invalid=%d/%zu)",
        100.0f * invalid_ctr / check_pts_flt, 100.0f * max_path_occupancy_ratio_,
        static_cast<int>(invalid_ctr), check_pts);
    }
    return;
  }

  const size_t batch_size = data.trajectories.x.shape(0);
  auto && cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});

  // Find integrated distance in the path
  std::vector<float> path_integrated_distances(num_pts, 0.0f);
  std::vector<utils::Pose2D> path(num_pts);
  float dx = 0.0f, dy = 0.0f;
  for (unsigned int i = 1; i != num_pts; i++) {
    auto & pose = path[i - 1];
    pose.x = data.path.x(i - 1);
    pose.y = data.path.y(i - 1);
    pose.theta = data.path.yaws(i - 1);

    dx = data.path.x(i) - pose.x;
    dy = data.path.y(i) - pose.y;
    path_integrated_distances[i] = path_integrated_distances[i - 1] + sqrtf(dx * dx + dy * dy);
  }

  // Finish populating the path vector
  auto & final_pose = path[num_pts - 1];
  final_pose.x = data.path.x(num_pts - 1);
  final_pose.y = data.path.y(num_pts - 1);
  final_pose.theta = data.path.yaws(num_pts - 1);

  float summed_path_dist = 0.0f, dyaw = 0.0f;
  unsigned int num_samples = 0u;
  unsigned int path_pt = 0u;
  float traj_integrated_distance = 0.0f;

  // Get strided trajectory information
  const auto T_x = xt::view(
    data.trajectories.x, xt::all(),
    xt::range(0, _, trajectory_point_step_));
  const auto T_y = xt::view(
    data.trajectories.y, xt::all(),
    xt::range(0, _, trajectory_point_step_));
  const auto T_yaw = xt::view(
    data.trajectories.yaws, xt::all(),
    xt::range(0, _, trajectory_point_step_));
  const auto traj_sampled_size = T_x.shape(1);

  for (size_t t = 0; t < batch_size; ++t) {
    summed_path_dist = 0.0f;
    num_samples = 0u;
    traj_integrated_distance = 0.0f;
    path_pt = 0u;
    float Tx_m1 = T_x(t, 0);
    float Ty_m1 = T_y(t, 0);
    for (size_t p = 1; p < traj_sampled_size; p++) {
      const float Tx = T_x(t, p);
      const float Ty = T_y(t, p);
      dx = Tx - Tx_m1;
      dy = Ty - Ty_m1;
      Tx_m1 = Tx;
      Ty_m1 = Ty;
      traj_integrated_distance += sqrtf(dx * dx + dy * dy);
      path_pt = utils::findClosestPathPt(
        path_integrated_distances, traj_integrated_distance, path_pt);

      const bool is_valid = (path_pt < path_pts_valid.size()) ? path_pts_valid[path_pt] : true;
      if (is_valid) {
        const auto & pose = path[path_pt];
        dx = pose.x - Tx;
        dy = pose.y - Ty;
        num_samples++;
        if (use_path_orientations_) {
          dyaw = angles::shortest_angular_distance(pose.theta, T_yaw(t, p));
          summed_path_dist += sqrtf(dx * dx + dy * dy + dyaw * dyaw);
        } else {
          summed_path_dist += sqrtf(dx * dx + dy * dy);
        }
      }

      // Truncate rollout evaluation once trajectory reaches the end of the available path.
      // Points beyond the endpoint are post-arrival dwell at the goal with zero path deviation.
      if (path_integrated_distances.back() > 0.0f &&
        traj_integrated_distance >= path_integrated_distances.back())
      {
        num_samples += (traj_sampled_size - 1 - p);
        break;
      }
    }
    if (num_samples > 0u) {
      cost[t] = summed_path_dist / static_cast<float>(num_samples);
    } else {
      cost[t] = 0.0f;
    }
  }

  if (power_ > 1u) {
    data.costs += xt::pow(std::move(cost) * weight_, power_);
  } else {
    data.costs += std::move(cost) * weight_;
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::PathAlignCritic,
  mppi::critics::CriticFunction)
