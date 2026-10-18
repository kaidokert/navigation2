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
  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 0.5f);
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
  // Final-goal proximity must not disable a required intermediate cusp.
  // On the final segment, preserve the usual handoff to goal critics.
  if (!enabled_ || (data.active_cusp_direction == 0 &&
    utils::withinPositionGoalTolerance(
      threshold_to_consider_, data.state.pose.pose, data.goal)))
  {
    auto node = parent_.lock();
    if (node) {
      const char * reason = !enabled_ ? "disabled" : "within goal tolerance";
      RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 1000,
        "[PAC] SKIP: %s (threshold=%.3f)", reason, threshold_to_consider_);
    }
    return;
  }

  // Averaged tracking cost can prefer an exact-position, wrong-yaw stop over
  // the transient motion needed to satisfy an intermediate cusp. Apply this
  // separate terminal objective only to a short real prefix. A distant cusp
  // must not pull the heading through intervening path curvature; final goals
  // keep their existing goal critic. Default zero preserves previous scoring.
  const size_t real_cusp_pts = std::min(
    data.real_path_point_count.value_or(0), data.path.yaws.size());
  if (cusp_terminal_heading_weight_ > 0.0f && data.active_cusp_direction != 0 &&
    real_cusp_pts > 0 && real_cusp_pts <= 4 && data.trajectories.yaws.shape(1) > 0)
  {
    auto error = xt::fabs(utils::shortest_angular_distance(
      xt::view(data.trajectories.yaws, xt::all(), -1),
      data.path.yaws(real_cusp_pts - 1)));
    data.costs += cusp_terminal_heading_weight_ * error;
  }

  if (data.path.x.shape(0) < 2) {
    return;
  }

  // Don't apply when first getting bearing w.r.t. the path, unless on terminal approach
  utils::setPathFurthestPointIfNotSet(data);
  const size_t total_path_pts = data.path.x.shape(0);
  const size_t total_segments = total_path_pts - 1;
  size_t path_segments_count =
    std::min(*data.furthest_reached_path_point, total_segments);

  // Preview ghosts must not turn a short real segment into a long-path startup.
  // Keep their orientation geometry, but classify the segment before extension.
  const size_t real_path_pts = std::min(
    data.real_path_point_count.value_or(total_path_pts), total_path_pts);
  // A required cusp segment already has a known outgoing bearing. Suppressing
  // alignment until the random batch makes progress loses curvature guidance
  // exactly during the steering reversal. Match travelled distance against the
  // complete real segment; preview ghosts must not extend this reference.
  const bool active_real_cusp = data.active_cusp_direction != 0 &&
    data.real_path_point_count.has_value() && real_path_pts > 0;
  if (active_real_cusp) {
    path_segments_count = real_path_pts;
  }
  const bool short_real_segment =
    data.real_path_point_count.has_value() && real_path_pts > 0 && real_path_pts <= 4;
  if (path_segments_count == 0) {
    if (!short_real_segment) {
      return;
    }
    // A stationary rollout at a pruned cusp still needs endpoint heading cost.
    // One path sample suffices; retaining zero would both disable alignment and
    // underflow the final-pose index below. Long-path startup remains unchanged.
    path_segments_count = 1;
  }
  const bool is_terminal_path =
    active_real_cusp || (path_segments_count >= total_segments) ||
    (real_path_pts > 0 && real_path_pts <= 4);

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

  float path_segments_flt = static_cast<float>(path_segments_count);

  // Don't apply when dynamic obstacles are blocking significant proportions of the local path
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  std::vector<bool> & path_pts_valid = *data.path_pts_valid;
  float invalid_ctr = 0.0f;
  for (size_t i = 0; i < path_segments_count; i++) {
    if (!path_pts_valid[i]) {invalid_ctr += 1.0f;}
    if (invalid_ctr / path_segments_flt > max_path_occupancy_ratio_ && invalid_ctr > 2.0f) {
      auto node = parent_.lock();
      if (node) {
        RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 1000,
          "[PAC] SKIP: path_occupancy %.1f%% > max %.1f%% (invalid=%d/%zu)",
          100.0f * invalid_ctr / path_segments_flt, 100.0f * max_path_occupancy_ratio_,
          static_cast<int>(invalid_ctr), path_segments_count);
      }
      return;
    }
  }

  const size_t batch_size = data.trajectories.x.shape(0);
  auto && cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});

  // Find integrated distance in the path
  std::vector<float> path_integrated_distances(path_segments_count, 0.0f);
  std::vector<utils::Pose2D> path(path_segments_count);
  float dx = 0.0f, dy = 0.0f;
  for (unsigned int i = 1; i != path_segments_count; i++) {
    auto & pose = path[i - 1];
    pose.x = data.path.x(i - 1);
    pose.y = data.path.y(i - 1);
    pose.theta = data.path.yaws(i - 1);

    dx = data.path.x(i) - pose.x;
    dy = data.path.y(i) - pose.y;
    path_integrated_distances[i] = path_integrated_distances[i - 1] + sqrtf(dx * dx + dy * dy);
  }

  // Finish populating the path vector
  auto & final_pose = path[path_segments_count - 1];
  final_pose.x = data.path.x(path_segments_count - 1);
  final_pose.y = data.path.y(path_segments_count - 1);
  final_pose.theta = data.path.yaws(path_segments_count - 1);

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

      // The nearest path point to align to needs to be not in collision, else
      // let the obstacle critic take over in this region due to dynamic obstacles
      if (path_pts_valid[path_pt]) {
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
