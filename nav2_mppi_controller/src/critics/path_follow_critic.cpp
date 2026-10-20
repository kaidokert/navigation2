#include "nav2_mppi_controller/tools/decision_recorder.hpp"
// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include "nav2_mppi_controller/critics/path_follow_critic.hpp"

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>

namespace mppi::critics
{

void PathFollowCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 1.4f);
  getParam(offset_from_furthest_, "offset_from_furthest", 6);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0f);
}

void PathFollowCritic::score(CriticData & data)
{
  if (!enabled_ || data.path.x.shape(0) == 0 ||
    (data.active_cusp_direction == 0 &&
    utils::withinPositionGoalTolerance(threshold_to_consider_, data.state.pose.pose, data.goal)))
  {
    recorder::observe("critic_skipped", [&] {
        return recorder::Json{{"name", "PathFollowCritic"}, {"reason", "entry_gate"}};
  });
    return;
  }

  // Ghosts preserve rollout preview/orientation but are not planner destinations.
  // Rewarding their endpoint pulls the robot past a real direction cusp, where
  // inversion handoff still requires the original endpoint (office regression).
  const size_t real_points = std::min(
    data.real_path_point_count.value_or(data.path.x.shape(0)), data.path.x.shape(0));
  if (real_points == 0) {
    return;
  }
  const size_t path_size = real_points - 1;
  size_t offseted_idx = 0;

  // Pruning can leave only the active endpoint before the robot reaches it.
  // Retain its position cost: dropping it removes all longitudinal guidance
  // when GoalCritic is disabled. A singleton has no segment/furthest index or
  // validity search; in particular, path_size - 1 would underflow here.
  if (path_size > 0) {
    utils::setPathFurthestPointIfNotSet(data);
    utils::setPathCostsIfNotSet(data, costmap_ros_);
    offseted_idx = std::min(
      *data.furthest_reached_path_point + offset_from_furthest_, path_size);

    // Drive to the first valid path point, in case of dynamic obstacles on path
    // we want to drive past it, not through it
    bool valid = false;
    while (!valid && offseted_idx < path_size - 1) {
      valid = (*data.path_pts_valid)[offseted_idx];
      if (!valid) {
        offseted_idx++;
      }
    }
  }

  const auto path_x = data.path.x(offseted_idx);
  const auto path_y = data.path.y(offseted_idx);
  recorder::observe("critic_target", [&] {
      return recorder::Json{
      {"name", "PathFollowCritic"}, {"index", offseted_idx}, {"x", path_x}, {"y", path_y},
      {"weight", weight_}, {"power", power_}, {"real_points", real_points},
      {"furthest",
        data.furthest_reached_path_point ? recorder::Json(*data.furthest_reached_path_point) :
        recorder::Json(nullptr)}};
  });

  // Diagnostic: target point selection
  {
    float dist_to_target = hypotf(
      data.state.pose.pose.position.x - path_x,
      data.state.pose.pose.position.y - path_y);
    auto node = parent_.lock();
    if (node) {
      RCLCPP_DEBUG_THROTTLE(logger_, *node->get_clock(), 200,
        "[PFC] target_idx=%zu/%zu furthest=%zu dist_to_target=%.4fm",
        offseted_idx, path_size,
        data.furthest_reached_path_point.value_or(0), dist_to_target);
    }
  }

  const auto last_x = xt::view(data.trajectories.x, xt::all(), -1);
  const auto last_y = xt::view(data.trajectories.y, xt::all(), -1);

  if (power_ > 1u) {
    data.costs += xt::pow(weight_ * std::move(xt::hypot(last_x - path_x, last_y - path_y)), power_);
  } else {
    data.costs += weight_ * std::move(xt::hypot(last_x - path_x, last_y - path_y));
  }
}
}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::PathFollowCritic,
  mppi::critics::CriticFunction)
