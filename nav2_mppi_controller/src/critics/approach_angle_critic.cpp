// Copyright (c) 2026 Kaido Kert
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

#include "nav2_mppi_controller/critics/approach_angle_critic.hpp"

#include <cmath>

#include "tf2/utils.h"

namespace mppi::critics
{

void ApproachAngleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.0f);
  getParam(activation_distance_, "activation_distance", 0.5f);

  RCLCPP_INFO(
    logger_,
    "ApproachAngleCritic instantiated with %d power, %f weight, "
    "%f activation distance",
    power_, weight_, activation_distance_);
}

void ApproachAngleCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  // Hard gate on euclidean distance to the TRUE goal (data.goal — never the
  // truncated/ghost path end). Exactly zero beyond activation_distance_: the
  // GoalAngleCritic A/B showed any heading pressure at range turns
  // pivot-in-place into the best-scoring sample on a car-like robot.
  const double dx = data.state.pose.pose.position.x - data.goal.position.x;
  const double dy = data.state.pose.pose.position.y - data.goal.position.y;
  const double dist = std::hypot(dx, dy);
  // activation_distance <= 0 disables the critic (also guards the 0/0 ramp
  // that would emit NaN into the softmax — review 2026-09-25)
  if (activation_distance_ <= 0.0f || dist > activation_distance_) {
    return;
  }

  // Linear ramp: 0 at the activation boundary (continuous — no cost cliff),
  // full weight at the goal.
  const float ramp = static_cast<float>(1.0 - dist / activation_distance_);

  const float goal_yaw = static_cast<float>(tf2::getYaw(data.goal.orientation));

  // Terminal yaw of each candidate trajectory: heading the robot would
  // ARRIVE with — the quantity the goal checker will judge.
  const auto terminal_yaws = xt::view(
    data.trajectories.yaws, xt::all(),
    data.trajectories.yaws.shape(1) - 1);

  auto angular_distances = xt::eval(
    xt::fabs(utils::shortest_angular_distance(terminal_yaws, goal_yaw)));

  if (power_ > 1u) {
    data.costs += xt::pow(angular_distances * (weight_ * ramp), power_);
  } else {
    data.costs += angular_distances * (weight_ * ramp);
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::ApproachAngleCritic,
  mppi::critics::CriticFunction)
