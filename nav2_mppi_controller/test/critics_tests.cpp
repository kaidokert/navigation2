// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "settled_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/critics/constraint_critic.hpp"
#include "nav2_mppi_controller/critics/approach_angle_critic.hpp"
#include "nav2_mppi_controller/critics/goal_angle_critic.hpp"
#include "nav2_mppi_controller/critics/goal_critic.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_mppi_controller/critics/cost_critic.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_angle_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/prefer_forward_critic.hpp"
#include "nav2_mppi_controller/critics/twirling_critic.hpp"
#include "nav2_mppi_controller/critics/velocity_deadband_critic.hpp"
#include "utils_test.cpp"  // NOLINT

// Tests the various critic plugin functions

// ROS lock used from utils_test.cpp

using namespace mppi;  // NOLINT
using namespace mppi::critics;  // NOLINT
using namespace mppi::utils;  // NOLINT
using xt::evaluation_strategy::immediate;

class PathAngleCriticWrapper : public PathAngleCritic
{
public:
  PathAngleCriticWrapper()
  : PathAngleCritic()
  {
  }

  void setMode(int mode)
  {
    mode_ = static_cast<PathAngleMode>(mode);
  }
};

TEST(CriticTests, ConstraintsCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly and that defaults are reasonable
  ConstraintCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
  EXPECT_TRUE(critic.getMaxVelConstraint() > 0.0);
  EXPECT_TRUE(critic.getMinVelConstraint() < 0.0);

  // Scoring testing

  // provide velocities in constraints, should not have any costs
  state.vx = 0.40 * xt::ones<float>({1000, 30});
  state.vy = xt::zeros<float>({1000, 30});
  state.wz = xt::ones<float>({1000, 30});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // provide out of maximum velocity constraint
  auto last_batch_traj_in_full = xt::view(state.vx, -1, xt::all());
  last_batch_traj_in_full = 0.60 * xt::ones<float>({30});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0);
  // 4.0 weight * 0.1 model_dt * 0.1 error introduced * 30 timesteps = 1.2
  EXPECT_NEAR(costs(999), 1.2, 0.01);
  costs = xt::zeros<float>({1000});

  // provide out of minimum velocity constraint
  auto first_batch_traj_in_full = xt::view(state.vx, 1, xt::all());
  first_batch_traj_in_full = -0.45 * xt::ones<float>({30});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0);
  // 4.0 weight * 0.1 model_dt * 0.1 error introduced * 30 timesteps = 1.2
  EXPECT_NEAR(costs(1), 1.2, 0.01);
  costs = xt::zeros<float>({1000});

  // Now with ackermann, all in constraint so no costs to score
  state.vx = 0.40 * xt::ones<float>({1000, 30});
  state.wz = 1.5 * xt::ones<float>({1000, 30});
  data.motion_model = std::make_shared<AckermannMotionModel>(&param_handler, node->get_name());
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // Now violating the ackermann constraints
  state.wz = 2.5 * xt::ones<float>({1000, 30});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0);
  // 4.0 weight * 0.1 model_dt * (0.2 - 0.4/2.5) * 30 timesteps = 0.48
  EXPECT_NEAR(costs(1), 0.48, 0.01);
}

TEST(CriticTests, ObstacleCriticMisalignedParams) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", true);

  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  ObstaclesCritic critic;
  // Expect throw when settings mismatched
  EXPECT_THROW(
    critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler),
    nav2_core::ControllerException
  );
}

TEST(CriticTests, ObstacleCriticAlignedParams) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);

  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  ObstaclesCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
}


TEST(CriticTests, CostCriticMisAlignedParams) {
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
   auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", true);
  costmap_ros->on_configure(lstate);

  CostCritic critic;
  // Expect throw when settings mismatched
  EXPECT_THROW(
    critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler),
    nav2_core::ControllerException
  );
}

TEST(CriticTests, CostCriticAlignedParams) {
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
   auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);
  costmap_ros->on_configure(lstate);

  CostCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");
}

TEST(CriticTests, GoalAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly
  GoalAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path too far from `threshold_to_consider` to consider
  state.pose.pose.position.x = 1.0;
  path.reset(10);
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  path.yaws(9) = 3.14;
  goal.position.x = 10.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // Lets move it even closer, just to be sure it still doesn't trigger
  state.pose.pose.position.x = 9.2;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // provide state pose and path below `threshold_to_consider` to consider
  state.pose.pose.position.x = 9.7;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0);
  EXPECT_NEAR(costs(0), 9.42, 0.02);  // (3.14 - 0.0) * 3.0 weight
}

TEST(CriticTests, GoalAngleFinalSegmentOnly)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter("critic.final_segment_only", true);
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state;
  models::Trajectories trajectories;
  trajectories.reset(2, 3);
  models::Path path;
  path.reset(1);
  path.x(0) = .03;
  path.yaws(0) = -.30657;
  geometry_msgs::msg::Pose goal;
  goal.position.x = .03;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float dt = .05;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  GoalAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  for (int direction : {-1, 1}) {
    data.active_cusp_direction = direction;
    critic.score(data);
    EXPECT_FLOAT_EQ(costs(0), 0.0f);
    EXPECT_FLOAT_EQ(costs(1), 0.0f);
  }
  data.active_cusp_direction = 0;
  // Same endpoint position, but only one rollout aligns its heading.
  xt::view(trajectories.yaws, 1, xt::all()) = -.30657f;
  critic.score(data);
  EXPECT_NEAR(costs(0), .91971f, 1e-5);
  EXPECT_NEAR(costs(1), 0.0f, 1e-5);
  costs.fill(0.0f);
  path.reset(0);
  EXPECT_NO_THROW(critic.score(data));
  EXPECT_FLOAT_EQ(costs(0), 0.0f);
}

TEST(CriticTests, GoalAngleSymmetricCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly
  auto getParam = param_handler.getParamGetter("critic");
  bool symmetric_yaw_tolerance = true;
  getParam(symmetric_yaw_tolerance, "symmetric_yaw_tolerance", true);
  GoalAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path too far from `threshold_to_consider` to consider
  state.pose.pose.position.x = 9.7;
  path.reset(10);
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  path.yaws(9) = 3.14;
  goal.position.x = 10.0;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0);
  EXPECT_NEAR(costs(0), 0, 0.02);  // Should be zero cost due to symmetry

  path.yaws(9) = 0.0;
  critic.score(data);
  EXPECT_NEAR(costs(0), 0, 0.02);  // (0.0 - 0.0) * 3.0 weight
}

TEST(CriticTests, GoalCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  // Initialization testing

  // Make sure initializes correctly
  GoalCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing with all trajectories set to 0

  // provide state poses and path far, should not trigger
  state.pose.pose.position.x = 1.0;
  path.reset(10);
  path.x(9) = 10.0;
  path.y(9) = 0.0;
  goal.position.x = 10.0;
  critic.score(data);
  EXPECT_NEAR(costs(2), 0.0, 1e-6);  // (0 * 5.0 weight
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);  // Should all be 0 * 1000
  costs = xt::zeros<float>({1000});

  // provide state pose and path close
  path.x(9) = 0.5;
  path.y(9) = 0.0;
  goal.position.x = 0.5;
  critic.score(data);
  EXPECT_NEAR(costs(2), 2.5, 1e-6);  // (sqrt(10.0 * 10.0) * 5.0 weight
  EXPECT_NEAR(xt::sum(costs, immediate)(), 2500.0, 1e-6);  // should be 2.5 * 1000
}

TEST(CriticTests, GoalFinalSegmentOnly)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter("critic.final_segment_only", true);
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state;
  models::Trajectories trajectories;
  trajectories.reset(2, 3);
  models::Path path;
  path.reset(1);
  path.x(0) = .03;
  geometry_msgs::msg::Pose goal;
  goal.position.x = .03;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float dt = .05;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  GoalCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  for (int direction : {-1, 1}) {
    data.active_cusp_direction = direction;
    critic.score(data);
    EXPECT_FLOAT_EQ(costs(0), 0.0f);
    EXPECT_FLOAT_EQ(costs(1), 0.0f);
  }
  data.active_cusp_direction = 0;
  critic.score(data);
  EXPECT_GT(costs(0), 0.0f);
}

TEST(CriticTests, PathAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally

  // Initialization testing

  // Make sure initializes correctly
  PathAngleCriticWrapper critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path close, within pose tolerance so won't do anything
  state.pose.pose.position.x = 0.0;
  state.pose.pose.position.y = 0.0;
  path.reset(10);
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with less than PI/2 angular diff.
  path.x(9) = 0.95;
  goal.position.x = 0.95;
  data.furthest_reached_path_point = 2;  // So it grabs the 2 + offset_from_furthest_ = 6th point
  path.x(6) = 1.0;  // angle between path point and pose = 0 < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose > max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0);
  EXPECT_NEAR(costs(0), 3.9947, 1e-2);  // atan2(4,-1) [1.81] * 2.2 weight

  // Set mode to no directional preferences + reset costs
  critic.setMode(1);
  costs = xt::zeros<float>({1000});

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = 1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path pt and pose < max_angle_to_furthest_ IF non-directional
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0);
  // should use reverse orientation as the closer angle in no dir preference mode
  EXPECT_NEAR(costs(0), 2.9167, 1e-2);

  // Set mode to consider path directionality + reset costs
  critic.setMode(2);
  costs = xt::zeros<float>({1000});

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = 1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path pt and pose < max_angle_to_furthest_ IF non-directional
  path.y(6) = 0.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close but outside of tol. with more than PI/2 angular diff.
  path.x(6) = -1.0;  // angle between path point and pose < max_angle_to_furthest_
  path.y(6) = 4.0;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0);
  // should use reverse orientation as the closer angle in no dir preference mode
  EXPECT_NEAR(costs(0), 2.9167, 1e-2);

  PathAngleMode mode;
  mode = PathAngleMode::FORWARD_PREFERENCE;
  EXPECT_EQ(modeToStr(mode), std::string("Forward Preference"));
  mode = PathAngleMode::CONSIDER_FEASIBLE_PATH_ORIENTATIONS;
  EXPECT_EQ(modeToStr(mode), std::string("Consider Feasible Path Orientations"));
  mode = PathAngleMode::NO_DIRECTIONAL_PREFERENCE;
  EXPECT_EQ(modeToStr(mode), std::string("No Directional Preference"));
  mode = static_cast<PathAngleMode>(4);
  EXPECT_EQ(modeToStr(mode), std::string("Invalid mode!"));
}

TEST(CriticTests, PreferForwardCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally

  // Initialization testing

  // Make sure initializes correctly
  PreferForwardCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path far away, not within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.reset(10);
  path.x(9) = 10.0;
  goal.position.x = 10.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0f, 1e-6f);

  // provide state pose and path close to trigger behavior but with all forward motion
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.vx = xt::ones<float>({1000, 30});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0f, 1e-6f);

  // provide state pose and path close to trigger behavior but with all reverse motion
  state.vx = -1.0 * xt::ones<float>({1000, 30});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0f);
  EXPECT_NEAR(costs(0), 15.0f, 1e-3f);  // 1.0 * 0.1 model_dt * 5.0 weight * 30 length
}

TEST(CriticTests, TwirlingCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  TwirlingCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path far away, not within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.reset(10);
  path.x(9) = 10.0;
  goal.position.x = 10.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path close to trigger behavior but with no angular variation
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  state.wz = xt::zeros<float>({1000, 30});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // Provide nearby with some motion
  auto traj_view = xt::view(state.wz, 0, xt::all());
  traj_view = 10.0;
  critic.score(data);
  EXPECT_NEAR(costs(0), 100.0, 1e-6);  // (mean(10.0) * 10.0 weight
  costs = xt::zeros<float>({1000});

  // Equal positive/negative yaw rates have zero signed mean but nonzero
  // twirling cost: mean absolute rate .5 times weight 10 is exactly 5.
  for (size_t i = 0; i < 30; ++i) {traj_view(i) = i % 2 == 0 ? 0.5f : -0.5f;}
  critic.score(data);
  EXPECT_NEAR(costs(0), 5.0, 1e-6);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 5.0, 1e-6);
}

TEST(CriticTests, PathFollowCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  PathFollowCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and goal close within positional tolerances
  state.pose.pose.position.x = 2.0;
  path.reset(6);
  path.x(5) = 1.8;
  goal.position.x = 1.8;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path far enough to enable
  // pose differential is (0, 0) and (0.15, 0)
  path.x(5) = 0.15;
  goal.position.x = 0.15;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 750.0, 1e-2);  // 0.15 * 5 weight * 1000
}

TEST(CriticTests, PathFollowSingletonRetainsActiveEndpointCost)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("singleton_test");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "singleton_costmap", "", "singleton_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state;
  state.reset(2, 3);
  models::Trajectories trajectories;
  trajectories.reset(2, 3);
  models::Path path;
  path.reset(1);
  path.x(0) = 0.2f;
  geometry_msgs::msg::Pose goal;
  goal.position.x = 5.0;  // A later global goal must not replace this endpoint.
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  PathFollowCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  trajectories.x(0, 2) = 0.2f;
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0f, 1e-6);
  EXPECT_NEAR(costs(1), 1.0f, 1e-6);  // 5 * 0.2 m, despite one remaining point.
  EXPECT_FALSE(data.furthest_reached_path_point.has_value());
  EXPECT_FALSE(data.path_pts_valid.has_value());

  costs.fill(0.0f);
  trajectories.x(1, 2) = 0.1f;
  critic.score(data);
  EXPECT_NEAR(costs(1), 0.5f, 1e-6);  // Progress toward that point lowers cost.

  costs.fill(0.0f);
  state.pose.pose.position.x = goal.position.x;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs)(), 0.0f, 1e-6);  // Existing near-goal gate retained.

  state.pose.pose.position.x = 0.0;
  path.reset(0);
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs)(), 0.0f, 1e-6);  // Empty paths remain safe.
}

TEST(CriticTests, PathFollowDoesNotRewardGhostOvershootAtCusp)
{
  // Recorded office control: one real endpoint plus four 5cm ghost points.
  // Stopping at the actual cusp must beat continuing 20cm past the reversal.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ghost_endpoint_test");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "ghost_endpoint_costmap", "", "ghost_endpoint_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state;
  state.reset(2, 3);
  models::Trajectories trajectories;
  trajectories.reset(2, 3);
  models::Path path;
  path.reset(5);
  for (size_t i = 0; i < 5; ++i) {path.x(i) = 0.05f * i;}
  geometry_msgs::msg::Pose goal;
  goal.position.x = 5.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.real_path_point_count = 1;
  trajectories.x(0, 2) = 0.0f;  // Physical cusp.
  trajectories.x(1, 2) = 0.2f;  // Ghost endpoint past the cusp.
  PathFollowCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  critic.score(data);
  EXPECT_LT(costs(0), costs(1));
  EXPECT_NEAR(costs(0), 0.0f, 1e-6);
  EXPECT_NEAR(costs(1), 1.0f, 1e-6);

  // A longer real prefix clamps to its own endpoint, not to the ghost tail.
  costs.fill(0.0f);
  data.real_path_point_count = 3;
  trajectories.x(0, 2) = 0.1f;
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0f, 1e-6);
  EXPECT_NEAR(costs(1), 0.5f, 1e-6);

  // Ordinary paths without metadata retain their previous endpoint objective.
  costs.fill(0.0f);
  data.real_path_point_count.reset();
  trajectories.x(0, 2) = 0.0f;
  critic.score(data);
  EXPECT_NEAR(costs(0), 1.0f, 1e-6);
  EXPECT_NEAR(costs(1), 0.0f, 1e-6);
}

TEST(CriticTests, PathAlignCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(1000, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;  // from utils_tests tolerance of 0.25 positionally
  data.goal_checker = &goal_checker;

  // Initialization testing

  // Make sure initializes correctly
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide state poses and path close within positional tolerances
  state.pose.pose.position.x = 1.0;
  path.reset(10);
  path.x(9) = 0.85;
  goal.position.x = 0.85;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path far enough to enable
  // but data furthest point reached is 0 and offset default is 20, so returns
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path far enough to enable, with data to pass condition
  // but with empty trajectories and paths, should still be zero
  *data.furthest_reached_path_point = 21;
  path.x(9) = 0.15;
  goal.position.x = 0.15;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // provide state pose and path far enough to enable, with data to pass condition
  // and with a valid path to pass invalid path condition
  state.pose.pose.position.x = 0.0;
  data.path_pts_valid.reset();  // Recompute on new path
  path.reset(22);
  path.x(0) = 0;
  path.x(1) = 0.1;
  path.x(2) = 0.2;
  path.x(3) = 0.3;
  path.x(4) = 0.4;
  path.x(5) = 0.5;
  path.x(6) = 0.6;
  path.x(7) = 0.7;
  path.x(8) = 0.8;
  path.x(9) = 0.9;
  path.x(10) = 0.9;
  path.x(11) = 0.9;
  path.x(12) = 0.9;
  path.x(13) = 0.9;
  path.x(14) = 0.9;
  path.x(15) = 0.9;
  path.x(16) = 0.9;
  path.x(17) = 0.9;
  path.x(18) = 0.9;
  path.x(19) = 0.9;
  path.x(20) = 0.9;
  path.x(21) = 0.9;
  goal.position.x = 0.9;
  generated_trajectories.x = 0.66 * xt::ones<float>({1000, 30});
  critic.score(data);
  // 0.66 * 1000 * 10 weight * 6 num pts eval / 6 normalization term
  EXPECT_NEAR(xt::sum(costs, immediate)(), 6600.0, 1e-2);

  // provide state pose and path far enough to enable, with data to pass condition
  // but path is blocked in collision
  auto * costmap = costmap_ros->getCostmap();
  // island in the middle of lethal cost to cross. Costmap defaults to size 5x5 @ 10cm resolution
  for (unsigned int i = 11; i <= 30; ++i) {  // 1.1m-3m
    for (unsigned int j = 11; j <= 30; ++j) {  // 1.1m-3m
      costmap->setCost(i, j, 254);
    }
  }

  data.path_pts_valid.reset();  // Recompute on new path
  costs = xt::zeros<float>({1000});
  path.x = 1.5 * xt::ones<float>({22});
  path.y = 1.5 * xt::ones<float>({22});
  goal.position.x = 1.5;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);
}

TEST(CriticTests, PathAlignCriticShortPath)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(100, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(100, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({100});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;
  data.goal_checker = &goal_checker;

  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  // 1. N = 0 (empty path -> no-op/zero cost, no crash)
  path.reset(0);
  data.furthest_reached_path_point.reset();
  data.path_pts_valid.reset();
  costs = xt::zeros<float>({100});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // 2. N = 1 (single point -> no-op/zero cost, no crash)
  path.reset(1);
  path.x(0) = 0.5;
  path.y(0) = 0.0;
  goal.position.x = 2.0;  // far from pose at origin (0, 0)
  data.furthest_reached_path_point.reset();
  data.path_pts_valid.reset();
  costs = xt::zeros<float>({100});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);

  // 3. N = 2 (two points -> valid alignment evaluation, no crash)
  path.reset(2);
  path.x(0) = 0.0;
  path.x(1) = 0.5;
  path.y(0) = 0.0;
  path.y(1) = 0.0;
  goal.position.x = 0.5;
  goal.position.y = 0.0;
  state.pose.pose.position.x = 0.0;
  state.pose.pose.position.y = 0.0;
  generated_trajectories.x = 0.25 * xt::ones<float>({100, 30});
  generated_trajectories.y = 0.10 * xt::ones<float>({100, 30});
  data.furthest_reached_path_point = 1;
  data.path_pts_valid.reset();
  costs = xt::zeros<float>({100});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0);

  // 4. N = 3 (three points -> valid evaluation)
  path.reset(3);
  path.x(0) = 0.0;
  path.x(1) = 0.25;
  path.x(2) = 0.5;
  path.y(0) = 0.0;
  path.y(1) = 0.0;
  path.y(2) = 0.0;
  goal.position.x = 0.5;
  data.furthest_reached_path_point = 2;
  data.path_pts_valid.reset();
  costs = xt::zeros<float>({100});
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0.0);

  // 5. N = 20, lookahead = 1 (normal path startup -> verify non-terminal short-path behavior is preserved)
  path.reset(20);
  for (unsigned int i = 0; i < 20; ++i) {
    path.x(i) = static_cast<float>(i) * 0.1f;
    path.y(i) = 0.0f;
  }
  goal.position.x = 2.0;  // far from state at (0, 0)
  data.furthest_reached_path_point = 1;  // Lookahead = 1 < offset (default 20)
  data.path_pts_valid.reset();
  costs = xt::zeros<float>({100});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0.0, 1e-6);
}

TEST(CriticTests, PathAlignPreviewGhostsDoNotHideShortRealSegment)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("preview_align_node");
  node->declare_parameter("critic.offset_from_furthest", 3);
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "preview_costmap", "", "preview_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state;
  state.reset(2, 12);
  models::Trajectories trajectories;
  trajectories.reset(2, 12);
  trajectories.x.fill(0.0f);
  trajectories.y.fill(0.0f);
  trajectories.yaws.fill(0.0f);
  for (size_t i = 0; i < 12; ++i) {
    trajectories.yaws(1, i) = 0.4f;
  }
  models::Path path;
  path.reset(7);
  for (size_t i = 0; i < 7; ++i) {
    path.x(i) = static_cast<float>(i) * 0.01f;
    path.y(i) = 0.0f;
    path.yaws(i) = 0.0f;
  }
  geometry_msgs::msg::Pose goal;
  goal.position.x = 5.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.real_path_point_count = 3;
  data.furthest_reached_path_point = 1;
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  critic.score(data);
  EXPECT_GT(costs(1), costs(0)) << "preview ghosts disabled short-segment yaw guidance";
  EXPECT_NEAR(costs(0), 0.0f, 1e-6);
  EXPECT_NEAR(costs(1), 4.0f, 1e-5);

  // Preserve the ordinary long-path startup bearing gate without short-path metadata.
  data.real_path_point_count.reset();
  costs.fill(0.0f);
  critic.score(data);
  EXPECT_NEAR(costs(1), 0.0f, 1e-6);
}

TEST(CriticTests, PathAlignShortRealEndpointKeepsYawCostAtZeroProgress)
{
  // Office first-cusp stall: pruning leaves one real endpoint and preview ghosts,
  // while stationary rollouts report furthest=0. Heading guidance must survive.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("zero_progress_align");
  node->declare_parameter("critic.offset_from_furthest", 3);
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "zero_progress_costmap", "", "zero_progress_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  models::State state; state.reset(2, 12);
  models::Trajectories trajectories; trajectories.reset(2, 12);
  trajectories.x.fill(0.0f); trajectories.y.fill(0.0f); trajectories.yaws.fill(0.0f);
  for (size_t i = 0; i < 12; ++i) {trajectories.yaws(1, i) = 0.4f;}
  models::Path path; path.reset(5);
  for (size_t i = 0; i < 5; ++i) {
    path.x(i) = static_cast<float>(i) * 0.05f;
    path.y(i) = 0.0f; path.yaws(i) = 0.0f;
  }
  geometry_msgs::msg::Pose goal; goal.position.x = 5.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.real_path_point_count = 1; data.furthest_reached_path_point = 0;
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0f, 1e-6);
  EXPECT_NEAR(costs(1), 4.0f, 1e-5) << "stationary endpoint lost its orientation cost";
  // The normal long-path startup gate remains unchanged.
  data.real_path_point_count.reset(); costs.fill(0.0f); critic.score(data);
  EXPECT_NEAR(costs(1), 0.0f, 1e-6);
}

TEST(CriticTests, ShortCuspTerminalHeadingRejectsWrongYawStopAndPreservesFinalGoal)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("cusp_terminal_heading");
  node->declare_parameter("critic.cost_weight", 0.0);
  node->declare_parameter("critic.cusp_terminal_heading_weight", 30.0);
  auto cm = std::make_shared<SettledTestCostmap>(
    "cusp_terminal_heading_costmap", "", "cusp_terminal_heading_costmap", true);
  ParametersHandler params(node);
  rclcpp_lifecycle::State lstate; cm->on_configure(lstate);
  models::State state; state.reset(2, 12);
  models::Trajectories trajectories; trajectories.reset(2, 12);
  trajectories.x.fill(0.0f); trajectories.y.fill(0.0f); trajectories.yaws.fill(-0.7f);
  trajectories.yaws(1, 11) = -0.63f;
  models::Path path; path.reset(9);
  for (size_t i = 0; i < 9; ++i) {
    path.x(i) = 0.05f * i; path.y(i) = 0.0f; path.yaws(i) = 1.5f;
  }
  path.yaws(1) = -0.7f;  // Real cusp; ghosts deliberately point elsewhere.
  geometry_msgs::msg::Pose goal; goal.position.x = 5.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.real_path_point_count = 2; data.furthest_reached_path_point = 0;
  PathAlignCritic critic; critic.on_configure(node, "mppi", "critic", cm, &params);
  for (int gear : {-1, 1}) {
    data.active_cusp_direction = gear; costs.fill(0.0f); critic.score(data);
    EXPECT_NEAR(costs(0), 0.0f, 1e-5);
    EXPECT_NEAR(costs(1), 2.1f, 1e-5);
  }
  // Reduced 174419 counterfactual: the averaged costs prefer an exact-position
  // stop with 0.0693rad yaw error over a gate-reaching arc at 0.03956rad.
  // Reproduce that ranking, then require the explicit terminal objective to
  // reverse it. This tests cost ranking, not physical trajectory generation.
  trajectories.yaws(0, 11) = -0.7f + 0.069298267f;
  trajectories.yaws(1, 11) = -0.7f + 0.039562702f;
  node->declare_parameter("baseline.cost_weight", 0.0);
  PathAlignCritic baseline;
  baseline.on_configure(node, "mppi", "baseline", cm, &params);
  costs(0) = 1.260418415f; costs(1) = 1.469377041f;
  baseline.score(data);
  EXPECT_LT(costs(0), costs(1)) << "Default scoring should retain the negative control";
  costs(0) = 1.260418415f; costs(1) = 1.469377041f;
  critic.score(data);
  EXPECT_LT(costs(1), costs(0)) << "Wrong-yaw positional stop still wins";
  data.active_cusp_direction = 0; costs.fill(0.0f); critic.score(data);
  EXPECT_NEAR(costs(1), 0.0f, 1e-5) << "Final goal behavior changed";
  data.active_cusp_direction = -1; data.real_path_point_count = 5;
  costs.fill(0.0f); critic.score(data);
  EXPECT_NEAR(costs(1), 0.0f, 1e-5) << "Long approach was pulled toward a distant cusp";
  data.real_path_point_count.reset(); costs.fill(0.0f); critic.score(data);
  EXPECT_NEAR(costs(1), 0.0f, 1e-5) << "Missing real-prefix metadata enabled the objective";
}

TEST(CriticTests, VelocityDeadbandCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  auto getParam = param_handler.getParamGetter("critic");
  std::vector<double> deadband_velocities_;
  getParam(deadband_velocities_, "deadband_velocities", std::vector<double>{0.08, 0.08, 0.08});
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<OmniMotionModel>();

  // Initialization testing

  // Make sure initializes correctly and that defaults are reasonable
  VelocityDeadbandCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Scoring testing

  // provide velocities out of deadband bounds, should not have any costs
  state.vx = 0.80 * xt::ones<float>({1000, 30});
  state.vy = 0.60 * xt::ones<float>({1000, 30});
  state.wz = 0.80 * xt::ones<float>({1000, 30});
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // Test cost value
  state.vx = 0.01 * xt::ones<float>({1000, 30});
  state.vy = 0.02 * xt::ones<float>({1000, 30});
  state.wz = 0.021 * xt::ones<float>({1000, 30});
  critic.score(data);
  // 35.0 weight * 0.1 model_dt * (0.07 + 0.06 + 0.059) * 30 timesteps = 56.7
  EXPECT_NEAR(costs(1), 19.845, 0.01);
}

TEST(CriticTests, ApproachAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(1000, 30);
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1000});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  ApproachAngleCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);
  EXPECT_EQ(critic.getName(), "critic");

  // Goal at (10,0) facing pi; every trajectory terminally misaligned (yaw 0)
  goal.position.x = 10.0;
  goal.orientation.w = 0.0;
  goal.orientation.z = 1.0;  // yaw = pi
  path.reset(10);

  // PROVABLY INERT beyond activation_distance (0.5): far away...
  state.pose.pose.position.x = 1.0;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);
  // ...and just outside the boundary
  state.pose.pose.position.x = 9.45;
  critic.score(data);
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6);

  // ACTIVE inside: dist 0.3 -> ramp = 1 - 0.3/0.5 = 0.4
  state.pose.pose.position.x = 9.7;
  critic.score(data);
  EXPECT_GT(xt::sum(costs, immediate)(), 0) <<
    "critic inert during final approach: M5 would park misaligned again";
  // terminal yaw error pi, weight 3.0, ramp 0.4 -> 3.7699
  EXPECT_NEAR(costs(0), 3.7699, 0.02);

  // A terminally-ALIGNED trajectory scores ~zero
  costs = xt::zeros<float>({1000});
  for (unsigned int t = 0; t != 30; t++) {
    generated_trajectories.yaws(0, t) = 3.14159;
  }
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0, 1e-3);
  EXPECT_GT(costs(1), 1.0);

  // Ramp grows monotonically as the goal nears: dist 0.1 -> ramp 0.8
  auto costs_at_03 = costs(1);
  costs = xt::zeros<float>({1000});
  state.pose.pose.position.x = 9.9;
  critic.score(data);
  EXPECT_GT(costs(1), costs_at_03);
}

class FootprintThresholdWitness : public ObstaclesCritic
{
public:
  float threshold() const {return possible_collision_cost_;}
};

TEST(CriticTests, HardFootprintVetoBelowInflationThreshold)
{
  // Run222626, filter capture1660.95, filtered forecast step31. Three
  // recorded cells isolate the raster corner/center-threshold disagreement.
  // This is a frozen scoring reduction, not proof of a live escape trajectory.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("raster_footprint");
  node->declare_parameter("critic.consider_footprint", false);
  node->declare_parameter("critic.collision_check_footprint_only", true);
  node->declare_parameter("critic.repulsion_weight", 0.0);
  auto cm = std::make_shared<SettledTestCostmap>(
    "raster_footprint_map", "", "raster_footprint_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{"inflation_layer"}));
  cm->declare_parameter("inflation_layer.plugin", std::string("nav2_costmap_2d::InflationLayer"));
  cm->declare_parameter("inflation_layer.cost_scaling_factor", 6.0);
  cm->declare_parameter("inflation_layer.inflation_radius", 0.42);
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.06000000052154064,-0.1609999928623438],"
      "[0.2859999928623438,-0.1609999928623438],"
      "[0.2859999928623438,0.1609999928623438],"
      "[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.0));
  rclcpp_lifecycle::State lifecycle;
  ASSERT_EQ(cm->on_configure(lifecycle), nav2_util::CallbackReturn::SUCCESS);
  cm->getLayeredCostmap()->resizeMap(150, 150, .02, 3.2200000000000055, -1.220000000000001);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 22500, 0);
  grid->setCost(72, 75, 49);
  grid->setCost(56, 70, nav2_costmap_2d::LETHAL_OBSTACLE);
  grid->setCost(56, 71, nav2_costmap_2d::LETHAL_OBSTACLE);
  models::State state; state.reset(2, 1);
  models::Trajectories trajectories; trajectories.reset(2, 1);
  trajectories.x.fill(4.6692705154418945f);
  trajectories.y.fill(0.29113420844078064f);
  trajectories.yaws.fill(2.9674172401428223f);
  // Second rollout is an independently clear pose in the same grid.
  trajectories.x(1, 0) = 5.0f;
  models::Path path; path.reset(2);
  geometry_msgs::msg::Pose goal; goal.position.x = 100.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = .05;
  CriticData data{state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  ParametersHandler handler(node);
  FootprintThresholdWitness critic;
  critic.on_configure(node, "mppi", "critic", cm, &handler);
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(grid);
  ASSERT_EQ(checker.footprintCostAtPose(trajectories.x(0, 0), trajectories.y(0, 0),
      trajectories.yaws(0, 0), cm->getRobotFootprint()), 254.0);
  critic.score(data);
  ASSERT_GT(critic.threshold(), 49.0f);  // Confirm the actual optimization shortcut premise.
  EXPECT_GE(costs(0), 1000000.0f);
  EXPECT_FLOAT_EQ(costs(1), 0.0f);
  EXPECT_FALSE(data.fail_flag);
  trajectories.x(1, 0) = trajectories.x(0, 0);
  costs.fill(0.0f); data.fail_flag = false;
  critic.score(data);
  EXPECT_GE(costs(0), 1000000.0f);
  EXPECT_GE(costs(1), 1000000.0f);
  EXPECT_TRUE(data.fail_flag);  // No safe rollout is not a successful batch.
  trajectories.x(1, 0) = 5.0f;
  // Removing only the two lethal cells must restore clear soft scoring.
  grid->setCost(56, 70, 0); grid->setCost(56, 71, 0);
  costs.fill(0.0f); data.fail_flag = false;
  critic.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0f);  // Same reference-cell soft score as before.
  EXPECT_FLOAT_EQ(costs(1), 0.0f);
  EXPECT_FALSE(data.fail_flag);
  cm->on_cleanup(lifecycle);
}

TEST(CriticTests, HardFootprintVetoPreservesPointSoftScoring)
{
  // Office correction: rear reference is free while a front corner touches a wall.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("hard_footprint");
  node->declare_parameter("critic.consider_footprint", false);
  node->declare_parameter("critic.collision_check_footprint_only", true);
  auto cm = std::make_shared<SettledTestCostmap>(
    "hard_footprint_map", "", "hard_footprint_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.050,-0.151],[0.276,-0.151],[0.276,0.151],[-0.050,0.151]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.01));
  rclcpp_lifecycle::State lifecycle;
  cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  unsigned int mx, my;
  for (double y = 0.5; y < 1.5; y += 0.025) {
    ASSERT_TRUE(grid->worldToMap(1.275, y, mx, my));
    grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  models::State state; state.reset(2, 10);
  models::Trajectories trajectories; trajectories.reset(2, 10);
  trajectories.x.fill(0.0); trajectories.y.fill(1.0); trajectories.yaws.fill(0.0);
  for (size_t t = 0; t < 10; ++t) {trajectories.x(0, t) = 1.0;}
  models::Path path; path.reset(2);
  geometry_msgs::msg::Pose goal; goal.position.x = 100.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05;
  CriticData data{state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  ParametersHandler handler(node);
  ObstaclesCritic critic;
  critic.on_configure(node, "mppi", "critic", cm, &handler);
  critic.score(data);
  EXPECT_GT(costs(0), 1000.0);
  EXPECT_FLOAT_EQ(costs(1), 0.0);
  EXPECT_FALSE(data.fail_flag);  // The second rollout remains clear.
  // Same footprint collision with a free reference point inside the goal
  // truncation radius: proximity must not bypass the physical corner veto.
  node->set_parameter(rclcpp::Parameter("critic.goal_truncation_distance", 0.22));
  ObstaclesCritic near_goal;
  near_goal.on_configure(node, "mppi", "critic", cm, &handler);
  goal.position.x = 1.15; goal.position.y = 1.0;
  costs.fill(0.0); data.fail_flag = false;
  near_goal.score(data);
  EXPECT_GT(costs(0), 1000.0);
  EXPECT_FLOAT_EQ(costs(1), 0.0);
  EXPECT_FALSE(data.fail_flag);
  goal.position.x = 100.0;
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", false));
  ObstaclesCritic point;
  point.on_configure(node, "mppi", "critic", cm, &handler);
  costs.fill(0.0); data.fail_flag = false; point.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0);  // Reproduces the unsafe center-only blind spot.
  cm->on_cleanup(lifecycle);
}

class NearGoalCollisionReduction : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("near_goal_collision");
    node->declare_parameter("critic.consider_footprint", false);
    node->declare_parameter("critic.goal_truncation_distance", 0.22);
    cm = std::make_shared<SettledTestCostmap>(
      "near_goal_map", "", "near_goal_map", true);
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    cm->on_configure(lifecycle);
    cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
    std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 6400, 0);
    handler = std::make_unique<ParametersHandler>(node);
    critic.on_configure(node, "mppi", "critic", cm, handler.get());
  }
  void TearDown() override {cm->on_cleanup(lifecycle);}
  void lethal(double x, double y)
  {
    unsigned int mx, my;
    ASSERT_TRUE(cm->getCostmap()->worldToMap(x, y, mx, my));
    cm->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  void score(const std::vector<float> & xs, bool expect_collision, int cusp = 0)
  {
    models::State state; state.reset(2, xs.size());
    models::Trajectories trajectories; trajectories.reset(2, xs.size());
    trajectories.y.fill(0.0f); trajectories.yaws.fill(0.0f);
    for (size_t j = 0; j < xs.size(); ++j) {
      trajectories.x(0, j) = xs[j];
      trajectories.x(1, j) = xs[j];
      trajectories.y(1, j) = 0.50f;  // Clear positive control outside truncation radius.
    }
    models::Path path; path.reset(3);
    geometry_msgs::msg::Pose goal; goal.position.x = 1.0; goal.orientation.w = 1.0;
    xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05;
    CriticData data{state, trajectories, path, goal, costs, dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    data.active_cusp_direction = cusp;
    critic.score(data);
    if (expect_collision) {EXPECT_GT(costs(0), 1000.0f);} else {EXPECT_FLOAT_EQ(costs(0), 0.0f);}
    EXPECT_FLOAT_EQ(costs(1), 0.0f);
    EXPECT_FALSE(data.fail_flag);  // Mixed batch: a clear alternative exists.
    if (expect_collision) {
      for (size_t j = 0; j < xs.size(); ++j) {trajectories.y(1, j) = 0.0f;}
      costs.fill(0.0f);
      critic.score(data);
      EXPECT_TRUE(data.fail_flag);  // All-colliding batch must request fallback.
    }
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
  rclcpp_lifecycle::State lifecycle;
  std::unique_ptr<ParametersHandler> handler;
  ObstaclesCritic critic;
};

TEST_F(NearGoalCollisionReduction, LethalInitialPoseInsideGoalRadiusIsNotIgnored)
{
  lethal(0.875, 0.0);
  score({0.875f, 0.9375f, 1.0f}, true);
}

TEST_F(NearGoalCollisionReduction, LethalFirstEntryInsideGoalRadiusIsNotIgnored)
{
  lethal(0.875, 0.0);
  score({0.60f, 0.875f, 1.0f}, true);
}

TEST_F(NearGoalCollisionReduction, InitialFreePoseInsideRadiusDoesNotHideImmediateCollision)
{
  // Binary-exact sample coordinates avoid an unrelated float/double cell-edge
  // discrepancy between setting the grid and evaluating the float rollout.
  lethal(0.9375, 0.0);
  score({0.8125f, 0.9375f, 1.0f}, true);
}

TEST_F(NearGoalCollisionReduction, IntermediateCuspNearGlobalGoalDoesNotTruncateCollisionChecks)
{
  lethal(1.3125, 0.0);
  score({0.60f, 0.875f, 1.3125f}, true, -1);
  score({0.60f, 0.875f, 1.3125f}, true, 1);
}

TEST_F(NearGoalCollisionReduction, ClearTerminalEntryRetainsPostGoalWallTruncation)
{
  lethal(1.3125, 0.0);
  score({0.60f, 0.875f, 1.3125f}, false);
  score({0.60f, 0.875f, 1.0f}, false);
}

TEST(CriticTests, TerminalStubRanksArrivalAboveStoppedAndOvershotForBothGears)
{
  // Reduction of the reported 3-point/7.5cm terminal stall. This tests the real
  // critic ranking, not whole-controller convergence or a braking guarantee.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("terminal_stub_ranking");
  node->declare_parameter("critic.threshold_to_consider", 0.0);
  auto cm = std::make_shared<SettledTestCostmap>(
    "terminal_stub_map", "", "terminal_stub_map", true);
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  ParametersHandler handler(node);
  PathFollowCritic critic; critic.on_configure(node, "mppi", "critic", cm, &handler);
  for (int gear : {-1, 1}) {
    for (size_t points : {1u, 2u, 3u}) {
      models::State state; state.reset(3, 41);
      models::Trajectories trajectories; trajectories.reset(3, 41);
      trajectories.x.fill(0.0f); trajectories.y.fill(0.0f); trajectories.yaws.fill(0.0f);
      for (size_t j = 0; j < 41; ++j) {
        trajectories.x(1, j) = gear * 0.075f * j / 40.0f;
        trajectories.x(2, j) = gear * 0.150f * j / 40.0f;
      }
      models::Path path; path.reset(points);
      for (size_t j = 0; j < points; ++j) {
        path.x(j) = gear * 0.075f * (j + 1) / points;
        path.y(j) = 0.0f; path.yaws(j) = 0.0f;
      }
      geometry_msgs::msg::Pose goal; goal.position.x = gear * 0.075;
      xt::xtensor<float, 1> costs = xt::zeros<float>({3}); float dt = 0.05;
      CriticData data{state, trajectories, path, goal, costs, dt,
        false, nullptr, nullptr, std::nullopt, std::nullopt};
      critic.score(data);
      EXPECT_NEAR(costs(1), 0.0f, 1e-6);
      EXPECT_GT(costs(0), costs(1) + 0.3f);  // Stopped short must lose.
      EXPECT_GT(costs(2), costs(1) + 0.3f);  // Overshoot must also lose.
      EXPECT_NEAR(costs(0), costs(2), 1e-5);
    }
  }
  cm->on_cleanup(lifecycle);
}

TEST(CriticTests, ObstaclesCriticGoalTruncation)
{
  // M6 regression (notes/cusp_handoff_fix_plan.md, 2026-09-25): rollout tails
  // extend ~2s PAST the goal; when the goal sits near a wall, every
  // progress-making sample "collides" beyond it and the optimizer parks at a
  // standoff (critics_stats showed ObstaclesCritic 0 -> 5e10 at the brake
  // moment). Points past the goal are fantasy — the executed robot stops.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);
  double trunc;
  getParam(trunc, "goal_truncation_distance", 0.3);

  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  // Lethal wall at x = 2.0m, beyond the goal at (1.5, 0)
  auto * cm = costmap_ros->getCostmap();
  unsigned int mx, my;
  for (double y = -1.0; y <= 1.0; y += 0.05) {
    if (cm->worldToMap(2.0, y, mx, my)) {
      cm->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }

  models::State state;
  state.pose.pose.position.x = 1.0;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(50, 30);
  // One straight line from x=1.0 through the goal at 1.5 into the wall at 2.0
  for (unsigned int i = 0; i != 50; i++) {
    for (unsigned int t = 0; t != 30; t++) {
      generated_trajectories.x(i, t) = 1.0 + 0.05 * static_cast<double>(t);
      generated_trajectories.y(i, t) = 0.0;
    }
  }
  models::Path path;
  path.reset(10);
  geometry_msgs::msg::Pose goal;
  goal.position.x = 1.5;
  goal.orientation.w = 1.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({50});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  ObstaclesCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  critic.score(data);
  // With goal truncation, the post-goal wall must not veto the trajectory
  EXPECT_NEAR(xt::sum(costs, immediate)(), 0, 1e-6) <<
    "post-goal obstacle vetoed a trajectory that stops at the goal";
  EXPECT_FALSE(data.fail_flag);
}

TEST(CriticTests, IntermediateCuspKeepsPathCostsNearFinalGoal)
{
  // A turnaround ends near its start, but its intermediate cusp is still ahead.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("cusp_near_goal");
  node->declare_parameter("align.use_path_orientations", true);
  node->declare_parameter("align.offset_from_furthest", 3);
  auto cm = std::make_shared<SettledTestCostmap>(
    "cusp_costmap", "", "cusp_costmap", true);
  ParametersHandler handler(node);
  rclcpp_lifecycle::State lstate;
  cm->on_configure(lstate);
  models::State state; state.reset(2, 12);
  models::Trajectories trajectories; trajectories.reset(2, 12);
  trajectories.x.fill(0.0f); trajectories.y.fill(0.0f); trajectories.yaws.fill(0.0f);
  for (size_t i = 0; i < 12; ++i) {trajectories.yaws(1, i) = 0.4f;}
  models::Path path; path.reset(5);
  geometry_msgs::msg::Pose goal; goal.position.x = 0.008;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.real_path_point_count = 1; data.furthest_reached_path_point = 0;
  PathAlignCritic align;
  align.on_configure(node, "mppi", "align", cm, &handler);
  PathFollowCritic follow;
  follow.on_configure(node, "mppi", "follow", cm, &handler);
  for (int gear : {-1, 1}) {
    for (size_t i = 0; i < 5; ++i) {
      path.x(i) = gear * (0.15f + static_cast<float>(i) * 0.05f);
      path.y(i) = 0.0f; path.yaws(i) = 0.0f;
    }
    data.active_cusp_direction = gear;
    data.path_pts_valid = std::vector<bool>(5, true);
    costs.fill(0.0f); align.score(data);
    EXPECT_NEAR(costs(0), 1.5f, 1e-5);
    EXPECT_NEAR(costs(1), 10.0f * std::sqrt(0.15f * 0.15f + 0.4f * 0.4f), 1e-5)
      << "active cusp lost yaw guidance";
    trajectories.x.fill(0.0f);
    trajectories.x(1, 11) = gear * 0.10f;
    costs.fill(0.0f); follow.score(data);
    EXPECT_NEAR(costs(0), 0.75f, 1e-5);
    EXPECT_NEAR(costs(1), 0.25f, 1e-5) << "active cusp lost endpoint guidance";
    // Final approach retains existing proximity suppression for both critics.
    data.active_cusp_direction = 0;
    costs.fill(0.0f); align.score(data); follow.score(data);
    EXPECT_NEAR(costs(0), 0.0f, 1e-6); EXPECT_NEAR(costs(1), 0.0f, 1e-6);
    trajectories.x.fill(0.0f);
  }
}

TEST(CriticTests, PathAlignActiveCuspRetainsCurvedRealReferenceBeforeOffset)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("cusp_start_align");
  node->declare_parameter("critic.offset_from_furthest", 3);
  node->declare_parameter("critic.trajectory_point_step", 1);
  node->declare_parameter("critic.use_path_orientations", true);
  auto cm = std::make_shared<SettledTestCostmap>(
    "cusp_start_map", "", "cusp_start_map", true);
  ParametersHandler params(node);
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  PathAlignCritic critic; critic.on_configure(node, "mppi", "critic", cm, &params);
  for (int gear : {1, -1}) {
    SCOPED_TRACE(gear);
    models::State state; state.reset(2, 7);
    models::Trajectories trajectories; trajectories.reset(2, 7);
    models::Path path; path.reset(11);
    for (size_t k = 0; k < 7; ++k) {
      const float angle = static_cast<float>(k) * 0.1f;
      path.x(k) = gear * 0.55f * std::sin(angle);
      path.y(k) = gear * 0.55f * (std::cos(angle) - 1.0f);
      path.yaws(k) = -angle;
      for (size_t b = 0; b < 2; ++b) {
        trajectories.x(b, k) = path.x(k); trajectories.y(b, k) = path.y(k);
        trajectories.yaws(b, k) = b == 0 ? path.yaws(k) : 0.0f;
      }
    }
    for (size_t k = 7; k < 11; ++k) {
      path.x(k) = 5.0f; path.y(k) = -5.0f; path.yaws(k) = 2.0f;
    }
    geometry_msgs::msg::Pose goal; goal.position.x = 10.0;
    xt::xtensor<float, 1> costs = xt::zeros<float>({2}); float dt = 0.05f;
    CriticData data = {state, trajectories, path, goal, costs, dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    data.real_path_point_count = 7; data.furthest_reached_path_point = 1;
    data.path_pts_valid = std::vector<bool>(11, true);
    data.active_cusp_direction = gear;
    critic.score(data);
    EXPECT_NEAR(costs(0), 0.0f, 1e-5) << "exact curved reference must not be penalized";
    EXPECT_GT(costs(1), costs(0) + 0.1f) << "startup gate discarded required cusp heading";
    const auto reference_costs = costs;
    for (size_t k = 7; k < 11; ++k) {
      path.x(k) = -20.0f; path.y(k) = 20.0f; path.yaws(k) = -2.0f;
    }
    costs.fill(0.0f); critic.score(data);
    EXPECT_EQ(costs, reference_costs) << "preview ghosts changed the active real reference";
    data.active_cusp_direction = 0; costs.fill(0.0f); critic.score(data);
    EXPECT_NEAR(costs(1), 0.0f, 1e-6) << "ordinary startup bearing policy changed";
  }
  cm->on_cleanup(lifecycle);
}
