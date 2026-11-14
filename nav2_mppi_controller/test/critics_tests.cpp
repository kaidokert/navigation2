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

#include <array>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"
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
#include "nav2_mppi_controller/critics/terminal_approach_critic.hpp"
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

TEST(CriticTests, GoalAngleSymmetricCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

TEST(CriticTests, PathAngleCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

  // Now try again with some wiggling noise
  traj_view = xt::random::randn<float>({30}, 0.0, 0.5);
  critic.score(data);
  EXPECT_NEAR(costs(0), 3.3, 4e-1);  // (mean of noise with mu=0, sigma=0.5 * 10.0 weight
}

TEST(CriticTests, PathFollowCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

TEST(CriticTests, PathAlignCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  // The stationary rollout starts 0.66 m along the first edge's tangent.
  // Its origin projection is capped at that edge's endpoint (0.10 m), so it
  // remains 0.56 m away instead of jumping to a remote reference vertex.
  EXPECT_NEAR(xt::sum(costs, immediate)(), 5600.0, 1e-2);
  generated_trajectories.x.fill(0.0f);
  costs.fill(0.0f);
  critic.score(data);
  EXPECT_FLOAT_EQ(xt::sum(costs, immediate)(), 0.0f);
  generated_trajectories.x.fill(0.66f);

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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

  // 5. N = 20, lookahead = 1: preserve non-terminal short-path behavior.
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

TEST(CriticTests, VelocityDeadbandCritic)
{
  // Standard preamble
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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

TEST(CriticTests, ObstaclesCriticDoesNotTruncateAtGoal)
{
  // A forecast crossing the goal still describes motion unless it models a stop.
  // The deprecated truncation radius cannot hide its later lethal wall sample.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  EXPECT_GE(costs(0), 1000000.0f);
  EXPECT_TRUE(data.fail_flag);
}

TEST(CriticTests, PathAlignIncludesReachedEndpoint)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("endpoint_alignment");
  node->declare_parameter("critic.trajectory_point_step", 1);
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "endpoint_costmap", "", "endpoint_costmap", true);
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto * grid = costmap_ros->getCostmap();
  grid->resizeMap(300, 300, 0.01, 0.0, 0.0);
  ParametersHandler parameters(node);
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &parameters);

  for (const int direction : {-1, 1}) {
    for (const size_t point_count : {2u, 3u}) {
      models::State state;
      state.reset(3, point_count);
      models::Trajectories trajectories;
      trajectories.reset(3, point_count);
      models::Path path;
      path.reset(point_count);
      geometry_msgs::msg::Pose goal;
      goal.position.x = 2.5;
      goal.position.y = 2.5;
      xt::xtensor<float, 1> costs = xt::zeros<float>({3});
      float dt = 0.05;
      CriticData data = {state, trajectories, path, goal, costs, dt,
        false, nullptr, nullptr, std::nullopt, std::nullopt};
      for (size_t i = 0; i < point_count; ++i) {
        const float yaw = direction * static_cast<float>(i) * 0.13962634f;
        path.x(i) = 1.0f + 0.561f * std::sin(yaw);
        path.y(i) = 1.0f + 0.561f * (1.0f - std::cos(yaw));
        path.yaws(i) = yaw;
        for (size_t row = 0; row < 3; ++row) {
          trajectories.x(row, i) = path.x(i);
          trajectories.y(row, i) = path.y(i) + (row == 2 ? 0.04f : 0.0f);
          trajectories.yaws(row, i) = row == 1 ? -yaw : yaw;
        }
      }
      data.furthest_reached_path_point = point_count - 1;
      critic.score(data);
      EXPECT_NEAR(costs(0), 0.0f, 1e-5) << direction << ":" << point_count;
      EXPECT_GT(costs(1), costs(0) + 0.1f);
      EXPECT_GT(costs(2), costs(0) + 0.1f);

      // The newly included endpoint must be checked, never assumed traversable:
      // valid while its cell is free, invalid once it is lethal.
      ASSERT_TRUE(data.path_pts_valid.has_value());
      ASSERT_EQ(data.path_pts_valid->size(), point_count);
      EXPECT_TRUE(data.path_pts_valid->back());
      unsigned int mx, my;
      ASSERT_TRUE(grid->worldToMap(path.x(point_count - 1), path.y(point_count - 1), mx, my));
      grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
      data.path_pts_valid.reset();
      costs.fill(0.0f);
      critic.score(data);
      ASSERT_EQ(data.path_pts_valid->size(), point_count);
      EXPECT_FALSE(data.path_pts_valid->back());
      EXPECT_NEAR(costs(0), 0.0f, 1e-5);
      grid->setCost(mx, my, nav2_costmap_2d::FREE_SPACE);
    }
  }
}


TEST(CriticTests, PathAlignPhaseOriginPreservesExactCurveRanking)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("phase_alignment");
  node->declare_parameter("critic.trajectory_point_step", 4);
  node->declare_parameter("critic.offset_from_furthest", 2);
  node->declare_parameter("critic.cost_weight", 28.0);
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "phase_costmap", "", "phase_costmap", true);
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  costmap_ros->getCostmap()->resizeMap(400, 400, 0.01, -2.0, -2.0);
  ParametersHandler parameters(node);
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &parameters);

  constexpr double radius = 0.561;
  constexpr double speed = 0.15;
  constexpr double dt = 0.05;
  constexpr double edge_angle = 0.13962634015954636;  // Eight degrees.
  const std::array<double, 8> fractions = {0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.0};
  const size_t exact_row = fractions.size() - 1;
  // Positive phases reproduce the wrong 90%/75% under-steering winners from EXP-066;
  // negative phases verify that negative initial projections do not prefer over-steering.
  for (int gear : {-1, 1}) {
    for (double phase : {-0.5, -0.25, 0.0, 0.25, 0.5}) {
      for (bool duplicate_origin : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "gear=" << gear << " phase=" << phase <<
          " duplicate_origin=" << duplicate_origin);
        const size_t duplicate = duplicate_origin ? 1u : 0u;
        models::State state;
        state.reset(fractions.size(), 41);
        const double initial_yaw = gear * phase * edge_angle;
        state.pose.pose.position.x = radius * std::sin(initial_yaw);
        state.pose.pose.position.y = radius * (1.0 - std::cos(initial_yaw));
        models::Trajectories trajectories;
        trajectories.reset(fractions.size(), 41);
        models::Path path;
        path.reset(15 + duplicate);
        for (size_t i = 0; i < 15 + duplicate; ++i) {
          const size_t unique_index = i >= duplicate ? i - duplicate : 0u;
          const double yaw = gear * static_cast<double>(unique_index) * edge_angle;
          path.x(i) = radius * std::sin(yaw);
          path.y(i) = radius * (1.0 - std::cos(yaw));
          path.yaws(i) = yaw;
        }
        for (size_t row = 0; row < fractions.size(); ++row) {
          double x = state.pose.pose.position.x;
          double y = state.pose.pose.position.y;
          double yaw = initial_yaw;
          const double curvature = fractions[row] / radius;
          for (size_t j = 0; j < 41; ++j) {
            const double distance = gear * speed * dt;
            const double angle = distance * curvature;
            if (curvature == 0.0) {
              x += distance * std::cos(yaw);
              y += distance * std::sin(yaw);
            } else {
              x += (std::sin(yaw + angle) - std::sin(yaw)) / curvature;
              y += (std::cos(yaw) - std::cos(yaw + angle)) / curvature;
            }
            yaw += angle;
            trajectories.x(row, j) = x;
            trajectories.y(row, j) = y;
            trajectories.yaws(row, j) = yaw;
          }
        }
        geometry_msgs::msg::Pose goal;
        goal.position.x = 1.5;
        goal.position.y = 1.5;
        xt::xtensor<float, 1> costs = xt::zeros<float>({fractions.size()});
        float model_dt = dt;
        CriticData data = {state, trajectories, path, goal, costs, model_dt,
          false, nullptr, nullptr, std::nullopt, std::nullopt};
        // Same reached endpoint for every row/policy: four complete path edges.
        data.furthest_reached_path_point = 4u + duplicate;
        data.path_pts_valid = std::vector<bool>(15 + duplicate, true);
        critic.score(data);
        for (size_t row = 0; row < exact_row; ++row) {
          EXPECT_LT(costs(exact_row), costs(row)) << "exact=" << costs(exact_row) <<
            " fraction=" << fractions[row] << " cost=" << costs(row);
        }
      }
    }
  }
}

TEST(CriticTests, PathAlignDegenerateOriginRemainsFinite)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("degenerate_alignment");
  node->declare_parameter("critic.trajectory_point_step", 4);
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "degenerate_costmap", "", "degenerate_costmap", true);
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  ParametersHandler parameters(node);
  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &parameters);
  models::State state;
  state.reset(1, 41);
  models::Trajectories trajectories;
  trajectories.reset(1, 41);
  trajectories.x.fill(0.25f);
  models::Path path;
  path.reset(3);  // All edges <= 1e-6f; projection must not divide by zero.
  path.x(2) = 5e-7f;
  path.yaws(2) = 1.0f;
  geometry_msgs::msg::Pose goal;
  goal.position.x = 2.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({1});
  float dt = 0.05;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.furthest_reached_path_point = 2u;
  for (bool valid : {true, false}) {
    data.path_pts_valid = std::vector<bool>(3, valid);
    costs.fill(0.0f);
    critic.score(data);
    EXPECT_TRUE(std::isfinite(costs(0)));
    EXPECT_NEAR(costs(0), valid ? 2.5f : 0.0f, 1e-4f);
  }
}

TEST(CriticTests, TerminalApproachCriticDefaultOffAndGating)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("terminal_approach_unit");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "terminal_costmap", "", "terminal_costmap", true);
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto * costmap = costmap_ros->getCostmap();
  costmap->resizeMap(400, 400, 0.01, -2.0, -2.0);
  std::fill_n(
    costmap->getCharMap(),
    costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    nav2_costmap_2d::FREE_SPACE);
  ParametersHandler parameters(node);
  parameters.start();

  TerminalApproachCritic terminal_critic;
  EXPECT_FALSE(terminal_critic.defaultEnabled());
  terminal_critic.on_configure(
    node, "mppi", "mppi.TerminalApproachCritic", costmap_ros, &parameters);
  terminal_critic.on_activate();

  bool enabled_param = true;
  ASSERT_TRUE(node->get_parameter("mppi.TerminalApproachCritic.enabled", enabled_param));
  EXPECT_FALSE(enabled_param);

  std::string policy_param;
  ASSERT_TRUE(
    node->get_parameter("mppi.TerminalApproachCritic.infeasible_policy", policy_param));
  EXPECT_EQ(policy_param, "abort");
  double approach_r_param = 0.0;
  double cutoff_r_param = 0.0;
  ASSERT_TRUE(
    node->get_parameter("mppi.TerminalApproachCritic.approach_radius", approach_r_param));
  ASSERT_TRUE(
    node->get_parameter("mppi.TerminalApproachCritic.cutoff_radius", cutoff_r_param));
  EXPECT_NEAR(approach_r_param, 0.65, 1e-5);
  EXPECT_NEAR(cutoff_r_param, 0.35, 1e-5);
  EXPECT_EQ(node->count_publishers("mppi/terminal_replan_requested"), 0u);

  node->declare_parameter("mppi.PathAlignCritic.threshold_to_consider", 0.35);
  node->declare_parameter("mppi.PathFollowCritic.threshold_to_consider", 0.35);
  PathAlignCritic align_critic;
  align_critic.on_configure(node, "mppi", "mppi.PathAlignCritic", costmap_ros, &parameters);
  PathFollowCritic follow_critic;
  follow_critic.on_configure(node, "mppi", "mppi.PathFollowCritic", costmap_ros, &parameters);

  // Place robot inside the 0.35 m < d_goal <= 0.65 m overlap ring at (-0.45, -0.06, 0.0)
  // (d_goal = 0.454 m) with a 6 cm lateral offset from a straight path along y = 0 to goal
  // (0.0, 0.0, 0.0), so the enabled_ gate is falsifiably exercised rather than the
  // approach_radius distance check.
  models::State state;
  state.reset(2, 20);
  state.pose.header.frame_id = "odom";
  state.pose.pose.position.x = -0.45;
  state.pose.pose.position.y = -0.06;
  state.pose.pose.orientation.w = 1.0;

  models::Trajectories trajectories;
  trajectories.reset(2, 20);
  for (size_t j = 0; j < 20; ++j) {
    const float x_j = -0.45f + 0.022f * static_cast<float>(j);
    trajectories.x(0, j) = x_j;
    trajectories.y(0, j) = -0.06f * (1.0f - static_cast<float>(j) / 19.0f);
    trajectories.x(1, j) = x_j;
    trajectories.y(1, j) = -0.25f;
  }

  models::Path path;
  path.reset(25);
  for (size_t i = 0; i < 25; ++i) {
    path.x(i) = -0.60f + 0.025f * static_cast<float>(i);
    path.y(i) = 0.0f;
    path.yaws(i) = 0.0f;
  }
  geometry_msgs::msg::Pose goal;
  goal.position.x = 0.0;
  goal.position.y = 0.0;
  goal.orientation.w = 1.0;

  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float dt = 0.05f;
  CriticData data = {state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};

  // 1. Default-off TerminalApproachCritic inside the 0.65 m approach circle must remain
  //    unlatched and add 0 cost
  terminal_critic.score(data);
  EXPECT_FALSE(terminal_critic.isLatched());
  EXPECT_FLOAT_EQ(costs(0), 0.0f);
  EXPECT_FLOAT_EQ(costs(1), 0.0f);

  // 2. Enabling TerminalApproachCritic in the overlap ring (d_goal = 0.454 m > 0.35 m)
  //    latches a corrective curve and scores > 0 while stock critics ALSO remain active
  ASSERT_TRUE(
    node->set_parameters({rclcpp::Parameter("mppi.TerminalApproachCritic.enabled", true)})[0]
    .successful);
  terminal_critic.score(data);
  EXPECT_TRUE(terminal_critic.isLatched());
  EXPECT_FALSE(terminal_critic.isReferencePathLatched());
  EXPECT_GT(costs(0), 0.0f);
  EXPECT_GT(costs(1), costs(0));

  costs.fill(0.0f);
  align_critic.score(data);
  const float overlap_align_cost = costs(1);
  EXPECT_GT(overlap_align_cost, 0.0f);
  follow_critic.score(data);
  EXPECT_GT(costs(1), overlap_align_cost);

  // 3. Inside the stock-critic cutoff (d_goal = 0.25 m <= 0.35 m) on a nominal approach,
  //    PathAlignCritic and PathFollowCritic switch off unconditionally via their own
  //    threshold_to_consider (0.35 m), while TerminalApproachCritic latches the reference
  //    path and owns arrival guidance.
  terminal_critic.reset();
  state.pose.pose.position.x = -0.25;
  state.pose.pose.position.y = 0.0;
  for (size_t j = 0; j < 20; ++j) {
    const float x_j = -0.25f + 0.012f * static_cast<float>(j);
    trajectories.x(0, j) = x_j;
    trajectories.y(0, j) = 0.0f;
    trajectories.x(1, j) = x_j;
    trajectories.y(1, j) = -0.15f;
  }
  data.furthest_reached_path_point.reset();
  data.path_pts_valid.reset();
  costs.fill(0.0f);
  align_critic.score(data);
  follow_critic.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0f);
  EXPECT_FLOAT_EQ(costs(1), 0.0f);

  terminal_critic.score(data);
  EXPECT_TRUE(terminal_critic.isLatched());
  EXPECT_TRUE(terminal_critic.isReferencePathLatched());
  EXPECT_GT(costs(0), 0.0f);
  EXPECT_GT(costs(1), costs(0));

  // 4. Unlatched 1-point path (n_path == 1) inside the stock-critic cutoff (d_goal = 0.20 m):
  //    Stock PathAlignCritic and PathFollowCritic contribute 0, so an unlatched
  //    TerminalApproachCritic must NOT silently return on n_path == 1; it must either
  //    establish valid guidance (latching reference_path or a capture curve) or explicitly
  //    escalate via ENTRY_INFEASIBLE (Review #2 Round 4 Finding 2).
  terminal_critic.reset();
  path.reset(1);
  path.x(0) = 0.0f;
  path.y(0) = 0.0f;
  path.yaws(0) = 0.0f;
  state.pose.pose.position.x = -0.20;
  state.pose.pose.position.y = 0.0;
  for (size_t j = 0; j < 20; ++j) {
    const float x_j = -0.20f + 0.010f * static_cast<float>(j);
    trajectories.x(0, j) = x_j;
    trajectories.y(0, j) = 0.0f;
    trajectories.x(1, j) = x_j;
    trajectories.y(1, j) = -0.15f;
  }
  data.furthest_reached_path_point.reset();
  data.path_pts_valid.reset();
  costs.fill(0.0f);
  align_critic.score(data);
  follow_critic.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0f);
  EXPECT_FLOAT_EQ(costs(1), 0.0f);

  terminal_critic.score(data);
  EXPECT_TRUE(terminal_critic.isLatched());
  EXPECT_TRUE(terminal_critic.isReferencePathLatched());
  ASSERT_GT(terminal_critic.latchedCurve().size(), 4u);
  EXPECT_NEAR(terminal_critic.latchedCurve().front().x, -0.20f, 1e-3f);
  EXPECT_GT(costs(0), 0.0f);
  EXPECT_GT(costs(1), costs(0));

  // Tick 1 on the latched 1-point reference path: verifyLatchedCurveOrRedraw() must validate
  // minDistanceToCurve() without throwing DRIFT_EXCEEDED (falsifies reverting min_lead_distance).
  data.reuse_obstacles_cache = false;
  costs.fill(0.0f);
  EXPECT_NO_THROW(terminal_critic.score(data));
  EXPECT_TRUE(terminal_critic.isLatched());
  EXPECT_GT(costs(0), 0.0f);
  EXPECT_GT(costs(1), costs(0));

  // 4b. Unlatched 1-point path at d_goal = 0.20 m with an infeasible lateral offset (-0.18 m)
  //     must explicitly escalate via InvalidPath (ENTRY_INFEASIBLE) rather than silently returning.
  terminal_critic.reset();
  state.pose.pose.position.x = -0.20;
  state.pose.pose.position.y = -0.18;
  costs.fill(0.0f);
  try {
    terminal_critic.score(data);
    FAIL() << "Expected InvalidPath (ENTRY_INFEASIBLE) on infeasible unlatched 1-point path";
  } catch (const nav2_core::InvalidPath & ex) {
    EXPECT_NE(std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos)
      << ex.what();
  }
  EXPECT_FALSE(terminal_critic.isLatched());

  // 4c. Laterally-displaced short S-curve (start and goal share heading 0.0, start at
  //     (-0.16, -0.06, 0.0), goal at (0.0, 0.0, 0.0), span = 0.1709 m < d_goal = 0.1897 m)
  //     with a lethal obstacle on the straight goal axis at (-0.28, 0.0) (10 cm behind a fake
  //     lead-in at (-0.18, 0.0) but > 12 cm from all S-curve poses):
  //     buildReferencePathCurve() must check positional collinearity with the goal axis rather
  //     than only start/end heading equality, so it does NOT prepend a fake goal-axis lead-in
  //     or create a 6 cm lateral jump to (-0.16, -0.06) (Review #2 & #3 Round 5).
  terminal_critic.reset();
  unsigned int ox = 0u, oy = 0u;
  ASSERT_TRUE(costmap->worldToMap(-0.28, 0.0, ox, oy));
  costmap->setCost(ox, oy, nav2_costmap_2d::LETHAL_OBSTACLE);

  path.reset(5);
  const float s_xs[5] = {-0.16f, -0.12f, -0.08f, -0.04f, 0.0f};
  const float s_ys[5] = {-0.06f, -0.05f, -0.03f, -0.01f, 0.0f};
  const float s_yaws[5] = {0.0f, 0.18f, 0.25f, 0.12f, 0.0f};
  for (size_t i = 0; i < 5u; ++i) {
    path.x(i) = s_xs[i];
    path.y(i) = s_ys[i];
    path.yaws(i) = s_yaws[i];
  }
  state.pose.pose.position.x = -0.18;
  state.pose.pose.position.y = -0.06;
  costs.fill(0.0f);
  EXPECT_NO_THROW(terminal_critic.score(data));
  EXPECT_TRUE(terminal_critic.isLatched());
  EXPECT_TRUE(terminal_critic.isReferencePathLatched());
  ASSERT_GE(terminal_critic.latchedCurve().size(), 5u);
  EXPECT_NEAR(terminal_critic.latchedCurve().front().x, -0.16f, 1e-4f);
  EXPECT_NEAR(terminal_critic.latchedCurve().front().y, -0.06f, 1e-4f);
  for (size_t i = 0; i + 1u < terminal_critic.latchedCurve().size(); ++i) {
    const double step_d = std::hypot(
      terminal_critic.latchedCurve()[i + 1u].x - terminal_critic.latchedCurve()[i].x,
      terminal_critic.latchedCurve()[i + 1u].y - terminal_critic.latchedCurve()[i].y);
    EXPECT_LE(step_d, 0.03) << "Unexpected jump segment at index " << i;
  }
  costmap->setCost(ox, oy, nav2_costmap_2d::FREE_SPACE);
}
