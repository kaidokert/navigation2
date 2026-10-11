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

  // 5. N = 20, lookahead = 1 (normal path startup:
  // verify non-terminal short-path behavior is preserved)
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

TEST(CriticTests, PathAlignCriticTerminalSingletonAlignment)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter("critic.use_path_orientations", true);
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  models::State state;
  state.reset(2, 30);
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  generated_trajectories.reset(2, 30);
  models::Path path;
  path.reset(1);
  path.x(0) = 1.0;
  path.y(0) = 0.0;
  path.yaws(0) = 1.5707963f;

  geometry_msgs::msg::Pose goal;
  goal.position.x = 1.0;
  goal.position.y = 0.0;
  // Goal yaw = pi/2 (1.5707963 rad)
  goal.orientation.z = 0.7071068;
  goal.orientation.w = 0.7071068;

  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();
  TestGoalChecker goal_checker;
  data.goal_checker = &goal_checker;

  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  // 1. Achievable alignment vs Misaligned:
  // Trajectory 0 achieves alignment with goal (yaw = pi/2 across horizon)
  // Trajectory 1 remains misaligned (yaw = 0 across horizon)
  for (unsigned int t = 0; t < 30; ++t) {
    generated_trajectories.yaws(0, t) = 1.5707963f;
    generated_trajectories.yaws(1, t) = 0.0f;
  }
  critic.score(data);

  // Trajectory 0 must have zero heading error cost
  EXPECT_NEAR(costs(0), 0.0f, 1e-4f);
  // Trajectory 1 must have positive heading error cost (~1.57 * weight)
  EXPECT_GT(costs(1), 10.0f);

  // 2. Approach requiring repositioning:
  // Robot state is at (0.85, 0.0) — inside threshold_to_consider (0.25m from goal)
  // Verify heading alignment is actively evaluated inside threshold_to_consider!
  state.pose.pose.position.x = 0.85;
  state.pose.pose.position.y = 0.0;
  costs = xt::zeros<float>({2});
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0f, 1e-4f);
  EXPECT_GT(costs(1), 10.0f);

  // 3. Already-satisfied goal:
  // Robot pose already at goal with correct heading, and staying at goal
  state.pose.pose.position.x = 1.0;
  state.pose.pose.position.y = 0.0;
  state.pose.pose.orientation = goal.orientation;
  generated_trajectories.yaws = 1.5707963f * xt::ones<float>({2, 30});
  costs = xt::zeros<float>({2});
  critic.score(data);
  EXPECT_NEAR(costs(0), 0.0f, 1e-4f);
  EXPECT_NEAR(costs(1), 0.0f, 1e-4f);
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

TEST(CriticTests, ObstaclesCriticFullRolloutCollision)
{
  // Verifies that ObstaclesCritic unconditionally evaluates every sampled pose
  // across the full rollout horizon. Post-goal obstacles are NOT bypassed by
  // goal proximity: a trajectory penetrating into a lethal obstacle past the goal
  // is reliably flagged in collision, while a trajectory that cleanly arrests at
  // the goal remains collision-free.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  auto getParam = param_handler.getParamGetter("critic");
  bool consider_footprint;
  getParam(consider_footprint, "consider_footprint", false);

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
  generated_trajectories.reset(2, 30);

  // Trajectory 0: Decelerates and stops cleanly at the goal (x = 1.5m, never reaches wall at 2.0m)
  for (unsigned int t = 0; t < 30; ++t) {
    double progress = std::min(1.5, 1.0 + 0.05 * static_cast<double>(t));
    generated_trajectories.x(0, t) = progress;
    generated_trajectories.y(0, t) = 0.0;
  }

  // Trajectory 1: Penetrates through the goal into the lethal wall at x = 2.0m
  // (progresses to 2.45m)
  for (unsigned int t = 0; t < 30; ++t) {
    generated_trajectories.x(1, t) = 1.0 + 0.05 * static_cast<double>(t);
    generated_trajectories.y(1, t) = 0.0;
  }

  models::Path path;
  path.reset(10);
  geometry_msgs::msg::Pose goal;
  goal.position.x = 1.5;
  goal.orientation.w = 1.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  ObstaclesCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  critic.score(data);

  // Trajectory 0 (stopping cleanly at goal before wall) must NOT collide
  EXPECT_NEAR(costs(0), 0.0f, 1e-4f) <<
    "trajectory cleanly stopping at goal falsely flagged in collision";

  // Trajectory 1 (penetrating into lethal wall) MUST be flagged in collision
  EXPECT_GT(costs(1), 1e6f) <<
    "trajectory penetrating into post-goal lethal obstacle failed to trigger collision cost";
  EXPECT_FALSE(data.fail_flag) <<
    "fail_flag should be false when at least one trajectory is collision-free";

  // When ALL trajectories collide, fail_flag MUST be set to true
  for (unsigned int t = 0; t < 30; ++t) {
    generated_trajectories.x(0, t) = generated_trajectories.x(1, t);
  }
  costs = xt::zeros<float>({2});
  data.fail_flag = false;
  critic.score(data);
  EXPECT_TRUE(data.fail_flag) <<
    "fail_flag was not set when all trajectories collide with obstacle";
}

class FootprintThresholdWitness : public ObstaclesCritic
{
public:
  float threshold() const {return possible_collision_cost_;}
};

TEST(CriticTests, HardFootprintVetoBelowInflationThreshold)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("raster_footprint");
  node->declare_parameter("critic.consider_footprint", false);
  node->declare_parameter("critic.collision_check_footprint_only", true);
  node->declare_parameter("critic.repulsion_weight", 0.0);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("hard_footprint");
  node->declare_parameter("critic.consider_footprint", false);
  node->declare_parameter("critic.collision_check_footprint_only", true);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
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
  for (size_t t = 0; t < 10; ++t) {
    trajectories.x(0, t) = 1.0;
                                                              }
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
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", false));
  ObstaclesCritic point;
  point.on_configure(node, "mppi", "critic", cm, &handler);
  costs.fill(0.0); data.fail_flag = false; point.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0);  // Reproduces the unsafe center-only blind spot.
  cm->on_cleanup(lifecycle);
}

TEST(CriticTests, PathAlignCriticIntermediateCuspAlignment)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test_node_intermediate_cusp");
  ParametersHandler param_handler(node);

  node->declare_parameter("critic.threshold_to_consider", 1.0f);
  node->declare_parameter("critic.use_path_orientations", true);
  node->declare_parameter("critic.offset_from_furthest", 0);
  node->declare_parameter("critic.cusp_terminal_heading_weight", 30.0f);

  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "costmap_for_path_align_intermediate_cusp", "", "costmap_for_path_align_intermediate_cusp",
    true);
  rclcpp_lifecycle::State state_lifecycle;
  costmap_ros->on_configure(state_lifecycle);

  models::State state;
  state.reset(2, 10);
  state.pose.pose.position.x = 0.1;  // Within 1.0m of goal (0,0)
  state.pose.pose.position.y = 0.0;

  models::Trajectories generated_trajectories;
  generated_trajectories.reset(2, 10);
  // Trajectory 0: terminal yaw aligned with path cusp (yaw = 0.0)
  generated_trajectories.x.fill(0.0);
  generated_trajectories.y.fill(0.0);
  generated_trajectories.yaws.fill(0.0);

  // Trajectory 1: terminal yaw misaligned with path cusp (yaw = M_PI_2)
  for (unsigned int t = 0; t < 10; ++t) {
    generated_trajectories.yaws(1, t) = static_cast<float>(M_PI_2);
  }

  models::Path path;
  path.reset(3);
  path.x(0) = 0.0; path.y(0) = 0.0; path.yaws(0) = 0.0;
  path.x(1) = 0.05; path.y(1) = 0.0; path.yaws(1) = 0.0;
  path.x(2) = 0.10; path.y(2) = 0.0; path.yaws(2) = 0.0;

  geometry_msgs::msg::Pose goal;
  goal.position.x = 0.0;  // Robot is within threshold_to_consider (1.0m) of this goal
  goal.position.y = 0.0;

  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.05f;

  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  data.motion_model = std::make_shared<DiffDriveMotionModel>();

  PathAlignCritic critic;
  critic.on_configure(node, "mppi", "critic", costmap_ros, &param_handler);

  // Case 1: active_cusp_direction == 0 (final goal approach)
  // Inside threshold_to_consider, PathAlignCritic skips scoring so goal critic can take over.
  data.active_cusp_direction = 0;
  critic.score(data);
  EXPECT_FLOAT_EQ(costs(0), 0.0f);
  EXPECT_FLOAT_EQ(costs(1), 0.0f);

  // Case 2: active_cusp_direction == 1 (intermediate forward cusp segment)
  // Even though within threshold_to_consider of the final goal, critic MUST NOT skip!
  // And cusp_terminal_heading_weight penalizes yaw error on trajectory 1.
  costs.fill(0.0f);
  data.active_cusp_direction = 1;
  data.real_path_point_count = 3;
  critic.score(data);
  EXPECT_GT(costs(1), costs(0));
  EXPECT_GT(costs(1), 20.0f);  // 30.0 * M_PI_2 ~= 47.1

  costmap_ros->on_cleanup(state_lifecycle);
}
