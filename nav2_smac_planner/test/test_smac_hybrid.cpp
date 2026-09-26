// Copyright (c) 2020, Samsung Research America
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
// limitations under the License. Reserved.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/utils.h"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"
#include "nav2_smac_planner/smac_planner_2d.hpp"
#include "nav2_core/planner_exceptions.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

// SMAC smoke tests for plugin-level issues rather than algorithms
// (covered by more extensively testing in other files)
// System tests in nav2_system_tests will actually plan with this work

TEST(SmacTest, test_smac_se2)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeSE2 =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacSE2Test");
  nodeSE2->declare_parameter("test.debug_visualizations", rclcpp::ParameterValue(true));

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  nodeSE2->declare_parameter("test.downsample_costmap", true);
  nodeSE2->set_parameter(rclcpp::Parameter("test.downsample_costmap", true));
  nodeSE2->declare_parameter("test.downsampling_factor", 2);
  nodeSE2->set_parameter(rclcpp::Parameter("test.downsampling_factor", 2));

  auto dummy_cancel_checker = []() {
      return false;
    };

  geometry_msgs::msg::PoseStamped start, goal;
  start.pose.position.x = 0.0;
  start.pose.position.y = 0.0;
  start.pose.orientation.w = 1.0;
  goal.pose.position.x = 1.0;
  goal.pose.position.y = 1.0;
  goal.pose.orientation.w = 1.0;
  auto planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  planner->configure(nodeSE2, "test", nullptr, costmap_ros);
  planner->activate();

  try {
    planner->createPlan(start, goal, dummy_cancel_checker);
  } catch (...) {
  }

  // corner case where the start and goal are on the same cell
  goal.pose.position.x = 0.01;
  goal.pose.position.y = 0.01;

  nav_msgs::msg::Path plan = planner->createPlan(start, goal, dummy_cancel_checker);
  EXPECT_EQ(plan.poses.size(), 1);  // single point path

  planner->deactivate();
  planner->cleanup();

  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
  costmap_ros.reset();
  nodeSE2.reset();
}

TEST(SmacTest, test_smac_se2_reconfigure)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeSE2 =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacSE2Test");

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  auto planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  planner->configure(nodeSE2, "test", nullptr, costmap_ros);
  planner->activate();

  nodeSE2->declare_parameter("resolution", 0.05);

  auto rec_param = std::make_shared<rclcpp::AsyncParametersClient>(
    nodeSE2->get_node_base_interface(), nodeSE2->get_node_topics_interface(),
    nodeSE2->get_node_graph_interface(),
    nodeSE2->get_node_services_interface());

  auto results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.downsample_costmap", true),
      rclcpp::Parameter("test.downsampling_factor", 2),
      rclcpp::Parameter("test.angle_quantization_bins", 100),
      rclcpp::Parameter("test.allow_unknown", false),
      rclcpp::Parameter("test.max_iterations", -1),
      rclcpp::Parameter("test.minimum_turning_radius", 1.0),
      rclcpp::Parameter("test.cache_obstacle_heuristic", true),
      rclcpp::Parameter("test.reverse_penalty", 5.0),
      rclcpp::Parameter("test.change_penalty", 1.0),
      rclcpp::Parameter("test.non_straight_penalty", 2.0),
      rclcpp::Parameter("test.cost_penalty", 2.0),
      rclcpp::Parameter("test.tolerance", 0.2),
      rclcpp::Parameter("test.retrospective_penalty", 0.2),
      rclcpp::Parameter("test.analytic_expansion_ratio", 4.0),
      rclcpp::Parameter("test.max_planning_time", 10.0),
      rclcpp::Parameter("test.lookup_table_size", 30.0),
      rclcpp::Parameter("test.smooth_path", false),
      rclcpp::Parameter("test.analytic_expansion_max_length", 42.0),
      rclcpp::Parameter("test.max_on_approach_iterations", 42),
      rclcpp::Parameter("test.terminal_checking_interval", 42),
      rclcpp::Parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"))});

  rclcpp::spin_until_future_complete(
    nodeSE2->get_node_base_interface(),
    results);

  EXPECT_EQ(nodeSE2->get_parameter("test.downsample_costmap").as_bool(), true);
  EXPECT_EQ(nodeSE2->get_parameter("test.downsampling_factor").as_int(), 2);
  EXPECT_EQ(nodeSE2->get_parameter("test.angle_quantization_bins").as_int(), 100);
  EXPECT_EQ(nodeSE2->get_parameter("test.allow_unknown").as_bool(), false);
  EXPECT_EQ(nodeSE2->get_parameter("test.max_iterations").as_int(), -1);
  EXPECT_EQ(nodeSE2->get_parameter("test.minimum_turning_radius").as_double(), 1.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.cache_obstacle_heuristic").as_bool(), true);
  EXPECT_EQ(nodeSE2->get_parameter("test.reverse_penalty").as_double(), 5.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.change_penalty").as_double(), 1.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.non_straight_penalty").as_double(), 2.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.cost_penalty").as_double(), 2.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.retrospective_penalty").as_double(), 0.2);
  EXPECT_EQ(nodeSE2->get_parameter("test.tolerance").as_double(), 0.2);
  EXPECT_EQ(nodeSE2->get_parameter("test.analytic_expansion_ratio").as_double(), 4.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.smooth_path").as_bool(), false);
  EXPECT_EQ(nodeSE2->get_parameter("test.max_planning_time").as_double(), 10.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.lookup_table_size").as_double(), 30.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.analytic_expansion_max_length").as_double(), 42.0);
  EXPECT_EQ(nodeSE2->get_parameter("test.max_on_approach_iterations").as_int(), 42);
  EXPECT_EQ(nodeSE2->get_parameter("test.terminal_checking_interval").as_int(), 42);
  EXPECT_EQ(
    nodeSE2->get_parameter("test.motion_model_for_search").as_string(),
    std::string("REEDS_SHEPP"));

  auto results2 = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("resolution", 0.2)});
  rclcpp::spin_until_future_complete(
    nodeSE2->get_node_base_interface(),
    results2);
  EXPECT_EQ(nodeSE2->get_parameter("resolution").as_double(), 0.2);
}

TEST(SmacTest, test_smac_m5_terminal_density)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeSE2 =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacM5DensityTest");

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  // Configure an empty costmap covering fixture bounds:
  // resolution 0.05m, 200x200 cells (10m x 10m), origin (-5, -5)
  costmap_ros->getCostmap()->resizeMap(200, 200, 0.05, -5.0, -5.0);

  // Parameters mirroring nav2_params.yaml
  nodeSE2->declare_parameter("test.downsample_costmap", false);
  nodeSE2->set_parameter(rclcpp::Parameter("test.downsample_costmap", false));
  nodeSE2->declare_parameter("test.angle_quantization_bins", 72);
  nodeSE2->set_parameter(rclcpp::Parameter("test.angle_quantization_bins", 72));
  nodeSE2->declare_parameter("test.minimum_turning_radius", 0.5);
  nodeSE2->set_parameter(rclcpp::Parameter("test.minimum_turning_radius", 0.5));
  nodeSE2->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  nodeSE2->set_parameter(
    rclcpp::Parameter("test.motion_model_for_search", std::string("REEDS_SHEPP")));
  nodeSE2->declare_parameter("test.smooth_path", false);
  nodeSE2->set_parameter(rclcpp::Parameter("test.smooth_path", false));
  nodeSE2->declare_parameter("test.analytic_expansion_max_length", 3.0);
  nodeSE2->set_parameter(rclcpp::Parameter("test.analytic_expansion_max_length", 3.0));
  nodeSE2->declare_parameter("test.analytic_expansion_ratio", 2.0);
  nodeSE2->set_parameter(rclcpp::Parameter("test.analytic_expansion_ratio", 2.0));
  nodeSE2->declare_parameter("test.lookup_table_size", 10.0);
  nodeSE2->set_parameter(rclcpp::Parameter("test.lookup_table_size", 10.0));

  auto planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  planner->configure(nodeSE2, "test", nullptr, costmap_ros);
  planner->activate();

  auto dummy_cancel_checker = []() {
      return false;
    };

  // M5 fixture geometry from test_fixtures/smac_m5_terminal_stub.yaml
  // start: (-1.188, 0.843, 252.0 deg)
  // goal:  (-1.18,  0.90,  270.0 deg)
  geometry_msgs::msg::PoseStamped start, goal;
  start.header.frame_id = "map";
  start.pose.position.x = -1.188;
  start.pose.position.y = 0.843;
  double start_yaw = 252.0 * M_PI / 180.0;
  start.pose.orientation.z = std::sin(start_yaw / 2.0);
  start.pose.orientation.w = std::cos(start_yaw / 2.0);

  goal.header.frame_id = "map";
  goal.pose.position.x = -1.18;
  goal.pose.position.y = 0.90;
  double goal_yaw = 270.0 * M_PI / 180.0;
  goal.pose.orientation.z = std::sin(goal_yaw / 2.0);
  goal.pose.orientation.w = std::cos(goal_yaw / 2.0);

  nav_msgs::msg::Path plan;
  EXPECT_NO_THROW(plan = planner->createPlan(start, goal, dummy_cancel_checker));

  // Assert path pose count >= 4
  EXPECT_GE(plan.poses.size(), 4u) << "Plan only produced " << plan.poses.size() << " poses.";

  auto ang_norm = [](double a) {
      return std::atan2(std::sin(a), std::cos(a));
    };

  const double R_min = 0.5;

  for (size_t i = 1; i < plan.poses.size(); ++i) {
    double dx = plan.poses[i].pose.position.x - plan.poses[i - 1].pose.position.x;
    double dy = plan.poses[i].pose.position.y - plan.poses[i - 1].pose.position.y;
    double step = std::hypot(dx, dy);
    double th0 = tf2::getYaw(plan.poses[i - 1].pose.orientation);
    double th1 = tf2::getYaw(plan.poses[i].pose.orientation);
    double dyaw = ang_norm(th1 - th0);
    double r_implied = (std::abs(dyaw) > 1e-9) ? (step / std::abs(dyaw)) : 999.0;
    if (step > 0.001) {
      EXPECT_GE(r_implied, 0.9 * R_min)
        << "Sub-feasible chord at step " << i << ": step=" << step
        << " m, dyaw=" << (dyaw * 180.0 / M_PI) << " deg, r_implied=" << r_implied
        << " m (must be >= " << (0.9 * R_min) << " m)";
    }
  }

  // Verify that direction reversals (cusps) are faithfully preserved
  size_t reversals = 0;
  for (size_t i = 2; i < plan.poses.size(); ++i) {
    double dx1 = plan.poses[i - 1].pose.position.x - plan.poses[i - 2].pose.position.x;
    double dy1 = plan.poses[i - 1].pose.position.y - plan.poses[i - 2].pose.position.y;
    double dx2 = plan.poses[i].pose.position.x - plan.poses[i - 1].pose.position.x;
    double dy2 = plan.poses[i].pose.position.y - plan.poses[i - 1].pose.position.y;
    if (dx1 * dx2 + dy1 * dy2 < 0.0) {
      ++reversals;
    }
  }
  EXPECT_GE(reversals, 1u) << "Plan should contain at least one direction reversal (cusp).";

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
  costmap_ros.reset();
  nodeSE2.reset();
}

TEST(SmacTest, test_smac_costmap_mutation_throws)
{
  rclcpp_lifecycle::LifecycleNode::SharedPtr nodeSE2 =
    std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacMutationTest");

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros =
    std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());

  // Set known initial dimensions and resolution (0.05m/cell, 100x100)
  costmap_ros->getCostmap()->resizeMap(100, 100, 0.05, 0.0, 0.0);

  nodeSE2->declare_parameter("test.downsample_costmap", false);
  nodeSE2->set_parameter(rclcpp::Parameter("test.downsample_costmap", false));
  nodeSE2->declare_parameter("test.angle_quantization_bins", 72);
  nodeSE2->set_parameter(rclcpp::Parameter("test.angle_quantization_bins", 72));
  nodeSE2->declare_parameter("test.minimum_turning_radius", 0.5);
  nodeSE2->set_parameter(rclcpp::Parameter("test.minimum_turning_radius", 0.5));
  nodeSE2->declare_parameter("test.lookup_table_size", 4.0);
  nodeSE2->set_parameter(rclcpp::Parameter("test.lookup_table_size", 4.0));
  nodeSE2->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  nodeSE2->set_parameter(
    rclcpp::Parameter("test.motion_model_for_search", std::string("REEDS_SHEPP")));

  auto planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  planner->configure(nodeSE2, "test", nullptr, costmap_ros);
  planner->activate();

  auto dummy_cancel_checker = []() {
      return false;
    };

  geometry_msgs::msg::PoseStamped start, goal;
  start.header.frame_id = "map";
  start.pose.position.x = 0.5;
  start.pose.position.y = 0.5;
  start.pose.orientation.w = 1.0;
  goal.header.frame_id = "map";
  goal.pose.position.x = 1.0;
  goal.pose.position.y = 1.0;
  goal.pose.orientation.w = 1.0;

  // Case 1: Resolution mutated (simulating StaticLayer receiving map with different resolution)
  costmap_ros->getCostmap()->resizeMap(100, 100, 0.02, 0.0, 0.0);

  try {
    planner->createPlan(start, goal, dummy_cancel_checker);
    FAIL() <<
      "Expected nav2_core::PlannerException when costmap resolution mutated, but nothing thrown.";
  } catch (const nav2_core::PlannerException & ex) {
    std::string msg = ex.what();
    EXPECT_NE(msg.find("resolution"), std::string::npos)
      << "Exception message did not mention 'resolution': " << msg;
  } catch (const std::exception & ex) {
    FAIL() << "Expected nav2_core::PlannerException, but caught other: " << ex.what();
  }

  // Case 2: Mutate size_x/size_y while resolution matches original
  costmap_ros->getCostmap()->resizeMap(200, 150, 0.05, 0.0, 0.0);
  try {
    planner->createPlan(start, goal, dummy_cancel_checker);
    FAIL() <<
      "Expected nav2_core::PlannerException when costmap dimensions mutated, but nothing thrown.";
  } catch (const nav2_core::PlannerException & ex) {
    std::string msg = ex.what();
    EXPECT_TRUE(
      msg.find("dimension") != std::string::npos || msg.find("size") != std::string::npos)
      << "Exception message did not mention dimension or size: " << msg;
  } catch (const std::exception & ex) {
    FAIL() << "Expected nav2_core::PlannerException, but caught different exception: " << ex.what();
  }

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
  costmap_ros.reset();
  nodeSE2.reset();
}
