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
#include <algorithm>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
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

TEST(SmacTest, analytic_expansion_trace_is_buffered_complete_and_noninvasive)
{
  using json = nlohmann::json;
  auto traced_node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacTraceTest");
  auto baseline_node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("SmacTraceBaselineTest");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("trace_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  costmap_ros->getCostmap()->resizeMap(200, 200, 0.05, -5.0, -5.0);

  traced_node->declare_parameter("test.publish_analytic_expansion_trace", true);
  traced_node->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  traced_node->declare_parameter("test.minimum_turning_radius", 0.5);
  traced_node->declare_parameter("test.smooth_path", false);
  traced_node->declare_parameter("test.analytic_expansion_max_length", 3.0);
  traced_node->declare_parameter("test.analytic_expansion_ratio", 2.0);
  traced_node->declare_parameter("test.lookup_table_size", 10.0);
  baseline_node->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  baseline_node->declare_parameter("test.minimum_turning_radius", 0.5);
  baseline_node->declare_parameter("test.smooth_path", false);
  baseline_node->declare_parameter("test.analytic_expansion_max_length", 3.0);
  baseline_node->declare_parameter("test.analytic_expansion_ratio", 2.0);
  baseline_node->declare_parameter("test.lookup_table_size", 10.0);

  auto traced_planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  auto baseline_planner = std::make_unique<nav2_smac_planner::SmacPlannerHybrid>();
  traced_planner->configure(traced_node, "test", nullptr, costmap_ros);
  baseline_planner->configure(baseline_node, "test", nullptr, costmap_ros);
  traced_planner->activate();
  baseline_planner->activate();

  auto subscriber_node = std::make_shared<rclcpp::Node>("smac_trace_test_subscriber");
  std::vector<std::string> batches;
  auto subscription = subscriber_node->create_subscription<std_msgs::msg::String>(
    "/analytic_expansion_trace", rclcpp::QoS(10),
    [&batches](std_msgs::msg::String::ConstSharedPtr message) {
      batches.push_back(message->data);
    });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(subscriber_node);
  for (int i = 0; i < 20; ++i) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto wait_for_batch = [&executor, &batches](size_t expected_count) {
      for (int i = 0; i < 100 && batches.size() < expected_count; ++i) {
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      return batches.size() >= expected_count;
    };
  auto make_pose = [](double x, double y, double yaw) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.pose.position.x = x;
      pose.pose.position.y = y;
      pose.pose.orientation.z = std::sin(yaw / 2.0);
      pose.pose.orientation.w = std::cos(yaw / 2.0);
      return pose;
    };
  const auto start = make_pose(-1.188, 0.843, 252.0 * M_PI / 180.0);
  const auto goal = make_pose(-1.18, 0.90, 270.0 * M_PI / 180.0);
  auto no_cancel = []() {return false;};

  const auto traced_plan = traced_planner->createPlan(start, goal, no_cancel);
  ASSERT_TRUE(wait_for_batch(1));
  const auto baseline_plan = baseline_planner->createPlan(start, goal, no_cancel);
  ASSERT_EQ(traced_plan.poses.size(), baseline_plan.poses.size());
  for (size_t i = 0; i < traced_plan.poses.size(); ++i) {
    EXPECT_DOUBLE_EQ(
      traced_plan.poses[i].pose.position.x, baseline_plan.poses[i].pose.position.x);
    EXPECT_DOUBLE_EQ(
      traced_plan.poses[i].pose.position.y, baseline_plan.poses[i].pose.position.y);
    EXPECT_DOUBLE_EQ(
      traced_plan.poses[i].pose.orientation.z, baseline_plan.poses[i].pose.orientation.z);
    EXPECT_DOUBLE_EQ(
      traced_plan.poses[i].pose.orientation.w, baseline_plan.poses[i].pose.orientation.w);
  }

  const json records = json::parse(batches[0]);
  ASSERT_TRUE(records.is_array());
  ASSERT_GE(records.size(), 3u);
  EXPECT_EQ(records.front()["event"], "plan_boundary");
  EXPECT_EQ(records.front()["result"], "started");
  EXPECT_EQ(records.back()["event"], "plan_boundary");
  EXPECT_EQ(records.back()["result"], "succeeded");
  const auto request_id = records.front()["request_id"];
  size_t candidate_count = 0;
  size_t selection_count = 0;
  std::vector<uint64_t> candidate_ids;
  std::vector<uint64_t> selection_ids;
  for (const auto & record : records) {
    EXPECT_EQ(record["request_id"], request_id);
    if (record["event"] == "candidate") {
      ++candidate_count;
      candidate_ids.push_back(record["attempt_id"].get<uint64_t>());
    } else if (record["event"] == "selection") {
      ++selection_count;
      selection_ids.push_back(record["attempt_id"].get<uint64_t>());
    }
  }
  EXPECT_EQ(candidate_count, records.back()["candidate_attempts"].get<size_t>());
  ASSERT_EQ(selection_count, 1u);
  EXPECT_NE(
    std::find(candidate_ids.begin(), candidate_ids.end(), selection_ids.front()),
    candidate_ids.end());

  auto same_cell_start = make_pose(0.0, 0.0, 0.0);
  auto same_cell_goal = make_pose(0.01, 0.01, 0.0);
  EXPECT_EQ(
    traced_planner->createPlan(same_cell_start, same_cell_goal, no_cancel).poses.size(),
    1u);
  ASSERT_TRUE(wait_for_batch(2));
  const json same_cell_records = json::parse(batches[1]);
  EXPECT_EQ(same_cell_records.front()["result"], "started");
  EXPECT_EQ(same_cell_records.back()["result"], "succeeded");
  EXPECT_EQ(same_cell_records.back()["candidate_attempts"], 0u);

  EXPECT_FALSE(traced_node->set_parameter(
      rclcpp::Parameter("test.publish_analytic_expansion_trace", false)).successful);
  EXPECT_TRUE(traced_node->get_parameter("test.publish_analytic_expansion_trace").as_bool());

  EXPECT_THROW(
    traced_planner->createPlan(start, make_pose(20.0, 20.0, 0.0), no_cancel),
    nav2_core::GoalOutsideMapBounds);
  ASSERT_TRUE(wait_for_batch(3));
  const json out_of_map_records = json::parse(batches[2]);
  EXPECT_EQ(out_of_map_records.front()["result"], "started");
  EXPECT_EQ(out_of_map_records.back()["result"], "exception");
  EXPECT_EQ(out_of_map_records.back()["failure_reason"], "goal_outside_map");
  EXPECT_EQ(out_of_map_records.back()["candidate_attempts"], 0u);

  EXPECT_TRUE(traced_node->set_parameter(
      rclcpp::Parameter("test.tolerance", 0.0)).successful);
  unsigned int goal_x;
  unsigned int goal_y;
  ASSERT_TRUE(costmap_ros->getCostmap()->worldToMap(-1.18, 0.90, goal_x, goal_y));
  costmap_ros->getCostmap()->setCost(
    goal_x, goal_y, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_THROW(
    traced_planner->createPlan(start, goal, no_cancel), nav2_core::GoalOccupied);
  ASSERT_TRUE(wait_for_batch(4));
  const json failed_records = json::parse(batches[3]);
  EXPECT_EQ(failed_records.front()["result"], "started");
  EXPECT_EQ(failed_records.back()["result"], "failed");
  EXPECT_EQ(failed_records.back()["failure_reason"], "goal_occupied");

  traced_planner->deactivate();
  baseline_planner->deactivate();
  traced_planner->cleanup();
  baseline_planner->cleanup();
  traced_planner.reset();
  baseline_planner.reset();
  executor.remove_node(subscriber_node->get_node_base_interface());
  subscription.reset();
  subscriber_node.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
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

  // Case 2: Mutate size_x/size_y while resolution matches original (A* adapts, does NOT throw)
  costmap_ros->getCostmap()->resizeMap(200, 150, 0.05, 0.0, 0.0);
  EXPECT_NO_THROW({
    planner->createPlan(start, goal, dummy_cancel_checker);
  });

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
  costmap_ros.reset();
  nodeSE2.reset();
}
