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

#include <math.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/start_failure_snapshot.hpp"
#include "nav2_smac_planner/successful_plan_snapshot.hpp"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"
#include "nav2_smac_planner/smac_planner_2d.hpp"

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


TEST(SmacTest, SuccessfulPlanSnapshotIsOptionalAtomicAndBounded)
{
  nav2_costmap_2d::Costmap2D grid(3, 2, 0.02, -1.0, 2.0, 0);
  grid.setCost(2, 1, 254);
  geometry_msgs::msg::PoseStamped start, goal;
  start.pose.orientation.w = 1.0; goal.pose.orientation.w = 1.0;
  nav_msgs::msg::Path plan;
  plan.header.stamp.sec = 42;
  plan.header.frame_id = "map";
  start.pose.position.x = 1.25;
  plan.poses.push_back(start);
  std::vector<geometry_msgs::msg::Point> footprint;
  geometry_msgs::msg::Point p; p.x = 0.1; p.y = 0.2; footprint.push_back(p);
  EXPECT_FALSE(nav2_smac_planner::writeSuccessfulPlanSnapshot(
    nullptr, 0, 42000000000, grid, plan, start, goal, footprint, 72, false, false));
  const char * prefix = "/tmp/office_smac_success_snapshot_unit";
  const std::string file = std::string(prefix) + ".plan_0.json";
  const std::string temp = file + ".tmp";
  std::remove(file.c_str()); std::remove(temp.c_str());
  for (size_t sequence : {0u, 64u}) {
    ASSERT_TRUE(nav2_smac_planner::writeSuccessfulPlanSnapshot(
      prefix, sequence, 42000000000, grid, plan, start, goal, footprint, 72, false, false));
    std::ifstream input(file);
    const std::string text((std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"sequence\":" + std::to_string(sequence)), std::string::npos);
    EXPECT_NE(text.find("\"plan_world_xy_quaternion\":[[1.25,0,0,0,0,1]]"), std::string::npos);
    EXPECT_NE(text.find("\"data\":[0,0,0,0,0,254]"), std::string::npos);
    EXPECT_NE(text.find("\"frame\":\"map\""), std::string::npos);
    EXPECT_FALSE(std::ifstream(temp).good());
  }
  EXPECT_EQ(grid.getCost(2, 1), 254);
  EXPECT_DOUBLE_EQ(plan.poses[0].pose.position.x, 1.25);
  std::remove(file.c_str());
}

TEST(SmacTest, StartFailureSnapshotIsOptionalAndPreservesGrid)
{
  nav2_costmap_2d::Costmap2D grid(3, 2, 0.02, -1.0, 2.0, 0);
  grid.setCost(2, 1, 254);
  geometry_msgs::msg::PoseStamped start, goal;
  start.pose.orientation.w = 1.0; goal.pose.orientation.w = 1.0;
  std::vector<geometry_msgs::msg::Point> footprint;
  geometry_msgs::msg::Point p; p.x = 0.1; p.y = 0.2; footprint.push_back(p);
  nav2_smac_planner::SearchInfo search;
  EXPECT_FALSE(nav2_smac_planner::writeStartFailureSnapshot(
    nullptr, grid, start, goal, 1.25, 0.5, 4, 72, false, false, footprint, search));
  const char * file = "/tmp/office_smac_start_snapshot_unit.json";
  std::remove(file);
  EXPECT_TRUE(nav2_smac_planner::writeStartFailureSnapshot(
    file, grid, start, goal, 1.25, 0.5, 4, 72, false, false, footprint, search));
  std::ifstream input(file);
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  EXPECT_NE(text.find("\"start_native_collision\":false"), std::string::npos);
  EXPECT_NE(text.find("\"search_iterations\":1"), std::string::npos);
  EXPECT_NE(text.find("\"data\":[0,0,0,0,0,254]"), std::string::npos);
  EXPECT_EQ(grid.getCost(2, 1), 254);
  EXPECT_EQ(grid.getCost(0, 0), 0);
  std::remove(file);
}

TEST(SmacTest, ExhaustedSearchSnapshotRecordsActualIterationCount)
{
  nav2_costmap_2d::Costmap2D grid(2, 2, 0.02, 0.0, 0.0, 0);
  grid.setCost(1, 0, 253);
  geometry_msgs::msg::PoseStamped start, goal;
  start.pose.orientation.w = 1.0; goal.pose.orientation.w = 1.0;
  nav2_smac_planner::SearchInfo search;
  const char * file = "/tmp/office_smac_no_path_snapshot_unit.json";
  EXPECT_TRUE(nav2_smac_planner::writeStartFailureSnapshot(
    file, grid, start, goal, 0.5, 0.5, 0, 72, false, false, {}, search, 37, 123456789));
  std::ifstream input(file);
  std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  EXPECT_NE(text.find("\"search_iterations\":37"), std::string::npos);
  EXPECT_NE(text.find("\"capture_stamp_ns\":123456789"), std::string::npos);
  EXPECT_NE(text.find("\"data\":[0,253,0,0]"), std::string::npos);
  EXPECT_EQ(grid.getCost(1, 0), 253);
  std::remove(file);
}
