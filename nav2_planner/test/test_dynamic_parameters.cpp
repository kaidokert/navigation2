// Copyright (c) 2021, Samsung Research America
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

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_planner/planner_server.hpp"
#include "rclcpp/rclcpp.hpp"

class PlannerShim : public nav2_planner::PlannerServer
{
public:
  PlannerShim()
  : nav2_planner::PlannerServer(rclcpp::NodeOptions())
  {
  }

  // Since we cannot call configure/activate due to costmaps
  // requiring TF
  void setDynamicCallback()
  {
    auto node = shared_from_this();
    // Add callback for dynamic parameters
    dyn_params_handler_ = node->add_on_set_parameters_callback(
      std::bind(&PlannerShim::dynamicParamsShim, this, std::placeholders::_1));
  }

  rcl_interfaces::msg::SetParametersResult
  dynamicParamsShim(std::vector<rclcpp::Parameter> parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    dynamicParametersCallback(parameters);
    return result;
  }

  using nav2_planner::PlannerServer::selectThroughPosesLegStart;
  bool startAtRequestedWaypoint() const {return through_poses_start_at_requested_waypoint_;}
};

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

TEST(WPTest, test_dynamic_parameters)
{
  auto planner = std::make_shared<PlannerShim>();
  planner->setDynamicCallback();

  auto rec_param = std::make_shared<rclcpp::AsyncParametersClient>(
    planner->get_node_base_interface(), planner->get_node_topics_interface(),
    planner->get_node_graph_interface(),
    planner->get_node_services_interface());

  auto results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("expected_planner_frequency", 100.0)});

  rclcpp::spin_until_future_complete(
    planner->get_node_base_interface(),
    results);

  EXPECT_EQ(planner->get_parameter("expected_planner_frequency").as_double(), 100.0);

  // test edge case for = 0
  results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("expected_planner_frequency", -1.0)});

  rclcpp::spin_until_future_complete(
    planner->get_node_base_interface(),
    results);

  EXPECT_EQ(planner->get_parameter("expected_planner_frequency").as_double(), -1.0);

  // GENERIC through-poses leg start: default off, switchable at runtime
  EXPECT_FALSE(planner->get_parameter("through_poses_start_at_requested_waypoint").as_bool());
  EXPECT_FALSE(planner->startAtRequestedWaypoint());
  results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("through_poses_start_at_requested_waypoint", true)});
  rclcpp::spin_until_future_complete(planner->get_node_base_interface(), results);
  EXPECT_TRUE(planner->startAtRequestedWaypoint());
}

namespace
{
geometry_msgs::msg::PoseStamped makePose(const std::string & frame, double x, double y, double qz, double qw)
{
  geometry_msgs::msg::PoseStamped p;
  p.header.frame_id = frame;
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.pose.orientation.z = qz;
  p.pose.orientation.w = qw;
  return p;
}
}  // namespace

TEST(WPTest, through_poses_leg_start)
{
  // Request start, two waypoints; the path planned so far ends 1 cm and a hair of heading away from
  // waypoint 0 (a planner stopping within tolerance / floating-point end), in the costmap frame.
  const auto start = makePose("odom", 0.0, 0.0, 0.0, 1.0);
  const std::vector<geometry_msgs::msg::PoseStamped> goals = {
    makePose("map", 1.0, 2.0, -0.7934, 0.6088), makePose("map", 3.0, 4.0, 0.0, 1.0)};
  nav_msgs::msg::Path concat;
  concat.header.frame_id = "costmap";
  concat.poses.push_back(makePose("", 0.5, 0.5, 0.0, 1.0));
  concat.poses.push_back(makePose("", 1.01, 2.0, -0.7933, 0.6089));

  for (bool flag : {false, true}) {
    const auto leg0 = PlannerShim::selectThroughPosesLegStart(0, start, goals, concat, flag);
    EXPECT_EQ(leg0.header.frame_id, "odom");
    EXPECT_DOUBLE_EQ(leg0.pose.position.x, 0.0);
  }
  // Stock: leg 1 starts at the path end, in the path's frame
  const auto stock = PlannerShim::selectThroughPosesLegStart(1, start, goals, concat, false);
  EXPECT_EQ(stock.header.frame_id, "costmap");
  EXPECT_DOUBLE_EQ(stock.pose.position.x, 1.01);
  EXPECT_DOUBLE_EQ(stock.pose.orientation.z, -0.7933);
  // Flag on: leg 1 starts exactly at the requested waypoint 0, in its own frame
  const auto wp = PlannerShim::selectThroughPosesLegStart(1, start, goals, concat, true);
  EXPECT_EQ(wp.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(wp.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(wp.pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(wp.pose.orientation.z, -0.7934);
  EXPECT_DOUBLE_EQ(wp.pose.orientation.w, 0.6088);
}
