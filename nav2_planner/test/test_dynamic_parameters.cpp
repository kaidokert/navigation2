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
  double waypointStartMaxDist() const {return through_poses_waypoint_start_max_dist_;}
  double waypointStartMaxYaw() const {return through_poses_waypoint_start_max_yaw_;}
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
  EXPECT_DOUBLE_EQ(
    planner->get_parameter("through_poses_waypoint_start_max_dist").as_double(), 0.01);
  results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("through_poses_waypoint_start_max_dist", 0.03),
      rclcpp::Parameter("through_poses_waypoint_start_max_yaw", 0.04)});
  rclcpp::spin_until_future_complete(planner->get_node_base_interface(), results);
  EXPECT_DOUBLE_EQ(planner->waypointStartMaxDist(), 0.03);
  EXPECT_DOUBLE_EQ(planner->waypointStartMaxYaw(), 0.04);
}

namespace
{
geometry_msgs::msg::PoseStamped makePose(
  const std::string & frame, double x, double y, double qz, double qw)
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
  // The previous leg's goal (global frame) and paths that end on it within float noise, short of it
  // by 5 cm, and on it but turned by 0.1 rad.
  const auto start = makePose("odom", 0.0, 0.0, 0.0, 1.0);
  const auto prev_goal = makePose("map", 1.0, 2.0, -0.7934, 0.6088);
  auto path_ending_at = [](double x, double y, double qz, double qw) {
      nav_msgs::msg::Path p;
      p.header.frame_id = "map";
      p.poses.push_back(makePose("", 0.5, 0.5, 0.0, 1.0));
      p.poses.push_back(makePose("", x, y, qz, qw));
      return p;
    };
  const auto on_it = path_ending_at(1.0001, 2.0, -0.79335, 0.60886);
  const auto short_of_it = path_ending_at(1.05, 2.0, -0.7934, 0.6088);
  const auto turned = path_ending_at(1.0, 2.0, -0.8227, 0.5685);

  for (bool flag : {false, true}) {
    const auto leg0 = PlannerShim::selectThroughPosesLegStart(
      0, start, prev_goal, on_it, flag, 0.01, 0.01);
    EXPECT_EQ(leg0.header.frame_id, "odom");
  }
  // Stock: the path end
  auto s0 = PlannerShim::selectThroughPosesLegStart(1, start, prev_goal, on_it, false, 0.01, 0.01);
  EXPECT_DOUBLE_EQ(s0.pose.position.x, 1.0001);
  EXPECT_DOUBLE_EQ(s0.pose.orientation.z, -0.79335);
  // Flag on, path ended on the goal: the requested goal itself, in the path's frame
  auto s1 = PlannerShim::selectThroughPosesLegStart(1, start, prev_goal, on_it, true, 0.01, 0.01);
  EXPECT_EQ(s1.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(s1.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(s1.pose.orientation.z, -0.7934);
  EXPECT_DOUBLE_EQ(s1.pose.orientation.w, 0.6088);
  // Flag on, path stopped 5 cm short or 0.1 rad off: stock start
  auto s2 = PlannerShim::selectThroughPosesLegStart(
    1, start, prev_goal, short_of_it, true, 0.01, 0.01);
  EXPECT_DOUBLE_EQ(s2.pose.position.x, 1.05);
  auto s3 = PlannerShim::selectThroughPosesLegStart(1, start, prev_goal, turned, true, 0.01, 0.01);
  EXPECT_DOUBLE_EQ(s3.pose.orientation.z, -0.8227);
  // ... unless the tolerances admit it
  auto s4 = PlannerShim::selectThroughPosesLegStart(
    1, start, prev_goal, short_of_it, true, 0.06, 0.01);
  EXPECT_DOUBLE_EQ(s4.pose.position.x, 1.0);
}
