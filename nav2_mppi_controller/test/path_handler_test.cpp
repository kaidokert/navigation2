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

#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"
#include "settled_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "angles/angles.h"
#include "nav2_mppi_controller/tools/path_handler.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "tf2_ros/transform_broadcaster.h"

// Tests path handling

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

using namespace mppi;  // NOLINT

class PathHandlerWrapper : public PathHandler
{
public:
  PathHandlerWrapper()
  : PathHandler() {}

  void pruneGlobalPlanWrapper(nav_msgs::msg::Path & path, const PathIterator end)
  {
    return prunePlan(path, end);
  }

  double getMaxCostmapDistWrapper()
  {
    return getMaxCostmapDist();
  }

  std::pair<nav_msgs::msg::Path, PathIterator>
  getGlobalPlanConsideringBoundsInCostmapFrameWrapper(const geometry_msgs::msg::PoseStamped & pose)
  {
    return getGlobalPlanConsideringBoundsInCostmapFrame(pose);
  }

  bool transformPoseWrapper(
    const std::string & frame, const geometry_msgs::msg::PoseStamped & in_pose,
    geometry_msgs::msg::PoseStamped & out_pose) const
  {
    return transformPose(frame, in_pose, out_pose);
  }

  geometry_msgs::msg::PoseStamped transformToGlobalPlanFrameWrapper(
    const geometry_msgs::msg::PoseStamped & pose)
  {
    return transformToGlobalPlanFrame(pose);
  }

  void setGlobalPlanUpToInversion(const nav_msgs::msg::Path & path)
  {
    global_plan_up_to_inversion_ = path;
  }

  bool isWithinInversionTolerancesWrapper(const geometry_msgs::msg::PoseStamped & robot_pose)
  {
    return isWithinInversionTolerances(robot_pose);
  }

  nav_msgs::msg::Path & getInvertedPath()
  {
    return global_plan_up_to_inversion_;
  }

  unsigned int getInversionLocale() const
  {
    return inversion_locale_;
  }

  bool estimateGhostHeadingWrapper(
    const nav_msgs::msg::Path & transformed_plan,
    double & heading,
    double & heading_variance,
    double & seed_arc_length) const
  {
    return estimateGhostHeading(transformed_plan, heading, heading_variance, seed_arc_length);
  }

  void setGlobalPlan(const nav_msgs::msg::Path & path)
  {
    global_plan_ = path;
  }

  size_t appendGhostPathWrapper(
    nav_msgs::msg::Path & path, double heading, double target, double & length) const
  {
    return appendGhostPath(path, heading, target, length);
  }

  void setInversionLocale(unsigned int locale)
  {
    inversion_locale_ = locale;
  }

  void setEnforcePathInversion(bool enforce)
  {
    enforce_path_inversion_ = enforce;
  }

  void setInversionXYTolerance(double tolerance)
  {
    inversion_xy_tolerance_ = tolerance;
  }

  bool computeEscapeHeadingWrapper(double & heading) const
  {
    return computeEscapeHeading(heading);
  }

  void setHandoffEscapeAfterCycles(int cycles)
  {
    handoff_escape_after_cycles_ = cycles;
  }
};

// Helper: robot pose at (x, y) with the given yaw
geometry_msgs::msg::PoseStamped makePose(double x, double y, double yaw)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.orientation.z = sin(yaw / 2.0);
  pose.pose.orientation.w = cos(yaw / 2.0);
  return pose;
}

nav_msgs::msg::Path makeMicroCuspPath(const std::string & frame_id = "map")
{
  nav_msgs::msg::Path path;
  path.header.frame_id = frame_id;

  const std::vector<double> xs = {
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.34, 0.335, 0.33, 0.38, 0.45};
  for (const double x : xs) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = frame_id;
    pose.pose.position.x = x;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
  }

  return path;
}

std::shared_ptr<nav2_costmap_2d::Costmap2DROS> makeConfiguredCostmapRos()
{
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  // These path-only fixtures use grid metadata and TF, not sensor/map plugins.
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->set_parameters_atomically(
    {rclcpp::Parameter("global_frame", "odom"),
      rclcpp::Parameter("robot_base_frame", "base_link")});
  costmap_ros->on_configure(rclcpp_lifecycle::State{});
  return costmap_ros;
}

void publishIdentityTf(
  const std::shared_ptr<rclcpp_lifecycle::LifecycleNode> & node,
  const std::shared_ptr<tf2_ros::Buffer> & tf_buffer)
{
  auto tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(node);
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = "map";
  t.child_frame_id = "base_link";
  t.transform.rotation.w = 1.0;
  tf_broadcaster->sendTransform(t);
  t.child_frame_id = "odom";
  tf_broadcaster->sendTransform(t);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (std::chrono::steady_clock::now() < deadline) {
    if (tf_buffer->canTransform("map", "base_link", tf2::TimePointZero) &&
      tf_buffer->canTransform("map", "odom", tf2::TimePointZero))
    {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL() << "Timed out waiting for identity TFs";
}

TEST(PathHandlerTests, GetAndPrunePath)
{
  nav_msgs::msg::Path path;
  PathHandlerWrapper handler;

  path.header.frame_id = "fkframe";
  path.poses.resize(11);

  handler.setPath(path);
  auto & rtn_path = handler.getPath();
  EXPECT_EQ(path.header.frame_id, rtn_path.header.frame_id);
  EXPECT_EQ(path.poses.size(), rtn_path.poses.size());

  PathIterator it = rtn_path.poses.begin() + 5;
  handler.pruneGlobalPlanWrapper(rtn_path, it);
  auto rtn2_path = handler.getPath();
  EXPECT_EQ(rtn2_path.poses.size(), 6u);
}

TEST(PathHandlerTests, TestBounds)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  // These path-only fixtures use grid metadata and TF, not sensor/map plugins.
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  auto results = costmap_ros->set_parameters_atomically(
    {rclcpp::Parameter("global_frame", "odom"),
      rclcpp::Parameter("robot_base_frame", "base_link")});
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State state;
  costmap_ros->on_configure(state);

  // Test initialization and getting costmap basic metadata
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);
  EXPECT_EQ(handler.getMaxCostmapDistWrapper(), 2.5);

  // Set tf between map odom and base_link
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ =
    std::make_unique<tf2_ros::TransformBroadcaster>(node);
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = "map";
  t.child_frame_id = "base_link";
  tf_broadcaster_->sendTransform(t);
  t.header.frame_id = "map";
  t.child_frame_id = "odom";
  tf_broadcaster_->sendTransform(t);

  // Test getting the global plans within a bounds window
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.poses.resize(100);
  for (unsigned int i = 0; i != path.poses.size(); i++) {
    path.poses[i].pose.position.x = i;
    path.poses[i].header.frame_id = "map";
  }
  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "odom";
  robot_pose.pose.position.x = 25.0;

  handler.setPath(path);
  auto [transformed_plan, closest] =
    handler.getGlobalPlanConsideringBoundsInCostmapFrameWrapper(robot_pose);
  auto & path_inverted = handler.getInvertedPath();
  EXPECT_EQ(closest - path_inverted.poses.begin(), 25);
  handler.pruneGlobalPlanWrapper(path_inverted, closest);
  auto & path_pruned = handler.getInvertedPath();
  EXPECT_EQ(path_pruned.poses.size(), 75u);
}

TEST(PathHandlerTests, TestTransforms)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  auto costmap_ros = std::make_shared<SettledTestCostmap>(
    "dummy_costmap", "", "dummy_costmap", true);
  // These path-only fixtures use grid metadata and TF, not sensor/map plugins.
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State state;
  costmap_ros->on_configure(state);

  // Test basic transformations and path handling
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);

  // Set tf between map odom and base_link
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ =
    std::make_unique<tf2_ros::TransformBroadcaster>(node);
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = "map";
  t.child_frame_id = "base_link";
  tf_broadcaster_->sendTransform(t);
  t.header.frame_id = "map";
  t.child_frame_id = "odom";
  tf_broadcaster_->sendTransform(t);

  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  path.poses.resize(100);
  for (unsigned int i = 0; i != path.poses.size(); i++) {
    path.poses[i].pose.position.x = i;
    path.poses[i].header.frame_id = "map";
  }

  geometry_msgs::msg::PoseStamped robot_pose, output_pose;
  robot_pose.header.frame_id = "odom";
  robot_pose.pose.position.x = 2.5;

  EXPECT_TRUE(handler.transformPoseWrapper("map", robot_pose, output_pose));
  EXPECT_EQ(output_pose.pose.position.x, 2.5);

  EXPECT_THROW(handler.transformToGlobalPlanFrameWrapper(robot_pose), std::runtime_error);
  handler.setPath(path);
  EXPECT_NO_THROW(handler.transformToGlobalPlanFrameWrapper(robot_pose));

  auto [path_out, closest] =
    handler.getGlobalPlanConsideringBoundsInCostmapFrameWrapper(robot_pose);

  // Put it all together
  auto final_path = handler.transformPath(robot_pose);
  EXPECT_EQ(final_path.poses.size(), path_out.poses.size());
}

TEST(PathHandlerTests, TestInversionToleranceChecks)
{
  nav_msgs::msg::Path path;
  for (unsigned int i = 0; i != 10; i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = static_cast<double>(i);
    path.poses.push_back(pose);
  }
  path.poses.back().pose.orientation.w = 1;

  PathHandlerWrapper handler;
  handler.setGlobalPlanUpToInversion(path);

  // Not near (0,0)
  geometry_msgs::msg::PoseStamped robot_pose;
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // Exactly on top of it
  robot_pose.pose.position.x = 9;
  robot_pose.pose.orientation.w = 1.0;
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // Laterally of it
  robot_pose.pose.position.y = 9;
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // On top but off angled
  robot_pose.pose.position.y = 0;
  robot_pose.pose.orientation.z = 0.8509035;
  robot_pose.pose.orientation.w = 0.525322;
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // On top but off angled within tolerances
  robot_pose.pose.position.y = 0;
  robot_pose.pose.orientation.w = 0.9961947;
  robot_pose.pose.orientation.z = 0.0871558;
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // Offset spatially + off angled but both within tolerances
  robot_pose.pose.position.x = 9.10;
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(robot_pose));
}

TEST(PathHandlerTests, ShortSegmentCannotBeConsumedAtItsStart)
{
  // Recorded office plans contain 4.8-14.4 cm segments, shorter than the
  // configured 15 cm handoff radius. Being near their endpoint at their start
  // is not evidence that the intervening maneuver has been executed.
  nav_msgs::msg::Path plan;
  plan.poses = {
    makePose(0.0, 0.0, 0.0), makePose(0.048, 0.0, 0.0),
    makePose(0.096, 0.0, 0.0), makePose(0.048, 0.0, 0.0),
    makePose(0.0, 0.0, 0.0)};
  PathHandlerWrapper handler;
  handler.setEnforcePathInversion(true);
  handler.setInversionXYTolerance(0.15);
  handler.setPath(plan);
  ASSERT_GT(handler.getInversionLocale(), 0u);
  const auto gates = handler.getCuspTolerances();
  ASSERT_TRUE(gates.has_value());
  EXPECT_NEAR(gates->first, 0.048, 1e-6);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(plan.poses.front()));
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(0.096, 0.0, 0.0)));
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(0.096, 0.047, 0.0)));
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(makePose(0.096, 0.049, 0.0)));
  handler.setEnforcePathInversion(false);
  EXPECT_FALSE(handler.getCuspTolerances().has_value());
  handler.setEnforcePathInversion(true);
  plan.poses.resize(3);  // Forward final segment has no cusp metadata.
  handler.setPath(plan);
  EXPECT_FALSE(handler.getCuspTolerances().has_value());
}

TEST(PathHandlerTests, GhostHeadingFiltersZigZagSeedNoise)
{
  PathHandlerWrapper handler;
  nav_msgs::msg::Path transformed_plan;
  transformed_plan.header.frame_id = "odom";

  const std::vector<std::pair<double, double>> pts = {
    {0.00, 0.00}, {0.05, 0.004}, {0.10, -0.004}, {0.15, 0.002}, {0.20, 0.000}};
  for (const auto & pt : pts) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "odom";
    pose.pose.position.x = pt.first;
    pose.pose.position.y = pt.second;
    pose.pose.orientation.w = 1.0;
    transformed_plan.poses.push_back(pose);
  }

  double heading = 0.0;
  double heading_variance = 0.0;
  double seed_arc_length = 0.0;
  EXPECT_TRUE(
    handler.estimateGhostHeadingWrapper(
      transformed_plan, heading, heading_variance, seed_arc_length));
  EXPECT_NEAR(heading, 0.0f, 0.08f);
  EXPECT_GT(seed_arc_length, 0.15f);
  EXPECT_LT(heading_variance, 0.03);
}

TEST(PathHandlerTests, GhostPathContinuesCurrentSegmentInsteadOfBorrowingReversePoints)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ghost_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  node->declare_parameter("dummy.enforce_path_inversion", rclcpp::ParameterValue(true));
  node->declare_parameter("dummy.inversion_xy_tolerance", rclcpp::ParameterValue(0.01));
  node->declare_parameter("dummy.min_inversion_horizon", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_max_extension", rclcpp::ParameterValue(0.20));


  node->declare_parameter("dummy.ghost_min_seed_points", rclcpp::ParameterValue(2));
  auto costmap_ros = makeConfiguredCostmapRos();
  ParametersHandler param_handler(node);

  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);
  publishIdentityTf(node, costmap_ros->getTfBuffer());

  const auto path = makeMicroCuspPath();
  handler.setPath(path);

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "odom";
  robot_pose.pose.position.x = 0.30;
  robot_pose.pose.orientation.w = 1.0;

  auto transformed = handler.transformPath(robot_pose);
  ASSERT_GT(transformed.poses.size(), 2u);

  const auto & last_pose = transformed.poses.back().pose.position;
  EXPECT_GT(last_pose.x, 0.34);
  EXPECT_NEAR(last_pose.y, 0.0, 1e-4);
}

TEST(PathHandlerTests, GhostExtensionPreservesVehicleHeadingInReverse)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("reverse_ghost_orientation");
  node->declare_parameter("dummy.ghost_max_extension", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_point_spacing", rclcpp::ParameterValue(0.05));
  auto costmap_ros = makeConfiguredCostmapRos();
  costmap_ros->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  std::fill_n(costmap_ros->getCostmap()->getCharMap(), 10000,
    nav2_costmap_2d::FREE_SPACE);
  ParametersHandler param_handler(node);
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);

  // Vehicle faces +X while the reverse segment travels -X. Orientation is
  // chassis heading, not the direction of displacement.
  for (const double travel_heading : {0.0, M_PI}) {
    nav_msgs::msg::Path path;
    path.header.frame_id = "odom";
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = 0.0;
    pose.pose.position.y = 0.0;
    pose.pose.orientation.w = 1.0;
    path.poses.push_back(pose);
    double length = 0.0;
    ASSERT_GT(handler.appendGhostPathWrapper(path, travel_heading, 0.15, length), 0u);
    EXPECT_NEAR(path.poses.back().pose.position.x, 0.15 * std::cos(travel_heading), 1e-6);
    EXPECT_NEAR(path.poses.back().pose.position.y, 0.0, 1e-6);
    for (const auto & p : path.poses) {
      EXPECT_NEAR(tf2::getYaw(p.pose.orientation), 0.0, 1e-6);
    }
  }
}

TEST(PathHandlerTests, HandoffPreservesSingleEdgeReverseSegment)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("shared_cusp_node");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.01);
  node->declare_parameter("dummy.ghost_max_extension", 0.0);
  auto costmap_ros = makeConfiguredCostmapRos();
  costmap_ros->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  std::fill_n(costmap_ros->getCostmap()->getCharMap(), 10000,
    nav2_costmap_2d::FREE_SPACE);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &params);
  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  for (double x : {0.0, 0.048, 0.096, 0.048, 0.096, 0.144, 0.192}) {
    auto pose = makePose(x, 0.0, 0.0);
    pose.header = path.header;
    path.poses.push_back(pose);
  }
  handler.setPath(path);
  auto robot = makePose(0.096, 0.0, 0.0);
  robot.header = path.header;
  handler.transformPath(robot);
  ASSERT_GT(handler.getInversionLocale(), 0u) << "The one-edge reverse leg disappeared";
  ASSERT_EQ(handler.getInvertedPath().poses.size(), 2u);
  EXPECT_NEAR(handler.getInvertedPath().poses.front().pose.position.x, 0.096, 1e-6);
  EXPECT_NEAR(handler.getInvertedPath().poses.back().pose.position.x, 0.048, 1e-6);
  robot.pose.position.x = 0.048;
  handler.transformPath(robot);
  EXPECT_EQ(handler.getInversionLocale(), 0u);
  EXPECT_NEAR(handler.getPath().poses.front().pose.position.x, 0.048, 1e-6);
}

TEST(PathHandlerTests, GhostPathDoesNotMutateGlobalPlanOrInversionLocale)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ghost_state_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  node->declare_parameter("dummy.enforce_path_inversion", rclcpp::ParameterValue(true));
  node->declare_parameter("dummy.inversion_xy_tolerance", rclcpp::ParameterValue(0.01));
  node->declare_parameter("dummy.min_inversion_horizon", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_max_extension", rclcpp::ParameterValue(0.20));


  node->declare_parameter("dummy.ghost_min_seed_points", rclcpp::ParameterValue(2));
  auto costmap_ros = makeConfiguredCostmapRos();
  ParametersHandler param_handler(node);

  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);
  publishIdentityTf(node, costmap_ros->getTfBuffer());

  const auto path = makeMicroCuspPath();
  handler.setPath(path);
  const auto original_path = handler.getPath();
  const auto original_inversion_locale = handler.getInversionLocale();

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "odom";
  robot_pose.pose.position.x = 0.30;
  robot_pose.pose.orientation.w = 1.0;

  for (size_t i = 0; i < 20; ++i) {
    auto transformed = handler.transformPath(robot_pose);
    EXPECT_FALSE(transformed.poses.empty());
    EXPECT_GT(handler.getRealPathPointCount(), 0u);
    EXPECT_LT(handler.getRealPathPointCount(), transformed.poses.size()) <<
      "Ghost preview was incorrectly reported as real planner geometry";
  }

  const auto & final_path = handler.getPath();
  ASSERT_EQ(final_path.poses.size(), original_path.poses.size());
  for (size_t i = 0; i < final_path.poses.size(); ++i) {
    EXPECT_DOUBLE_EQ(final_path.poses[i].pose.position.x, original_path.poses[i].pose.position.x);
    EXPECT_DOUBLE_EQ(final_path.poses[i].pose.position.y, original_path.poses[i].pose.position.y);
  }
  EXPECT_EQ(handler.getInversionLocale(), original_inversion_locale);
}

TEST(PathHandlerTests, GhostPathStopsBeforeLethalObstacle)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ghost_obstacle_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  node->declare_parameter("dummy.enforce_path_inversion", rclcpp::ParameterValue(true));
  node->declare_parameter("dummy.inversion_xy_tolerance", rclcpp::ParameterValue(0.01));
  node->declare_parameter("dummy.min_inversion_horizon", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_max_extension", rclcpp::ParameterValue(0.20));


  node->declare_parameter("dummy.ghost_min_seed_points", rclcpp::ParameterValue(2));
  auto costmap_ros = makeConfiguredCostmapRos();
  ParametersHandler param_handler(node);

  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);
  publishIdentityTf(node, costmap_ros->getTfBuffer());

  unsigned int mx, my;
  ASSERT_TRUE(costmap_ros->getCostmap()->worldToMap(0.45, 0.0, mx, my));
  costmap_ros->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);

  handler.setPath(makeMicroCuspPath());

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "odom";
  robot_pose.pose.position.x = 0.30;
  robot_pose.pose.orientation.w = 1.0;

  auto transformed = handler.transformPath(robot_pose);
  ASSERT_FALSE(transformed.poses.empty());
  EXPECT_LT(transformed.poses.back().pose.position.x, 0.45);
}

// Helper: create a pose with position and yaw (orientation around Z)
static geometry_msgs::msg::PoseStamped makeYawPose(
  double x, double y, double yaw_rad, const std::string & frame = "map")
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frame;
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(yaw_rad);
  return pose;
}

// Helper: create the M7 180° turn plan reverse segment (from real recording).
// This is a 3-segment Reeds-Shepp path: FWD arc, REV backup, FWD arc to goal.
// Only includes the end of segment 1 + full segment 2 + start of segment 3
// for a focused cusp-transition test.
[[maybe_unused]] static nav_msgs::msg::Path makeM7CuspPath(const std::string & frame = "map")
{
  nav_msgs::msg::Path path;
  path.header.frame_id = frame;

  // End of forward arc (segment 1, last 4 points)
  // pose_yaw matches travel direction (forward)
  path.poses.push_back(makeYawPose(0.766, -0.992, -1.762, frame));  // idx 26, -101°
  path.poses.push_back(makeYawPose(0.755, -1.046, -1.782, frame));  // idx 27, -102°
  path.poses.push_back(makeYawPose(0.744, -1.100, -1.885, frame));  // idx 28, -108°

  // Reverse segment (segment 2, 9 points) — real M7 recording data
  // pose_yaw points with robot nose (~-112° to -131°)
  // travel direction is OPPOSITE (~+71° to +55°, i.e. backing up)
  path.poses.push_back(makeYawPose(0.753, -1.073, -1.954, frame));  // idx 29, -112°, REV
  path.poses.push_back(makeYawPose(0.763, -1.046, -1.990, frame));  // idx 30, -114°
  path.poses.push_back(makeYawPose(0.774, -1.020, -2.025, frame));  // idx 31, -116°
  path.poses.push_back(makeYawPose(0.787, -0.994, -2.060, frame));  // idx 32, -118°
  path.poses.push_back(makeYawPose(0.800, -0.969, -2.094, frame));  // idx 33, -120°
  path.poses.push_back(makeYawPose(0.814, -0.944, -2.129, frame));  // idx 34, -122°
  path.poses.push_back(makeYawPose(0.829, -0.919, -2.164, frame));  // idx 35, -124°
  path.poses.push_back(makeYawPose(0.845, -0.895, -2.199, frame));  // idx 36, -126°
  path.poses.push_back(makeYawPose(0.856, -0.879, -2.291, frame));  // idx 37, -131°

  // Start of final forward arc (segment 3, first 4 points)
  // pose_yaw matches travel direction again (forward, completing turn)
  path.poses.push_back(makeYawPose(0.837, -0.901, -2.302, frame));  // idx 38, -132°, FWD
  path.poses.push_back(makeYawPose(0.818, -0.922, -2.321, frame));  // idx 39, -133°
  path.poses.push_back(makeYawPose(0.798, -0.943, -2.346, frame));  // idx 40, -134°
  path.poses.push_back(makeYawPose(0.778, -0.964, -2.375, frame));  // idx 41, -136°

  return path;
}

// Test that ghost heading estimation produces REVERSE travel direction
// for single-point post-handoff TGP from a reverse segment.
TEST(PathHandlerTests, GhostHeadingSinglePointReverseSegmentFlipsDirection)
{
  PathHandlerWrapper handler;

  // Set up global_plan_up_to_inversion_ as the reverse segment
  // (this is what PathHandler has after a FWD->REV handoff)
  nav_msgs::msg::Path reverse_segment;
  reverse_segment.header.frame_id = "map";
  // Reverse segment points (travel direction is ~+65° = backing up northeast)
  reverse_segment.poses.push_back(makeYawPose(0.753, -1.073, -1.954));  // -112°
  reverse_segment.poses.push_back(makeYawPose(0.763, -1.046, -1.990));  // -114°
  reverse_segment.poses.push_back(makeYawPose(0.774, -1.020, -2.025));  // -116°
  handler.setGlobalPlanUpToInversion(reverse_segment);

  // Simulate single-point TGP (only 1 point survived costmap clipping)
  nav_msgs::msg::Path single_point_tgp;
  single_point_tgp.header.frame_id = "odom";
  single_point_tgp.poses.push_back(makeYawPose(0.753, -1.073, -1.954, "odom"));  // pose_yaw = -112°

  double heading = 0.0, variance = 0.0, arc = 0.0;
  bool ok = handler.estimateGhostHeadingWrapper(single_point_tgp, heading, variance, arc);
  ASSERT_TRUE(ok);

  // The pose yaw is -112° (-1.954 rad), but the travel direction should be
  // flipped to ~+68° (+1.187 rad) because this is a reverse segment.
  // The displacement from pt[0] to pt[1] in global_plan_up_to_inversion_ is
  // (+0.010, +0.027), i.e. travel_dir = atan2(0.027, 0.010) ≈ +70°
  // So the ghost heading should be near +68° (≈ 1.19 rad), NOT -112°.
  const double expected_reverse_heading = std::atan2(
    reverse_segment.poses[1].pose.position.y - reverse_segment.poses[0].pose.position.y,
    reverse_segment.poses[1].pose.position.x - reverse_segment.poses[0].pose.position.x);

  EXPECT_NEAR(heading, expected_reverse_heading, 0.2)
    << "Ghost heading should point in travel (reverse) direction, not pose direction. "
    << "Got " << heading << " rad, expected ~" << expected_reverse_heading << " rad";
}

// Test that multi-point ghost heading naturally produces correct reverse direction
// (displacement-based heading already handles this)
TEST(PathHandlerTests, GhostHeadingReverseGearSurvivesPruningToSingleton)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("pruned_gear_test");
  auto costmap_ros = makeConfiguredCostmapRos();
  costmap_ros->getCostmap()->resizeMap(160, 160, 0.05, -4.0, -4.0);
  ParametersHandler param_handler(node);
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &param_handler);
  nav_msgs::msg::Path active;
  active.header.frame_id = "map";
  active.poses.push_back(makeYawPose(2.612, -1.850, -2.102));
  active.poses.push_back(active.poses.front());  // Duplicate position, no gear information.
  active.poses.push_back(makeYawPose(2.627, -1.826, -2.102));
  auto with_future_segment = active;
  with_future_segment.poses.push_back(makeYawPose(2.58, -1.91, -2.102));
  handler.setGlobalPlan(with_future_segment);
  handler.setInversionLocale(active.poses.size());
  nav_msgs::msg::Path singleton;
  singleton.header.frame_id = "map";
  singleton.poses.push_back(active.poses.back());
  handler.setGlobalPlanUpToInversion(singleton);
  double heading = 0.0, variance = 0.0, arc = 0.0;
  ASSERT_TRUE(handler.estimateGhostHeadingWrapper(singleton, heading, variance, arc));
  EXPECT_NEAR(angles::shortest_angular_distance(-2.102 + M_PI, heading), 0.0, 1e-6);
  double length = 0.0;
  ASSERT_GT(handler.appendGhostPathWrapper(singleton, heading, 0.2, length), 0u);
  EXPECT_GT(singleton.poses.back().pose.position.x, active.poses.back().pose.position.x);
  EXPECT_GT(singleton.poses.back().pose.position.y, active.poses.back().pose.position.y);
  EXPECT_NEAR(angles::shortest_angular_distance(
      -2.102, tf2::getYaw(singleton.poses.back().pose.orientation)), 0.0, 1e-6);
}

TEST(PathHandlerTests, SingletonGearUsesEdgeHeadingOnLongCurves)
{
  PathHandlerWrapper handler;
  for (const bool reversing : {false, true}) {
    nav_msgs::msg::Path active;
    active.header.frame_id = "map";
    active.poses.push_back(makeYawPose(0.0, 0.0, 0.0));
    const double sign = reversing ? -1.0 : 1.0;
    active.poses.push_back(makeYawPose(sign * 0.1, 0.0, 2.1));
    active.poses.push_back(makeYawPose(
        sign * (0.1 + 0.1 * std::cos(2.1)), sign * 0.1 * std::sin(2.1), 2.1));
    handler.setGlobalPlan(active);
    nav_msgs::msg::Path singleton;
    singleton.header.frame_id = "map";
    singleton.poses.push_back(active.poses.back());
    handler.setGlobalPlanUpToInversion(singleton);
    double heading = 0.0, variance = 0.0, arc = 0.0;
    ASSERT_TRUE(handler.estimateGhostHeadingWrapper(singleton, heading, variance, arc));
    EXPECT_NEAR(angles::shortest_angular_distance(
        2.1 + (reversing ? M_PI : 0.0), heading), 0.0, 1e-6);
  }
}

TEST(PathHandlerTests, GhostHeadingMultiPointReverseSegmentUsesDisplacement)
{
  PathHandlerWrapper handler;

  // Multi-point reverse segment as TGP (2+ points)
  nav_msgs::msg::Path reverse_tgp;
  reverse_tgp.header.frame_id = "odom";
  // These are backing up: nose at -112° to -116° but moving northeast
  reverse_tgp.poses.push_back(makeYawPose(0.753, -1.073, -1.954, "odom"));
  reverse_tgp.poses.push_back(makeYawPose(0.763, -1.046, -1.990, "odom"));
  reverse_tgp.poses.push_back(makeYawPose(0.774, -1.020, -2.025, "odom"));
  reverse_tgp.poses.push_back(makeYawPose(0.787, -0.994, -2.060, "odom"));

  double heading = 0.0, variance = 0.0, arc = 0.0;
  bool ok = handler.estimateGhostHeadingWrapper(reverse_tgp, heading, variance, arc);
  ASSERT_TRUE(ok);

  // Displacement heading from consecutive points should be ~+65° (backing up)
  // NOT the pose heading of ~-115°
  EXPECT_GT(heading, 0.5) << "Multi-point heading should be positive (backing up northeast)";
  EXPECT_LT(heading, 1.5) << "Multi-point heading should be ~1.1 rad (~65°)";
  EXPECT_GT(arc, 0.05);
}

// ── Layer 1: time-adaptive handoff yaw gate ─────────────────────────────────
// Regression family for the M5 cusp-handoff deadlock (2026-09 Phase-1
// campaign): xy tolerance met, yaw error above tolerance, no escape existed.

TEST(PathHandlerTests, TestHandoffYawRelaxationUnlocks)
{
  nav_msgs::msg::Path path;
  for (unsigned int i = 0; i != 10; i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = static_cast<double>(i);
    path.poses.push_back(pose);
  }
  path.poses.back().pose.orientation.w = 1;  // cusp heading = 0

  PathHandlerWrapper handler;
  handler.setGlobalPlanUpToInversion(path);

  // On top of the cusp in xy, 0.6 rad off in yaw (tol 0.4, relax cap 0.8)
  auto robot_pose = makePose(9.0, 0.0, 0.6);

  // Blocked at the static tolerance
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));

  // Repeated blocked cycles must relax the gate until it unlocks
  bool unlocked = false;
  for (int cycle = 0; cycle != 300 && !unlocked; ++cycle) {
    unlocked = handler.isWithinInversionTolerancesWrapper(robot_pose);
  }
  EXPECT_TRUE(unlocked) <<
    "handoff gate never relaxed: the M5 deadlock (xy met, yaw blocked forever)";

  // Unlock must reset the blocked state: a fresh over-cap error blocks again
  auto worse_pose = makePose(9.0, 0.0, 1.2);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(worse_pose));
}

TEST(PathHandlerTests, TestHandoffNoRelaxationWhenFarInXY)
{
  nav_msgs::msg::Path path;
  for (unsigned int i = 0; i != 10; i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = static_cast<double>(i);
    path.poses.push_back(pose);
  }
  path.poses.back().pose.orientation.w = 1;

  PathHandlerWrapper handler;
  handler.setGlobalPlanUpToInversion(path);

  // Far away in xy: many cycles must NOT accumulate relaxation
  auto far_pose = makePose(4.0, 3.0, 0.6);
  for (int cycle = 0; cycle != 200; ++cycle) {
    EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(far_pose));
  }

  // First arrival at the cusp with yaw over tolerance: still blocked (no
  // banked relaxation from the far cycles)
  auto near_pose = makePose(9.0, 0.0, 0.6);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(near_pose));
}

// ── Layer 2: escape along the next segment after persistent blocking ────────
// The M5 deadlock geometry from the 2026-09 Phase-1 campaign: a ~0.28m first
// segment ending in a cusp, robot at the cusp with ~1.0 rad yaw error — above
// the relax cap, so Layer 1 alone must NOT unlock, and the escape policy must
// offer the post-cusp travel direction instead.
TEST(PathHandlerTests, TestHandoffEscapeHeadingAfterBlockedM5Regression)
{
  // Segment A: forward +x, 0.28 m; segment B: reversing back -x (inversion)
  nav_msgs::msg::Path full_plan;
  std::vector<std::pair<double, double>> pts = {
    {0.00, 0.0}, {0.04, 0.0}, {0.08, 0.0}, {0.12, 0.0}, {0.16, 0.0},
    {0.20, 0.0}, {0.24, 0.0}, {0.28, 0.0},           // cusp at index 7
    {0.23, 0.01}, {0.18, 0.02}, {0.13, 0.03}, {0.08, 0.04}};  // reverse leg
  for (const auto & [x, y] : pts) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.w = 1.0;
    full_plan.poses.push_back(pose);
  }
  nav_msgs::msg::Path segment_a = full_plan;
  segment_a.poses.resize(8);

  PathHandlerWrapper handler;
  handler.setEnforcePathInversion(true);
  handler.setGlobalPlan(full_plan);
  handler.setGlobalPlanUpToInversion(segment_a);
  handler.setInversionLocale(8u);  // first pose after the cusp

  // At the cusp in xy, 1.0 rad off in yaw: above the 0.8 relax cap
  auto robot_pose = makePose(0.28, 0.0, 1.0);

  double heading = 0.0;
  // Escape is DISABLED by default (handoff_escape_after_cycles <= 0): even a
  // long block must not engage it.
  for (int cycle = 0; cycle != 25; ++cycle) {
    EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));
    EXPECT_FALSE(handler.computeEscapeHeadingWrapper(heading)) <<
      "escape engaged while disabled, after " << cycle + 1 << " blocked cycles";
  }

  // Enable with a 20-cycle threshold: the accumulated block engages escape
  // with the post-cusp travel direction
  handler.setHandoffEscapeAfterCycles(20);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(robot_pose));
  ASSERT_TRUE(handler.computeEscapeHeadingWrapper(heading)) <<
    "escape policy never engaged: the M5 deadlock would park here forever";
  // B leg travels -x (slightly +y): heading ~ pi
  EXPECT_NEAR(std::fabs(heading), M_PI, 0.30);

  // Leaving the xy tolerance resets the blocked state and disarms escape
  auto away_pose = makePose(1.5, 0.0, 1.0);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(away_pose));
  EXPECT_FALSE(handler.computeEscapeHeadingWrapper(heading));
}

TEST(PathHandlerTests, OptInAdjacentGateCannotSkipShortOutgoingLeg)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("adjacent_gate_reduction");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
  node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
  auto costmap_ros = makeConfiguredCostmapRos();
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", costmap_ros, costmap_ros->getTfBuffer(), &params);
  nav_msgs::msg::Path plan;
  plan.header.frame_id = "odom";
  plan.poses = {makePose(0.0, 0.0, 0.0), makePose(0.5, 0.0, 0.0),
    makePose(1.0, 0.0, 0.0), makePose(0.972, 0.0, 0.0), makePose(1.2, 0.0, 0.0)};
  handler.setPath(plan);
  ASSERT_GT(handler.getInversionLocale(), 0u);
  // Recorded first reversal is only2.8cm, but the incoming gate allows6.76cm.
  // At6cm before the cusp, the next reverse endpoint is still ahead of robot.
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(makePose(0.94, 0.0, 0.0)));
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(makePose(0.98, 0.0, 0.0)));
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(0.99, 0.0, 0.0)));
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(1.0, 0.0, 0.0)));
}

TEST(PathHandlerTests, TestHandoffRelaxationCapHolds)
{
  nav_msgs::msg::Path path;
  for (unsigned int i = 0; i != 10; i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = static_cast<double>(i);
    path.poses.push_back(pose);
  }
  path.poses.back().pose.orientation.w = 1;

  // Just under the 0.8 relax cap: unlocks once relaxation reaches it
  {
    PathHandlerWrapper handler;
    handler.setGlobalPlanUpToInversion(path);
    auto pose = makePose(9.0, 0.0, 0.79);
    bool unlocked = false;
    for (int cycle = 0; cycle != 400 && !unlocked; ++cycle) {
      unlocked = handler.isWithinInversionTolerancesWrapper(pose);
    }
    EXPECT_TRUE(unlocked);
  }

  // Above the cap: must NEVER unlock, no matter how long it blocks
  {
    PathHandlerWrapper handler;
    handler.setGlobalPlanUpToInversion(path);
    auto pose = makePose(9.0, 0.0, 0.85);
    for (int cycle = 0; cycle != 400; ++cycle) {
      EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(pose)) <<
        "relaxation exceeded its cap at cycle " << cycle;
    }
  }
}


TEST(PathHandlerTests, RecordedIncomingHandoffLeavesMicroReverseEndpointAhead)
{
 for(bool bound : {false,true}) {
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>(bound ? "recorded_bound" : "recorded_default");
  node->declare_parameter("dummy.enforce_path_inversion",true);
  node->declare_parameter("dummy.inversion_xy_tolerance",0.0676);
  node->declare_parameter("dummy.inversion_yaw_tolerance",0.3);
  node->declare_parameter("dummy.bound_handoff_by_next_segment",bound);
  auto cm=makeConfiguredCostmapRos();ParametersHandler params(node);PathHandlerWrapper handler;
  handler.initialize(node,"dummy",cm,cm->getTfBuffer(),&params);
  nav_msgs::msg::Path plan;plan.header.frame_id="map";plan.poses={makePose(2.855249596359215, -0.68992668028340631, -1.0043557325946255),
    makePose(2.8698995594106691, -0.71433524907376977, -1.0561206976519986),
    makePose(2.8832668686821989, -0.7394695258557249, -1.1078856627093716),
    makePose(2.895315513432422, -0.76526115125579963, -1.1596506277667447),
    makePose(2.9060137553807976, -0.79164237625207079, -1.2114155928241179),
    makePose(2.9153326028287552, -0.81854117936177317, -1.2631805578814908),
    makePose(2.9232473365385658, -0.84588675980523931, -1.3149455229388636),
    makePose(2.9297362890302452, -0.87360465469350856, -1.3667104879962371),
    makePose(2.9347826756361997, -0.90162162184071803, -1.4184754530536099),
    makePose(2.9383724582708055, -0.92986197765480938, -1.4702404181109832),
    makePose(2.9404961764850555, -0.95825003854372426, -1.5220053831683562),
    makePose(2.9411483371150098, -0.98671012091540433, -1.573770348225729),
    makePose(2.9403268039302475, -1.0151659308262424, -1.6255357901202605),
    makePose(2.9380343235127384, -1.0435411743326313, -1.6773007551776336),
    makePose(2.9342763890264223, -1.0717595574909637, -1.7290657202350064),
    makePose(2.9330279149332341, -1.080291051440895, -1.7808306852923794),
    makePose(2.939681662343105, -1.0526122190517526, -1.8325956503497525),
    makePose(2.939681662343105, -1.0526122190517526, -1.8325956503497525),
    makePose(2.9272631445542743, -1.0989586535627041, -1.8325956503497525),
    makePose(2.9148446267654435, -1.1453050880736555, -1.8325956503497525),
    makePose(2.9024261089766128, -1.1916515225846069, -1.8325956503497525),
    makePose(2.8900075911877821, -1.2379979570955584, -1.8325956503497525),
    makePose(2.8775890733989513, -1.2843443916065098, -1.8325956503497525),
    makePose(2.8651705556101206, -1.3306908261174613, -1.8325956503497525),
    makePose(2.8527520378212898, -1.3770372606284127, -1.8325956503497525),
    makePose(2.8383239375578455, -1.4227977576524609, -1.9198620955096646),
    makePose(2.8199624267377885, -1.467126980294438, -2.0071285406695769),
    makePose(2.7996847172300932, -1.510612697095894, -2.0071285406695769)};handler.setPath(plan);
  // 032013 map-body sample converted to rear axle at the recorded handoff.
  const double yaw=-1.5681685397934615;
  const double x=2.9507091261067964-0.106*cos(yaw);
  const double y=-1.122164027889751-0.106*sin(yaw);
  EXPECT_EQ(handler.isWithinInversionTolerancesWrapper(makePose(x,y,yaw)),!bound);
  const auto &next=plan.poses[16].pose.position;
  EXPECT_GT((next.x-x)*cos(yaw)+(next.y-y)*sin(yaw),0.03);
  // Destroy handler/parameter callbacks before the costmap owner. Calling
  // cleanup here while those bindings are live can block fixture teardown.
 }
}

TEST(PathHandlerTests, PreserveCuspRealDirectionEdgeIsOptInAndDoesNotChangeFinalGoal)
{
 for (bool enabled : {false,true}) for (bool cusp : {false,true}) for (int gear : {1,-1}) {
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("edge_prune_regression");
  node->declare_parameter("dummy.enforce_path_inversion",true);
  node->declare_parameter("dummy.preserve_cusp_direction_edge",enabled);
  auto cm=makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(100,100,0.05,-2.5,-2.5);
  ParametersHandler params(node);PathHandlerWrapper handler;
  handler.initialize(node,"dummy",cm,cm->getTfBuffer(),&params);
  nav_msgs::msg::Path active;active.header.frame_id="odom";
  active.poses={makePose(0,0,0),makePose(0.1*gear,0,0),makePose(0.2*gear,0,0),makePose(0.2*gear,0,0)};
  handler.setGlobalPlan(active);handler.setGlobalPlanUpToInversion(active);
  handler.setInversionLocale(cusp ? 4 : 0);
  auto robot=makePose(0.25*gear,0,0);robot.header.frame_id="odom";
  auto [local,bound]=handler.getGlobalPlanConsideringBoundsInCostmapFrameWrapper(robot);
  ASSERT_GE(local.poses.size(),1u);
  EXPECT_NEAR(local.poses.front().pose.position.x,(enabled && cusp ? 0.1 : 0.2)*gear,1e-7);
  // The endpoint is never moved, and retained points are real planner geometry.
  EXPECT_NEAR(local.poses.back().pose.position.x,0.2*gear,1e-7);
  // Handler and parameter bindings are destroyed before the costmap owner.
 }
}

TEST(PathHandlerTests, ShortDepartureScopesAndExcludesFinalGoalsWithoutValidator)
{
  for (bool enabled : {false, true}) {
    for (int incoming : {-1, 1}) {
      for (bool final : {false, true}) {
        for (int points : {2, 4}) {
          auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("short_scope");
          node->declare_parameter("dummy.enforce_path_inversion", true);
          node->declare_parameter("dummy.validate_short_forward_departure", enabled);
          node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
          node->declare_parameter("dummy.inversion_yaw_tolerance", 0.1);
          node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
          auto cm = makeConfiguredCostmapRos();
          cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
          ParametersHandler params(node);
          PathHandlerWrapper handler;
          handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
          nav_msgs::msg::Path plan;
          plan.header.frame_id = "odom";
          plan.poses = {makePose(-incoming * 0.2, 0, 0),
            makePose(-incoming * 0.1, 0, 0), makePose(0, 0, 0)};
          for (int k = 1; k <= points; ++k) {
            plan.poses.push_back(makePose(-incoming * k * 0.1, 0, 0));
          }
          if (!final) {plan.poses.push_back(makePose(-incoming * (points - 1) * 0.1, 0, 0));}
          handler.setPath(plan);
          auto robot = makePose(-incoming * 0.02, 0, 0);
          robot.header.frame_id = "odom";
          const bool scoped = enabled && incoming == -1 && !final && points == 2;
          // Missing validator is an error only within the enabled policy scope.
          if (scoped) {
            EXPECT_THROW(handler.transformPath(robot), nav2_core::NoValidControl);
            EXPECT_FALSE(handler.didHandoff());
          } else {
            EXPECT_NO_THROW(handler.transformPath(robot));
            EXPECT_TRUE(handler.didHandoff());
          }
        }
      }
    }
  }
}

TEST(PathHandlerTests, ShortDepartureRejectionStopsWithoutConsumingCusp)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("short_departure_guard");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.validate_short_forward_departure", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.1);
  node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan;
  plan.header.frame_id = "odom";
  plan.poses = {makePose(0.2, 0, 0), makePose(0.1, 0, 0), makePose(0, 0, 0),
    makePose(0.1, 0, 0), makePose(0.2, 0, 0), makePose(0.1, 0, 0)};
  handler.setPath(plan);
  auto robot = makePose(0.02, 0, 0);
  robot.header.frame_id = "odom";
  int calls = 0;
  EXPECT_THROW(handler.transformPath(robot,
    [&](const auto &, const auto & outgoing, double xy, double yaw, bool terminal) {
      ++calls;
      EXPECT_TRUE(terminal);
      EXPECT_EQ(outgoing.poses.size(), 3u);
      EXPECT_NEAR(xy, 0.05, 1e-6);
      EXPECT_NEAR(yaw, 0.1, 1e-6);
      return false;
    }), nav2_core::FailedToMakeProgress);
  EXPECT_EQ(calls, 1);
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getInversionLocale(), 3u);
  EXPECT_EQ(handler.getPath().poses.size(), plan.poses.size());
  EXPECT_EQ(handler.getPathDirection(), -1);
  handler.transformPath(robot, [](const auto &, const auto &, double, double, bool) {return true;});
  EXPECT_TRUE(handler.didHandoff());
  EXPECT_EQ(handler.getPathDirection(), 1);
}

TEST(PathHandlerTests, RejectedShortDepartureAllowsBoundedIncomingApproach)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("retained_short_approach");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.validate_short_forward_departure", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.1);
  node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  ParametersHandler params(node); PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  // Structural dispatch reduction: reverse approach, short forward internal cusp.
  // Actual attempt8 outgoing feasibility is established by its separate native replay.
  nav_msgs::msg::Path plan; plan.header.frame_id = "odom";
  plan.poses = {makePose(0.2,0,0), makePose(0.1,0,0), makePose(0,0,0),
    makePose(0.1,0,0), makePose(0.2,0,0), makePose(0.1,0,0)};
  handler.setPath(plan);
  auto robot = makePose(0.048,0,0); robot.header.frame_id = "odom";
  robot.header.stamp = rclcpp::Time(1000000000);
  int decisions = 0;
  auto reject = [&](const auto & local_robot, const auto & outgoing, double xy, double yaw, bool terminal) {
      ++decisions;
      EXPECT_EQ(local_robot.header.frame_id,"odom"); EXPECT_EQ(outgoing.header.frame_id,"odom");
      EXPECT_DOUBLE_EQ(local_robot.pose.position.x,robot.pose.position.x);
      EXPECT_NEAR(xy,0.05,1e-8); EXPECT_NEAR(yaw,0.1,1e-8); EXPECT_TRUE(terminal);
      EXPECT_EQ(outgoing.poses.size(),3u);
      EXPECT_DOUBLE_EQ(outgoing.poses[0].pose.position.x,0.0);
      EXPECT_DOUBLE_EQ(outgoing.poses[1].pose.position.x,0.1);
      EXPECT_DOUBLE_EQ(outgoing.poses[2].pose.position.x,0.2);
      return false;
    };
  EXPECT_NO_THROW(handler.transformPath(robot,reject));
  ASSERT_TRUE(handler.getRetainedDepartureApproach());
  EXPECT_NEAR(handler.getRetainedDepartureApproach()->stop_margin,0.005,1e-8);
  EXPECT_EQ(handler.getRetainedDepartureApproach()->cusp.header.frame_id,"odom");
  EXPECT_DOUBLE_EQ(handler.getRetainedDepartureApproach()->cusp.pose.position.x,0.0);
  EXPECT_EQ(handler.getPath().poses,plan.poses);
  EXPECT_FALSE(handler.didHandoff()); EXPECT_EQ(handler.getPathDirection(),-1);
  EXPECT_EQ(handler.getInversionLocale(),3u);
  robot.pose.position.x=0.028; robot.header.stamp=rclcpp::Time(2000000000);
  EXPECT_NO_THROW(handler.transformPath(robot,reject));
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getPath().poses,plan.poses);
  handler.transformPath(robot,[&](const auto & p, const auto & outgoing, double xy, double yaw, bool terminal) {
    reject(p,outgoing,xy,yaw,terminal);return true;
  });
  EXPECT_TRUE(handler.didHandoff()); EXPECT_EQ(handler.getPathDirection(),1);
  EXPECT_FALSE(handler.getRetainedDepartureApproach());
  handler.transformPath(robot,reject);
  EXPECT_FALSE(handler.didHandoff()); EXPECT_EQ(decisions,3);
  for (int failure = 0; failure < 10; ++failure) {
    handler.setPath(plan);
    robot.pose = makePose(0.048,0,0).pose;
    if (failure == 7) {robot.pose.position.x=0.026;}
    robot.header.stamp = rclcpp::Time(1000000000);
    ASSERT_NO_THROW(handler.transformPath(robot,reject));
    robot.header.stamp = rclcpp::Time(1250000000);
    if (failure == 0) {robot.pose.position.x=0.0495;}  // Moving away.
    if (failure == 1) {robot.pose.position.y=0.03;}  // Leaves XY gate.
    if (failure == 2) {robot.header.stamp=rclcpp::Time(500000000);}  // Clock rollback.
    if (failure == 3) {robot.header.stamp=rclcpp::Time(3100000000LL);}  // Total timeout.
    if (failure == 4) {robot.header.stamp=rclcpp::Time(2000000000);}  // No progress.
    if (failure == 5) {robot.pose.position.x=0.022;}  // Travel budget exhausted.
    if (failure == 7) {robot.pose.position.x=0.004;}  // Stop margin reached.
    if (failure == 8) {robot.pose.position.x=0.046;robot.pose.position.y=0.019;}  // Radial worsening.
    if (failure == 9) {robot.pose.orientation=makePose(0,0,0.2).pose.orientation;}
    if (failure == 6) {
      for (int call=1;call<64;++call) {ASSERT_NO_THROW(handler.transformPath(robot,reject));}
    }
    EXPECT_THROW(handler.transformPath(robot,reject),nav2_core::FailedToMakeProgress);
    EXPECT_FALSE(handler.didHandoff()); EXPECT_FALSE(handler.getRetainedDepartureApproach());
    EXPECT_EQ(handler.getPathDirection(),-1);
    if (failure == 1) {
      robot.pose = makePose(0.043,0,0).pose;
      EXPECT_THROW(handler.transformPath(robot,[](const auto &,const auto &,double,double,bool) {
        return true;
      }),nav2_core::FailedToMakeProgress);  // Gate reentry cannot renew the failed window.
    }
  }
  auto different_plan=plan;
  for(auto & target:different_plan.poses) {target.pose.position.x+=0.3;}
  handler.setPath(different_plan);robot.pose=makePose(0.348,0,0).pose;
  robot.header.stamp=rclcpp::Time(1000000000);
  EXPECT_NO_THROW(handler.transformPath(robot,[](const auto &,const auto &,double,double,bool) {return false;}));
  EXPECT_TRUE(handler.getRetainedDepartureApproach());
  EXPECT_FALSE(handler.didHandoff());EXPECT_EQ(handler.getPathDirection(),-1);
}

TEST(PathHandlerTests, DepartureGuardRetainsIncomingCuspAndScopesToLongForwardExit)
{
  for (bool enabled : {false, true}) {
    for (int incoming : {-1, 1}) {
      for (int outgoing_points : {4, 12}) {
        for (bool allowed : {false, true}) {
          auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("departure_guard");
          node->declare_parameter("dummy.enforce_path_inversion", true);
          node->declare_parameter("dummy.validate_long_forward_departure", enabled);
          node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
          node->declare_parameter("dummy.inversion_yaw_tolerance", 0.1);
          node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
          auto cm = makeConfiguredCostmapRos();
          cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
          ParametersHandler params(node);
          PathHandlerWrapper handler;
          handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
          nav_msgs::msg::Path plan;
          plan.header.frame_id = "odom";
          plan.poses = {makePose(-incoming * 0.2, 0, 0),
            makePose(-incoming * 0.1, 0, 0), makePose(0, 0, 0)};
          for (int k = 1; k <= outgoing_points; ++k) {
            plan.poses.push_back(makePose(-incoming * k * 0.1, 0, 0));
          }
          handler.setPath(plan);
          auto robot = makePose(-incoming * 0.02, 0, 0);
          robot.header.frame_id = "odom";
          int calls = 0;
          auto validator = [&](const auto & local_robot, const auto & prefix, double xy, double yaw, bool terminal) {
              ++calls;
              EXPECT_EQ(local_robot.header.frame_id, "odom");
              EXPECT_EQ(prefix.header.frame_id, "odom");
              EXPECT_FALSE(terminal);
              EXPECT_EQ(prefix.poses.size(), 9u);
              EXPECT_DOUBLE_EQ(prefix.poses.front().pose.position.x, 0.0);
              EXPECT_NEAR(xy, 0.05, 1e-6);
              EXPECT_NEAR(yaw, 0.1, 1e-6);
              return allowed;
            };
          const bool scoped = enabled && incoming == -1 && outgoing_points == 12;
          handler.transformPath(robot, validator);
          EXPECT_EQ(calls, scoped ? 1 : 0);
          EXPECT_EQ(handler.didHandoff(), !scoped || allowed);
          EXPECT_EQ(handler.getPathDirection(), scoped && !allowed ? incoming : -incoming);
          if (scoped && !allowed) {
            EXPECT_EQ(handler.getInversionLocale(), 3u);
            EXPECT_DOUBLE_EQ(handler.getPath().poses.front().pose.position.x,
              plan.poses.front().pose.position.x);
            // A fresh positive decision can advance the same unconsumed cusp.
            handler.transformPath(robot, [](const auto &, const auto &, double, double, bool) {
              return true;
            });
            EXPECT_TRUE(handler.didHandoff());
          }
          // With no remaining inversion, the final segment never consults it.
          plan.poses.resize(3);
          handler.setPath(plan);
          handler.transformPath(robot, [](const auto &, const auto &, double, double, bool) {
            ADD_FAILURE() << "Final segment unexpectedly invoked departure guard";
            return false;
          });
          EXPECT_FALSE(handler.didHandoff());
          EXPECT_EQ(handler.getActiveCuspDirection(), 0);
        }
      }
    }
  }
}

TEST(PathHandlerTests, CuspGearMetadataSurvivesPruningAndClearsForFinalPlan)
{
  PathHandlerWrapper handler;
  handler.setEnforcePathInversion(true);
  for (int gear : {-1,1}) {
    nav_msgs::msg::Path plan;plan.header.frame_id="odom";
    plan.poses={makePose(0,0,0),makePose(0.1*gear,0,0),
      makePose(0.2*gear,0,0),makePose(0.1*gear,0,0)};
    handler.setPath(plan);
    ASSERT_NE(handler.getInversionLocale(),0u);
    EXPECT_EQ(handler.getActiveCuspDirection(),gear);
    // Exercise the production prune function, leaving just the active cusp.
    auto & active=handler.getInvertedPath();
    handler.pruneGlobalPlanWrapper(active,active.poses.end()-1);
    ASSERT_EQ(active.poses.size(),1u);
    EXPECT_EQ(handler.getActiveCuspDirection(),gear);
    plan.poses.resize(3);
    handler.setPath(plan);
    EXPECT_EQ(handler.getActiveCuspDirection(),0);
    EXPECT_EQ(handler.getPathDirection(),gear);
    auto & final_path=handler.getInvertedPath();
    handler.pruneGlobalPlanWrapper(final_path,final_path.poses.end()-1);
    EXPECT_EQ(handler.getPathDirection(),gear);
    EXPECT_EQ(handler.getActiveCuspDirection(),0);
  }
  handler.setEnforcePathInversion(false);
  EXPECT_EQ(handler.getActiveCuspDirection(),0);
  EXPECT_EQ(handler.getPathDirection(),0);
  handler.setEnforcePathInversion(true);
  nav_msgs::msg::Path singleton;singleton.poses={makePose(0,0,0)};
  handler.setPath(singleton);
  EXPECT_EQ(handler.getPathDirection(),0);
}

TEST(PathHandlerTests, Recorded045042OverrunDispatchesStoppedReplanWithoutHandoff)
{
  for (bool enabled : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_045042");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", enabled);
    node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
    node->declare_parameter("dummy.inversion_yaw_tolerance", 0.10);
    node->declare_parameter("dummy.ghost_max_extension", 0.0);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(200, 200, 0.05, 0.0, -5.0);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    // Captured incoming leg, repeated cusp and first outgoing edge. The
    // incoming 4.418 cm leg sets the effective 2.209 cm overrun gate.
    plan.poses = {
      makePose(2.9299715795523298, -1.0957042591041954, -1.752260510121481),
      makePose(2.9329708470634159, -1.0805583854192946, -1.7803342978106897),
      makePose(2.939681662343105, -1.0526122190517526, -1.8325956503497525),
      makePose(2.939681662343105, -1.0526122190517526, -1.8325956503497525),
      makePose(2.9272631445542743, -1.0989586535627041, -1.8325956503497525)};
    handler.setPath(plan);
    ASSERT_EQ(handler.getActiveCuspDirection(), -1);
    auto robot = makePose(2.9193935851747321, -1.0062219843016664, -1.7842502210185485);
    robot.header.frame_id = "odom";
    if (enabled) {
      EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
    } else {
      EXPECT_NO_THROW(handler.transformPath(robot));
    }
    EXPECT_FALSE(handler.didHandoff());
    EXPECT_EQ(handler.getActiveCuspDirection(), -1);
    EXPECT_EQ(handler.getRealPathPointCount(), 1u);
  }
}

TEST(PathHandlerTests, Recorded045042FarTurnOverrunPreservesReverseCusp)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_045042_far");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.replan_on_cusp_overrun", true);
  node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.05);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.10);
  node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
  node->declare_parameter("dummy.handoff_yaw_relax_per_cycle", 0.0);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(200, 200, 0.05, 0.0, -5.0);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan;
  plan.header.frame_id = "odom";
  // Exact native snapshot at 69154.385; retain enough of the outgoing leg
  // that the incoming length still determines the adjacent-segment gate.
  plan.poses = {
    makePose(4.9507973400474157, 0.17508184530402104, -1.5192430655108853),
    makePose(4.9506679455190579, 0.17770269485481549, -1.5236161390887661),
    makePose(4.9500002209246077, 0.20599981336295059, -1.5707958380328579),
    makePose(4.9520931163856403, 0.25393560330712717, -1.6580627600299283),
    makePose(4.9458302991428127, 0.20636480358916742, -1.7453292051898404),
    makePose(4.935445167539001, 0.15952093256589706, -1.8325956503497525)};
  handler.setPath(plan);
  ASSERT_EQ(handler.getActiveCuspDirection(), -1);
  auto robot = makePose(4.9244143867237007, 0.30540461176918282, -1.5813369403225011);
  robot.header.frame_id = "odom";
  EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getActiveCuspDirection(), -1);
  EXPECT_EQ(handler.getRealPathPointCount(), 1u);
}

TEST(PathHandlerTests, RecordedConsumedReverseCuspRequestsReplanOnlyWhenEnabled)
{
  for (bool enabled : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_recorded");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", enabled);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
    node->declare_parameter("dummy.inversion_yaw_tolerance", 0.3);
    node->declare_parameter("dummy.ghost_max_extension", 0.0);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(200, 200, 0.05, 0.0, -5.0);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    // Unmodified reverse leg and first outgoing edge from frozen053000 plan.
    plan.poses = {makePose(4.916831276353491, 0.01788885565349574, -1.9198620955096646),
      makePose(4.935193092349323, 0.06221807829547288, -2.007128540669577),
      makePose(4.957348243221304, 0.1047778917973119, -2.0943949858294886),
      makePose(4.9831282719419505, 0.14524480983811827, -2.1816614309894007),
      makePose(5.012337255664079, 0.1833112152372678, -2.2689278761493132),
      makePose(5.044753026423905, 0.21868658065750424, -2.356194321309225),
      makePose(5.080128391844141, 0.2511023514173303, -2.4434607664691375),
      makePose(5.118194797243291, 0.2803113351394586, -2.5307272116290496),
      makePose(5.0777278792024845, 0.2545306960672633, -2.6179936567889612)};
    handler.setPath(plan);
    ASSERT_EQ(handler.getActiveCuspDirection(), -1);
    // Last recorded map/base_link sample converted to rear_axle (x=-0.106m).
    auto robot = makePose(5.076372978711762, 0.4742844758058707, -2.1312458811053676);
    robot.header.frame_id = "odom";
    if (enabled) {
      EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(handler.didHandoff());
      EXPECT_EQ(handler.getActiveCuspDirection(), -1);
    } else {
      EXPECT_NO_THROW(handler.transformPath(robot));
      EXPECT_EQ(handler.getRealPathPointCount(), 1u);
    }
  }
}

TEST(PathHandlerTests, CuspOverrunPolicyPreservesApproachGateAndFinalGoal)
{
  for (int gear : {-1, 1}) {
    for (bool cusp : {false, true}) {
      for (double progress : {0.15, 0.23, 0.30}) {
        auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_boundaries");
        node->declare_parameter("dummy.enforce_path_inversion", true);
        node->declare_parameter("dummy.replan_on_cusp_overrun", true);
        node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
        node->declare_parameter("dummy.ghost_max_extension", 0.0);
        auto cm = makeConfiguredCostmapRos();
        cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
        ParametersHandler params(node);
        PathHandlerWrapper handler;
        handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
        nav_msgs::msg::Path plan;
        plan.header.frame_id = "odom";
        plan.poses = {makePose(0, 0, 0), makePose(gear * 0.1, 0, 0),
          makePose(gear * 0.2, 0, 0)};
        if (cusp) {
          plan.poses.push_back(makePose(gear * 0.1, 0, 0));
        }
        handler.setPath(plan);
        auto robot = makePose(gear * progress, 0, 0);
        robot.header.frame_id = "odom";
        if (cusp && progress > 0.2676) {
          EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
          EXPECT_FALSE(handler.didHandoff());
        } else {
          EXPECT_NO_THROW(handler.transformPath(robot));
        }
      }
    }
  }
}

TEST(PathHandlerTests, ConsumedCuspLateralErrorAloneDoesNotRequestReplan)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_lateral_control");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.replan_on_cusp_overrun", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
  node->declare_parameter("dummy.ghost_max_extension", 0.0);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan;
  plan.header.frame_id = "odom";
  plan.poses = {makePose(0, 0, 0), makePose(0.1, 0, 0),
    makePose(0.2, 0, 0), makePose(0.1, 0, 0)};
  handler.setPath(plan);
  auto robot = makePose(0.2, 0.15, 0.0);
  robot.header.frame_id = "odom";
  EXPECT_NO_THROW(handler.transformPath(robot));
  EXPECT_EQ(handler.getRealPathPointCount(), 1u);
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getActiveCuspDirection(), 1);
}

TEST(PathHandlerTests, CuspOverrunUsesTheSameAdjacentSegmentBoundAsHandoff)
{
  for (bool bounded : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_adjacent_bound");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", true);
    node->declare_parameter("dummy.bound_handoff_by_next_segment", bounded);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
    node->declare_parameter("dummy.ghost_max_extension", 0.0);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    // Sufficient geometry for the live first recovery: short outgoing leg
    // makes the effective handoff gate smaller than the configured6.76cm.
    plan.poses = {makePose(0, 0, 0), makePose(0.1, 0, 0), makePose(0.2, 0, 0),
      makePose(0.1412, 0, 0), makePose(0.3, 0, 0)};
    handler.setPath(plan);
    auto robot = makePose(0.2359, 0, 0);
    robot.header.frame_id = "odom";
    if (bounded) {
      EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(handler.didHandoff());
      EXPECT_EQ(handler.getActiveCuspDirection(), 1);
    } else {
      EXPECT_NO_THROW(handler.transformPath(robot));
      EXPECT_TRUE(handler.didHandoff());
      EXPECT_EQ(handler.getActiveCuspDirection(), -1);
    }
  }
}

TEST(PathHandlerTests, OptInOverrunCapturePreservesTheOriginalFailure)
{
  const char * previous_env = std::getenv("MPPI_STALL_SNAPSHOT_PATH");
  const std::string previous = previous_env ? previous_env : "";
  const std::string prefix = (std::filesystem::temp_directory_path() /
    ("mppi-overrun-" + std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()))).string();
  const std::string output = prefix + ".cusp_overrun.123000000000.json";
  for (bool capture : {false, true}) {
    if (capture) {setenv("MPPI_STALL_SNAPSHOT_PATH", prefix.c_str(), 1);} else {
      unsetenv("MPPI_STALL_SNAPSHOT_PATH");
    }
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("overrun_capture");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", true);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    plan.poses = {makePose(0, 0, 0), makePose(0.1, 0, 0),
      makePose(0.2, 0, 0), makePose(0.1, 0, 0)};
    handler.setPath(plan);
    auto robot = makePose(0.3, 0, 0);
    robot.header.frame_id = "odom";
    robot.header.stamp.sec = 123;
    EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
    EXPECT_FALSE(handler.didHandoff());
    EXPECT_EQ(handler.getActiveCuspDirection(), 1);
    EXPECT_EQ(std::filesystem::exists(output), capture);
    if (capture && std::filesystem::exists(output)) {
      const char * artifact = std::getenv("CUSP_CAPTURE_TEST_ARTIFACT");
      if (artifact) {
        std::filesystem::copy_file(output, artifact,
          std::filesystem::copy_options::overwrite_existing);
      }
      std::remove(output.c_str());
    }
  }
  if (previous_env) {setenv("MPPI_STALL_SNAPSHOT_PATH", previous.c_str(), 1);} else {
    unsetenv("MPPI_STALL_SNAPSHOT_PATH");
  }
}

TEST(PathHandlerTests, RecordedFarCornerOverrun062401PreservesReverseCusp)
{
  // Exact native capture at48781.45s from run062401, not a nearest-TF join.
  // Rename the common map frame to odom to remove TF from this reduction.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("recorded_overrun");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.replan_on_cusp_overrun", true);
  node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.3);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(200, 200, 0.05, 0.0, -5.0);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan;
  plan.header.frame_id = "odom";
  plan.poses = {
    makePose(4.9168312763534914, 0.017888855653495739, -1.9198620955096646),
    makePose(4.9351930923493228, 0.062218078295472878, -2.0071285406695769),
    makePose(4.9573482432213041, 0.1047778917973119, -2.0943949858294886),
    makePose(4.9831282719419505, 0.14524480983811827, -2.1816614309894007),
    makePose(5.0123372556640788, 0.18331121523726779, -2.2689278761493132),
    makePose(5.0447530264239049, 0.21868658065750424, -2.3561943213092249),
    makePose(5.0801283918441413, 0.25110235141733028, -2.4434607664691375),
    makePose(5.1181947972432909, 0.28031133513945861, -2.5307272116290496),
    makePose(5.0777278792024845, 0.25453069606726331, -2.6179936567889612),
    makePose(5.0351680657006455, 0.23237554519528203, -2.7052601019488738),
    makePose(4.9908388430586683, 0.21401372919945061, -2.7925265471087859),
    makePose(4.9450783460346202, 0.19958562893600629, -2.879792992268698)};
  handler.setPath(plan);
  auto robot = makePose(5.0788774041452056, 0.45567241000911873, -2.1452171929161512);
  robot.header.frame_id = "odom";
  EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getActiveCuspDirection(), -1);
  EXPECT_EQ(handler.getInversionLocale(), 8u);
  EXPECT_EQ(handler.getRealPathPointCount(), 1u);
  ASSERT_EQ(handler.getInvertedPath().poses.size(), 1u);
  EXPECT_DOUBLE_EQ(handler.getInvertedPath().poses.back().pose.position.x, 5.1181947972432909);
  EXPECT_DOUBLE_EQ(handler.getInvertedPath().poses.back().pose.position.y, 0.28031133513945861);
}

TEST(PathHandlerTests, RetainedCuspEdgeDoesNotBypassOverrunStop)
{
  for (int gear : {-1, 1}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("retained_edge_overrun");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", true);
    node->declare_parameter("dummy.preserve_cusp_direction_edge", true);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    plan.poses = {makePose(0, 0, 0), makePose(gear * 0.1, 0, 0),
      makePose(gear * 0.2, 0, 0), makePose(gear * 0.1, 0, 0)};
    handler.setPath(plan);
    auto robot = makePose(gear * 0.3, 0, 0);
    robot.header.frame_id = "odom";
    EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
    EXPECT_FALSE(handler.didHandoff());
    EXPECT_EQ(handler.getActiveCuspDirection(), gear);
    EXPECT_EQ(handler.getRealPathPointCount(), 2u);
  }
}

TEST(PathHandlerTests, StrictOrientationPolicyRejectsRecordedMicroCuspYawResidual)
{
  // Reduce the native 0.0562m / 0.659rad handoff to a straight micro-segment.
  // This tests the gate policy, not whether the vehicle can complete that path.
  for (bool strict : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("micro_yaw_policy");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.inversion_xy_tolerance", 0.0676);
    node->declare_parameter("dummy.inversion_yaw_tolerance", strict ? 0.1 : 0.3);
    node->declare_parameter("dummy.micro_cusp_length_threshold", 0.15);
    node->declare_parameter("dummy.micro_cusp_yaw_scale", strict ? 1.0 : 2.0);
    node->declare_parameter("dummy.handoff_yaw_relax_per_cycle", strict ? 0.0 : 0.005);
    auto cm = makeConfiguredCostmapRos(); ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan; plan.header.frame_id = "odom";
    plan.poses = {makePose(0.144, 0, 0), makePose(0.072, 0, 0),
      makePose(0, 0, 0), makePose(0.1, 0, 0), makePose(0.2, 0, 0)};
    handler.setPath(plan);
    const auto robot = makePose(0.0562, 0, 0.659);
    bool accepted = false;
    for (int cycle = 0; cycle < 200 && !accepted; ++cycle) {
      accepted = handler.isWithinInversionTolerancesWrapper(robot);
    }
    EXPECT_EQ(accepted, !strict);
    if (strict) {
      EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(0.0562, 0, 0.09)));
    }
  }
}

TEST(PathHandlerTests, RecordedCrossTrackResidualDistinguishesPositionGatePolicies)
{
  // Native map-frame capture52515.769s from102732; rename common frame to odom.
  // This is a policy counterfactual, not a navigation success or code-bug claim.
  for (bool strict_position : {true, false}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("recorded_position_policy");
    node->declare_parameter("dummy.enforce_path_inversion", true);
    node->declare_parameter("dummy.replan_on_cusp_overrun", true);
    node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
    node->declare_parameter("dummy.inversion_xy_tolerance", strict_position ? 0.02 : 0.0676);
    node->declare_parameter("dummy.inversion_yaw_tolerance", 0.1);
    node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
    node->declare_parameter("dummy.handoff_yaw_relax_per_cycle", 0.0);
    auto cm = makeConfiguredCostmapRos();
    cm->getCostmap()->resizeMap(200, 200, 0.05, -5.0, -5.0);
    ParametersHandler params(node);
    PathHandlerWrapper handler;
    handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
    nav_msgs::msg::Path plan;
    plan.header.frame_id = "odom";
    plan.poses = {
      makePose(2.9891408897532301, -1.3298760067997364, -2.6179936567889612),
      makePose(2.9465810762513911, -1.3520311576717177, -2.7052601019488738),
      makePose(2.9022521587851884, -1.3703929736675491, -2.7925265471087859),
      makePose(2.8564916617611402, -1.3848210739309934, -2.879792992268698),
      makePose(2.8096477907378699, -1.3952062055348051, -2.9670594374286101),
      makePose(2.7620769910199101, -1.4014690227776327, -3.0543261210071013),
      makePose(2.7141412010757335, -1.4035619182386654, -3.1415925661670134),
      makePose(2.7621224622103, -1.4035619182386654, -3.1415925661670134),
      makePose(2.8101037233448665, -1.4035619182386654, -3.1415925661670134),
      makePose(2.7621224622103, -1.4035619182386654, -3.1415925661670134)};
    handler.setPath(plan);
    auto robot = makePose(2.6930829369297324, -1.4311837786247326, -3.0864583174985976);
    robot.header.frame_id = "odom";
    EXPECT_EQ(handler.isWithinInversionTolerancesWrapper(robot), !strict_position);
    if (strict_position) {
      EXPECT_THROW(handler.transformPath(robot), nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(handler.didHandoff());
      EXPECT_EQ(handler.getActiveCuspDirection(), 1);
    } else {
      EXPECT_NO_THROW(handler.transformPath(robot));
      EXPECT_TRUE(handler.didHandoff());
      EXPECT_EQ(handler.getActiveCuspDirection(), -1);
    }
  }
}


TEST(PathHandlerTests, Recorded161154OverrunKeepsCuspAndRejectsOutsideGate)
{
  // Exact first-overrun snapshot. Ground truth is a separate diagnostic input,
  // never a correction fed into navigation; both poses miss the frozen gate.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("recorded_161154");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.replan_on_cusp_overrun", true);
  node->declare_parameter("dummy.bound_handoff_by_next_segment", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.02);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.04);
  node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
  node->declare_parameter("dummy.handoff_yaw_relax_per_cycle", 0.0);
  auto cm = makeConfiguredCostmapRos();
  cm->getCostmap()->resizeMap(200, 200, 0.05, 0.0, -5.0);
  ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan; plan.header.frame_id = "odom";
  plan.poses = {makePose(4.9203621600636325, 0.7614874718452711, -2.007128540669577),
      makePose(4.898207009191651, 0.7189276583434321, -2.0943949858294886),
      makePose(4.872426980471005, 0.6784607403026257, -2.1816614309894007),
      makePose(4.843217996748876, 0.6403943349034762, -2.2689278761493132),
      makePose(4.81080222598905, 0.6050189694832397, -2.356194321309225),
      makePose(4.775426860568814, 0.5726031987234137, -2.4434607664691375),
      makePose(4.737360455169664, 0.5433942150012854, -2.5307272116290496),
      makePose(4.777827373210471, 0.5691748540734807, -2.6179936567889612),
      makePose(4.82038718671231, 0.591330004945462, -2.705260101948874),
      makePose(4.864716409354287, 0.6096918209412934, -2.792526547108786)};
  handler.setPath(plan);
  auto estimate = makePose(4.725858736393789, 0.5175850481155505, -2.4871507468098675);
  estimate.header.frame_id = "odom";
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(estimate));
  const auto truth_at_overrun = makePose(4.7335934842526814, 0.5221860523226725, -2.4960956911245367);
  EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(truth_at_overrun));
  // Earlier physical sample entered the gate, but no inference is made about
  // a controller evaluation at its timestamp. Test the native predicate only.
  const auto earlier_truth = makePose(4.7376323811336345, 0.5252424328924388, -2.4909190141706086);
  EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(earlier_truth));
  EXPECT_THROW(handler.transformPath(estimate), nav2_core::FailedToMakeProgress);
  EXPECT_FALSE(handler.didHandoff());
  EXPECT_EQ(handler.getActiveCuspDirection(), 1);
}


TEST(PathHandlerTests, OptionalGateTraceRecordsDecisionWithoutChangingIt)
{
  const char * previous_env = std::getenv("MPPI_CUSP_GATE_TRACE_PATH");
  const std::string previous = previous_env ? previous_env : "";
  const std::string output = (std::filesystem::temp_directory_path() /
    ("mppi-gate-" + std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl")).string();
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("gate_trace_test");
  node->declare_parameter("dummy.enforce_path_inversion", true);
  node->declare_parameter("dummy.inversion_xy_tolerance", 0.02);
  node->declare_parameter("dummy.inversion_yaw_tolerance", 0.04);
  node->declare_parameter("dummy.micro_cusp_yaw_scale", 1.0);
  node->declare_parameter("dummy.handoff_yaw_relax_per_cycle", 0.0);
  auto cm = makeConfiguredCostmapRos(); ParametersHandler params(node);
  PathHandlerWrapper handler;
  handler.initialize(node, "dummy", cm, cm->getTfBuffer(), &params);
  nav_msgs::msg::Path plan; plan.header.frame_id = "odom";
  plan.poses = {makePose(0, 0, 0), makePose(0.3, 0, 0), makePose(0.1, 0, 0)};
  handler.setPath(plan);
  for (bool capture : {false, true}) {
    if (capture) {setenv("MPPI_CUSP_GATE_TRACE_PATH", output.c_str(), 1);} else {
      unsetenv("MPPI_CUSP_GATE_TRACE_PATH");
    }
    EXPECT_FALSE(handler.isWithinInversionTolerancesWrapper(makePose(0.27, 0, 0.05)));
    EXPECT_TRUE(handler.isWithinInversionTolerancesWrapper(makePose(0.29, 0, 0.03)));
    EXPECT_EQ(std::filesystem::exists(output), capture);
  }
  std::ifstream file(output);
  std::string contents, line;
  while (std::getline(file, line)) {contents += line + "\n";}
  EXPECT_NE(contents.find("\"accepted\":false"), std::string::npos);
  EXPECT_NE(contents.find("\"accepted\":true"), std::string::npos);
  EXPECT_NE(contents.find("\"blocked_cycles_before\":"), std::string::npos);
  file.close(); std::remove(output.c_str());
  if (previous_env) {setenv("MPPI_CUSP_GATE_TRACE_PATH", previous.c_str(), 1);} else {
    unsetenv("MPPI_CUSP_GATE_TRACE_PATH");
  }
}
