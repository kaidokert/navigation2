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

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
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
};

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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
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
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
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
  node->declare_parameter("dummy.ghost_activation_max_cusp_distance", rclcpp::ParameterValue(1.0));
  node->declare_parameter("dummy.ghost_min_seed_arc_length", rclcpp::ParameterValue(0.01));
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

TEST(PathHandlerTests, GhostPathDoesNotMutateGlobalPlanOrInversionLocale)
{
  PathHandlerWrapper handler;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("ghost_state_node");
  node->declare_parameter("dummy.max_robot_pose_search_dist", rclcpp::ParameterValue(99999.9));
  node->declare_parameter("dummy.enforce_path_inversion", rclcpp::ParameterValue(true));
  node->declare_parameter("dummy.inversion_xy_tolerance", rclcpp::ParameterValue(0.01));
  node->declare_parameter("dummy.min_inversion_horizon", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_max_extension", rclcpp::ParameterValue(0.20));
  node->declare_parameter("dummy.ghost_activation_max_cusp_distance", rclcpp::ParameterValue(1.0));
  node->declare_parameter("dummy.ghost_min_seed_arc_length", rclcpp::ParameterValue(0.01));
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
  node->declare_parameter("dummy.ghost_activation_max_cusp_distance", rclcpp::ParameterValue(1.0));
  node->declare_parameter("dummy.ghost_min_seed_arc_length", rclcpp::ParameterValue(0.01));
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
