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

#include "gtest/gtest.h"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>

#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_util/geometry_utils.hpp>

#include "nav2_mppi_controller/controller.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"

#include "utils/utils.hpp"

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

// Tests basic transition from configure->active->process->deactive->cleanup

TEST(ControllerStateTransitionTest, ControllerNotFail)
{
  const bool visualize = true;
  TestCostmapSettings costmap_settings{};

  // Node Options
  rclcpp::NodeOptions options;
  std::vector<rclcpp::Parameter> params;
  setUpControllerParams(visualize, params);
  options.parameter_overrides(params);

  auto node = getDummyNode(options);
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap_ros = getDummyCostmapRos(costmap_settings);
  costmap_ros->setRobotFootprint(getDummySquareFootprint(0.01));

  auto controller = getDummyController(node, tf_buffer, costmap_ros);

  TestPose start_pose = costmap_settings.getCenterPose();
  const double path_step = costmap_settings.resolution;
  TestPathSettings path_settings{start_pose, 8, path_step, path_step};

  // evalControl args
  auto pose = getDummyPointStamped(node, start_pose);
  auto velocity = getDummyTwist();
  auto path = getIncrementalDummyPath(node, path_settings);
  path.header.frame_id = costmap_ros->getGlobalFrameID();
  pose.header.frame_id = costmap_ros->getGlobalFrameID();

  controller->setPlan(path);

  EXPECT_NO_THROW(controller->computeVelocityCommands(pose, velocity, {}));

  controller->setSpeedLimit(0.5, true);
  controller->setSpeedLimit(0.5, false);
  controller->setSpeedLimit(1.0, true);
  controller->deactivate();
  controller->cleanup();
}

using namespace mppi;  // NOLINT

class TestableMPPIController : public nav2_mppi_controller::MPPIController
{
public:
  const mppi::Optimizer & getOptimizer() const {return optimizer_;}
  mppi::Optimizer & getOptimizer() {return optimizer_;}
  const mppi::PathHandler & getPathHandler() const {return path_handler_;}
  mppi::PathHandler & getPathHandler() {return path_handler_;}
  mppi::ParametersHandler * getParametersHandler() {return parameters_handler_.get();}
};

TEST(ControllerStateTransitionTest, CuspHeadingGuidanceIntegrationRegression)
{
  TestCostmapSettings costmap_settings{};

  rclcpp::NodeOptions options;
  std::vector<rclcpp::Parameter> params;
  params.emplace_back(rclcpp::Parameter("dummy.visualize", false));
  params.emplace_back(rclcpp::Parameter("controller_frequency", 20.0));
  params.emplace_back(rclcpp::Parameter("dummy.iteration_count", 1));
  params.emplace_back(rclcpp::Parameter("dummy.batch_size", 100));
  params.emplace_back(rclcpp::Parameter("dummy.time_steps", 15));
  params.emplace_back(rclcpp::Parameter("dummy.model_dt", 0.05));
  params.emplace_back(rclcpp::Parameter("dummy.lookahead_dist", 2.0));
  params.emplace_back(rclcpp::Parameter("dummy.motion_model", std::string("DiffDrive")));
  params.emplace_back(rclcpp::Parameter("dummy.critics",
    std::vector<std::string>{"PathAlignCritic"}));

  // PathHandler parameters with inversion and ghost extension enabled
  params.emplace_back(rclcpp::Parameter("dummy.enforce_path_inversion", true));
  params.emplace_back(rclcpp::Parameter("dummy.inversion_xy_tolerance", 0.20));
  params.emplace_back(rclcpp::Parameter("dummy.inversion_yaw_tolerance", 0.40));
  params.emplace_back(rclcpp::Parameter("dummy.min_inversion_horizon", 0.50));
  params.emplace_back(rclcpp::Parameter("dummy.ghost_max_extension", 0.35));
  params.emplace_back(rclcpp::Parameter("dummy.ghost_point_spacing", 0.05));
  params.emplace_back(rclcpp::Parameter("dummy.ghost_min_seed_points", 2));

  // PathAlignCritic parameters with cusp terminal heading guidance
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.enabled", true));
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.weight", 10.0));
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.cost_power", 1));
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.use_path_orientations", true));
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.threshold_to_consider", 0.05));
  params.emplace_back(rclcpp::Parameter("dummy.PathAlignCritic.cusp_terminal_heading_weight",
    30.0));
  options.parameter_overrides(params);

  auto node = getDummyNode(options, "dummy");
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap_ros = getDummyCostmapRos(costmap_settings);
  costmap_ros->setRobotFootprint(getDummySquareFootprint(0.10));

  auto controller = std::make_shared<TestableMPPIController>();
  std::weak_ptr<rclcpp_lifecycle::LifecycleNode> weak_ptr_node{node};
  controller->configure(weak_ptr_node, node->get_name(), tf_buffer, costmap_ros);
  controller->activate();

  auto make_pose = [&](double x, double y, double yaw) {
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = costmap_ros->getGlobalFrameID();
      p.header.stamp = node->now();
      p.pose.position.x = x;
      p.pose.position.y = y;
      p.pose.position.z = 0.0;
      p.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(yaw);
      return p;
    };

  // 1. Multi-segment path: Segment 1 (forward, 3 real points) -> Segment 2 (reverse)
  nav_msgs::msg::Path path;
  path.header.frame_id = costmap_ros->getGlobalFrameID();
  path.header.stamp = node->now();
  path.poses.push_back(make_pose(0.00, 0.0, 0.0));
  path.poses.push_back(make_pose(0.05, 0.0, 0.0));
  path.poses.push_back(make_pose(0.10, 0.0, 0.0));  // Cusp at (0.10, 0)
  path.poses.push_back(make_pose(0.05, 0.0, 0.0));  // Reversing segment
  path.poses.push_back(make_pose(0.00, 0.0, 0.0));
  path.poses.push_back(make_pose(-0.05, 0.0, 0.0));

  controller->setPlan(path);

  auto robot_pose = make_pose(0.00, 0.0, 0.0);
  geometry_msgs::msg::Twist robot_speed;

  EXPECT_NO_THROW(controller->computeVelocityCommands(robot_pose, robot_speed, nullptr));

  // Verify PathHandler captures real_path_point_count before ghost extension
  EXPECT_EQ(controller->getPathHandler().getRealPathPointCount(), 3u);
  EXPECT_EQ(controller->getPathHandler().getActiveCuspDirection(), 1);

  // Verify Controller wired setters to Optimizer
  EXPECT_EQ(controller->getOptimizer().getRealPathPointCount(), std::optional<size_t>(3u));
  EXPECT_EQ(controller->getOptimizer().getActiveCuspDirection(), 1);

  // Verify Optimizer populated CriticData
  const auto & critic_data = controller->getOptimizer().getCriticData();
  EXPECT_EQ(critic_data.real_path_point_count, std::optional<size_t>(3u));
  EXPECT_EQ(critic_data.active_cusp_direction, 1);

  // Verify that ghost extension occurred: total path points inside optimizer > 4
  const size_t total_transformed_pts = critic_data.path.yaws.shape(0);
  EXPECT_GT(total_transformed_pts, 4u);

  // 2. Behavioral verification: Test PathAlignCritic on this CriticData
  // Create test trajectories:
  // Trajectory 0: terminal yaw aligned with cusp (0.0 rad)
  // Trajectory 1: terminal yaw misaligned with cusp (M_PI_2 rad)
  mppi::critics::PathAlignCritic pac;
  pac.on_configure(node, "dummy", "dummy.PathAlignCritic", costmap_ros,
    controller->getParametersHandler());

  models::Trajectories test_trajectories;
  test_trajectories.reset(2, 10);
  test_trajectories.x.fill(0.10f);
  test_trajectories.y.fill(0.0f);
  test_trajectories.yaws.fill(0.0f);
  for (unsigned int t = 0; t < 10; ++t) {
    test_trajectories.yaws(1, t) = static_cast<float>(M_PI_2);
  }
  xt::xtensor<float, 1> test_costs = xt::zeros<float>({2});
  CriticData test_data = {
    critic_data.state,
    test_trajectories,
    critic_data.path,
    critic_data.goal,
    test_costs,
    critic_data.model_dt,
    critic_data.fail_flag,
    critic_data.goal_checker,
    critic_data.motion_model,
    critic_data.path_pts_valid,
    critic_data.furthest_reached_path_point,
    critic_data.real_path_point_count,
    critic_data.active_cusp_direction
  };

  // With real_path_point_count = 3 (our fix): cusp_terminal_heading_weight activates!
  pac.score(test_data);
  EXPECT_GT(test_data.costs(1), test_data.costs(0));
  EXPECT_GT(test_data.costs(1), 20.0f);  // 30.0 * M_PI_2 ~= 47.1

  // Negative behavioral check: Without our fix (simulated by nullopt or real count > 4),
  // ghost extension disables cusp terminal heading guidance because real_cusp_pts > 4.
  xt::xtensor<float, 1> unfixed_costs = xt::zeros<float>({2});
  CriticData unfixed_data = {
    critic_data.state,
    test_trajectories,
    critic_data.path,
    critic_data.goal,
    unfixed_costs,
    critic_data.model_dt,
    critic_data.fail_flag,
    critic_data.goal_checker,
    critic_data.motion_model,
    critic_data.path_pts_valid,
    critic_data.furthest_reached_path_point,
    std::nullopt,
    critic_data.active_cusp_direction
  };
  pac.score(unfixed_data);
  // Without real_path_point_count, the 47.1 terminal heading penalty is lost!
  EXPECT_LT(unfixed_data.costs(1), test_data.costs(1) - 20.0f);

  // 3. Final-segment negative case:
  // Single-segment path has no direction cusps (inversion_locale_ == 0)
  nav_msgs::msg::Path final_path;
  final_path.header = path.header;
  final_path.poses.push_back(make_pose(0.00, 0.0, 0.0));
  final_path.poses.push_back(make_pose(0.05, 0.0, 0.0));
  final_path.poses.push_back(make_pose(0.10, 0.0, 0.0));
  controller->setPlan(final_path);

  EXPECT_NO_THROW(controller->computeVelocityCommands(robot_pose, robot_speed, nullptr));
  EXPECT_EQ(controller->getPathHandler().getActiveCuspDirection(), 0);
  EXPECT_EQ(controller->getOptimizer().getActiveCuspDirection(), 0);
  EXPECT_EQ(controller->getOptimizer().getCriticData().active_cusp_direction, 0);

  // 4. Cusp Handoff / Reset case:
  // Reset the multi-segment plan, then advance robot to the cusp point
  controller->setPlan(path);
  auto cusp_pose = make_pose(0.10, 0.0, 0.0);
  EXPECT_NO_THROW(controller->computeVelocityCommands(cusp_pose, robot_speed, nullptr));

  // The handoff occurred
  EXPECT_TRUE(controller->getPathHandler().didHandoff());
  // After handoff, segment 2 is now active. Since segment 2 is the final segment,
  // active_cusp_direction is updated to 0
  EXPECT_EQ(controller->getPathHandler().getActiveCuspDirection(), 0);
  EXPECT_EQ(controller->getOptimizer().getActiveCuspDirection(), 0);

  controller->deactivate();
  controller->cleanup();
}
