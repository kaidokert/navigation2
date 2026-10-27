// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
// Isolated stationary synthetic-state test: never publishes motor commands.
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <thread>
#include <xtensor/xrandom.hpp>
#include "utils/utils.hpp"
#include "nav2_util/control_timing.hpp"

TEST(ControlTimingBenchmark, AlternatingEnabledDisabledRealMPPI)
{
  rclcpp::init(0, nullptr);
  TestCostmapSettings settings{};
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "timing_map", "", "timing_map", true);
  costmap->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{"inflation_layer"}));
  costmap->declare_parameter("inflation_layer.plugin",
    std::string("nav2_costmap_2d::InflationLayer"));
  costmap->declare_parameter("inflation_layer.inflation_radius", 0.25);
  costmap->declare_parameter("inflation_layer.cost_scaling_factor", 6.0);
  costmap->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[0.170,0.151],[0.170,-0.151],[-0.156,-0.151],[-0.156,0.151]]")));
  costmap->set_parameter(rclcpp::Parameter("footprint_padding", 0.012));
  ASSERT_EQ(costmap->on_configure(rclcpp_lifecycle::State{}), nav2_util::CallbackReturn::SUCCESS);
  costmap->getLayeredCostmap()->resizeMap(
    settings.cells_x, settings.cells_y, settings.resolution, settings.origin_x, settings.origin_y);
  auto * grid = costmap->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + settings.cells_x * settings.cells_y, 0);
  std::vector<double> measurements[2];
  std::vector<std::array<double, 2>> reference_commands;
  for (int block = 0; block < 4; ++block) {
    const bool enabled = (block == 1 || block == 2);  // ABBA order
    rclcpp::NodeOptions options;
    std::vector<rclcpp::Parameter> params;
    params.emplace_back("dummy.visualize", false);
    params.emplace_back("controller_frequency", 20.0);
    params.emplace_back("timing_enabled", enabled);
    params.emplace_back("dummy.batch_size", 1000);
    params.emplace_back("dummy.time_steps", 41);
    params.emplace_back("dummy.regenerate_noises", false);
    params.emplace_back("dummy.motion_model", "Ackermann");
    params.emplace_back("dummy.critics", std::vector<std::string>{
      "PathFollowCritic", "PathAlignCritic", "ObstaclesCritic"});
    params.emplace_back("dummy.ObstaclesCritic.consider_footprint", true);
    params.emplace_back("dummy.ObstaclesCritic.collision_check_footprint_only", true);
    options.parameter_overrides(params);
    auto node = getDummyNode(options);
    auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    xt::random::seed(42);
    auto controller = getDummyController(node, tf, costmap);
    nav2_util::ControlTiming timing;
    timing.configure(node, "~/control_timing", .05);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    size_t published_windows = 0;
    auto sub = node->create_subscription<std_msgs::msg::String>(
      "~/control_timing", 4,
      [&](std_msgs::msg::String::ConstSharedPtr) {++published_windows;});
    auto pose = getDummyPointStamped(node, settings.getCenterPose());
    auto velocity = getDummyTwist();
    TestPathSettings path_settings{settings.getCenterPose(), 20, settings.resolution, 0};
    auto path = getIncrementalDummyPath(node, path_settings);
    path.header.frame_id = costmap->getGlobalFrameID();
    pose.header.frame_id = path.header.frame_id;
    controller->setPlan(path);
    for (int i = 0; i < 110; ++i) {
      const auto start = nav2_util::ControlTiming::Clock::now();
      {
        nav2_util::ControlTiming::Cycle cycle(timing);
        auto command = controller->computeVelocityCommands(pose, velocity, {});
        if (block == 0) {
          reference_commands.push_back({command.twist.linear.x, command.twist.angular.z});
        } else {
          EXPECT_DOUBLE_EQ(command.twist.linear.x, reference_commands[i][0]);
          EXPECT_DOUBLE_EQ(command.twist.angular.z, reference_commands[i][1]);
        }
        cycle.outcome(nav2_util::ControlTiming::Outcome::Completed);
      }
      // Include aggregation/serialization when timer fires, as real overhead.
      executor.spin_some();
      const double elapsed = nav2_util::ControlTiming::seconds(
        nav2_util::ControlTiming::Clock::now() - start);
      if (i >= 10) {measurements[enabled].push_back(elapsed);}
      std::this_thread::sleep_until(start + std::chrono::milliseconds(50));
    }
    if (enabled) {EXPECT_GE(published_windows, 3u);}
    timing.cleanup();
    controller->deactivate();
    controller->cleanup();
    executor.remove_node(node->get_node_base_interface());
  }
  for (int arm = 0; arm < 2; ++arm) {
    auto & v = measurements[arm];
    std::sort(v.begin(), v.end());
    ASSERT_EQ(v.size(), 200u);
    std::cout << "TIMING_BENCH enabled=" << arm << " samples=" << v.size()
              << " p50=" << v[99] << " p95=" << v[189]
              << " p99=" << v[197] << " max=" << v.back() << std::endl;
  }
  costmap->on_cleanup(rclcpp_lifecycle::State{});
  costmap.reset();
  rclcpp::shutdown();
}
