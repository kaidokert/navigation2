// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
// Isolated stationary synthetic-state test: never publishes motor commands.
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <xtensor/xrandom.hpp>
#include <iostream>
#include <vector>
#include "utils/utils.hpp"
#include "nav2_util/control_timing.hpp"

TEST(ControlTimingBenchmark, AlternatingEnabledDisabledRealMPPI)
{
  rclcpp::init(0, nullptr);
  TestCostmapSettings settings{};
  auto costmap = getDummyCostmapRos(settings);
  costmap->setRobotFootprint(getDummySquareFootprint(0.01));
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
    options.parameter_overrides(params);
    auto node = getDummyNode(options);
    auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    xt::random::seed(42);
    auto controller = getDummyController(node, tf, costmap);
    nav2_util::ControlTiming timing;
    timing.configure(node, "~/control_timing", .05);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
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
        cycle.outcome("completed");
      }
      // Include aggregation/serialization when timer fires, as real overhead.
      executor.spin_some();
      const double elapsed = nav2_util::ControlTiming::seconds(
        nav2_util::ControlTiming::Clock::now() - start);
      if (i >= 10) {measurements[enabled].push_back(elapsed);}
    }
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
  rclcpp::shutdown();
}
