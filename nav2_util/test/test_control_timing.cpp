// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>
#include "nav2_util/control_timing.hpp"

using nav2_util::ControlTiming;
using namespace std::chrono_literals;

TEST(ControlTiming, AggregatesStagesOutcomesAndIdleWithoutPublishingPerCycle)
{
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("timing_test");
  node->declare_parameter("timing_enabled", true);
  std::vector<std::string> messages;
  auto sub = node->create_subscription<std_msgs::msg::String>(
    "~/control_timing", 4, [&](std_msgs::msg::String::ConstSharedPtr msg) {
      messages.push_back(msg->data);
    });
  ControlTiming timing;
  timing.configure(node, "~/control_timing", .001);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::this_thread::sleep_for(100ms);
  {
    ControlTiming::Cycle cycle(timing);
    EXPECT_EQ(ControlTiming::currentCycle(), &cycle);
    auto start = ControlTiming::Clock::now();
    std::this_thread::sleep_for(3ms);
    ControlTiming::stageCurrent(6, start);
    cycle.published();
    cycle.outcome("completed");
    cycle.endMeasurement();
    cycle.missed(true);
  }
  EXPECT_EQ(ControlTiming::currentCycle(), nullptr);
  {
    ControlTiming::Cycle cycle(timing);
    cycle.outcome("cancelled");
  }
  {ControlTiming::Cycle cycle(timing);}  // Default exception outcome.
  EXPECT_TRUE(messages.empty());
  const auto deadline = ControlTiming::Clock::now() + 3s;
  while (messages.size() < 2 && ControlTiming::Clock::now() < deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  ASSERT_GE(messages.size(), 2u);
  EXPECT_NE(messages[0].find("\"samples\":3"), std::string::npos);
  EXPECT_NE(messages[0].find("\"completed\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"cancelled\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"exceptions\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"scheduler_misses\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"optimizer\":{\"samples\":1"), std::string::npos);
  EXPECT_NE(messages[1].find("\"samples\":0"), std::string::npos);
  timing.cleanup();
  executor.remove_node(node);
  rclcpp::shutdown();
}

TEST(ControlTiming, DisabledHasNoPublisherOrTimer)
{
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("timing_disabled_test");
  ControlTiming timing;
  timing.configure(node, "~/control_timing", .05);
  EXPECT_FALSE(timing.enabled());
  {ControlTiming::Cycle cycle(timing); cycle.published();}
  EXPECT_EQ(node->count_publishers("/timing_disabled_test/control_timing"), 0u);
  rclcpp::shutdown();
}
