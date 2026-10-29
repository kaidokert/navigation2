// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <regex>
#include <limits>
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
    ControlTiming::stageCurrent(nav2_util::ControlTiming::Stage::Optimizer, start);
    cycle.published();
    cycle.outcome(nav2_util::ControlTiming::Outcome::Completed);
    cycle.endMeasurement();
    cycle.missed(true);
  }
  EXPECT_EQ(ControlTiming::currentCycle(), nullptr);
  timing.setBudget(1.0);
  {
    ControlTiming::Cycle cycle(timing);
    std::this_thread::sleep_for(3ms);
    cycle.published();
    cycle.outcome(nav2_util::ControlTiming::Outcome::Completed);
  }
  {
    ControlTiming::Cycle cycle(timing);
    cycle.outcome(nav2_util::ControlTiming::Outcome::Cancelled);
  }
  {ControlTiming::Cycle cycle(timing);}  // Default exception outcome.
  {
    ControlTiming::Cycle cycle(timing);
    cycle.outcome(nav2_util::ControlTiming::Outcome::Inactive);
  }
  EXPECT_TRUE(messages.empty());
  const auto deadline = ControlTiming::Clock::now() + 3s;
  while (messages.size() < 2 && ControlTiming::Clock::now() < deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  ASSERT_GE(messages.size(), 2u);
  EXPECT_TRUE(std::regex_search(messages[0], std::regex(
      R"("thread_id":[0-9]+,"samples":5,"dropped_samples":0)")));
  EXPECT_NE(messages[0].find("\"completed\":2"), std::string::npos);
  EXPECT_NE(messages[0].find("\"cancelled\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"exceptions\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"inactive\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"scheduler_misses\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"optimizer\":{\"samples\":1"), std::string::npos);
  EXPECT_NE(messages[0].find("\"deadline_misses\":1,"), std::string::npos);
  EXPECT_NE(messages[0].find("\"deadline_budget_seconds\":null"), std::string::npos);
  EXPECT_NE(messages[0].find("\"publication_interval\":{\"samples\":1,"), std::string::npos);
  EXPECT_TRUE(std::regex_search(messages[1], std::regex(
      R"("thread_id":0,"samples":0,"dropped_samples":0)")));
  timing.cleanup();
  executor.remove_node(node);
  rclcpp::shutdown();
}

TEST(ControlTiming, DisabledHasNoPublisher)
{
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("timing_disabled_test");
  ControlTiming timing;
  timing.configure(node, "~/control_timing", .05);
  EXPECT_FALSE(timing.enabled());
  EXPECT_THROW(timing.setBudget(0), std::invalid_argument);
  EXPECT_THROW(timing.setBudget(std::numeric_limits<double>::infinity()), std::invalid_argument);
  {ControlTiming::Cycle cycle(timing); cycle.published();}
  EXPECT_EQ(node->count_publishers("/timing_disabled_test/control_timing"), 0u);
  rclcpp::shutdown();
}
