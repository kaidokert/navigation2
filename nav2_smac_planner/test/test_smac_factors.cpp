// Copyright (c) 2026, Kaido Kert
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

// Tests for the two opt-in Smac hybrid factors: rear_axle_offset (plan for the
// rear-axle point) and analytic_score: length_direction (score analytic shots by
// true length with search-primitive weights). Both default to stock behaviour.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/utils.h"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/analytic_expansion.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/smac_planner_hybrid.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

namespace
{

nav2_costmap_2d::Footprint boxFootprint()
{
  nav2_costmap_2d::Footprint footprint(4);
  footprint[0].x = 0.3; footprint[0].y = 0.15;
  footprint[1].x = 0.3; footprint[1].y = -0.15;
  footprint[2].x = -0.2; footprint[2].y = -0.15;
  footprint[3].x = -0.2; footprint[3].y = 0.15;
  return footprint;
}

struct TraceStats
{
  double max_lateral_ratio{0.0};
  int reverse_steps{0};
  int steps{0};
};

// Kinematic consistency of the point `offset` metres ahead of each pose along its heading:
// for a point with no lateral velocity, each step's chord is aligned with the mean heading.
TraceStats traceStats(const nav_msgs::msg::Path & plan, double offset)
{
  TraceStats stats;
  for (size_t i = 0; i + 1 < plan.poses.size(); ++i) {
    const double h0 = tf2::getYaw(plan.poses[i].pose.orientation);
    const double h1 = tf2::getYaw(plan.poses[i + 1].pose.orientation);
    const double ax = plan.poses[i].pose.position.x + offset * cos(h0);
    const double ay = plan.poses[i].pose.position.y + offset * sin(h0);
    const double bx = plan.poses[i + 1].pose.position.x + offset * cos(h1);
    const double by = plan.poses[i + 1].pose.position.y + offset * sin(h1);
    const double dx = bx - ax;
    const double dy = by - ay;
    const double length = std::hypot(dx, dy);
    if (length < 1e-6) {
      continue;
    }
    const double hm = h0 + 0.5 * std::atan2(sin(h1 - h0), cos(h1 - h0));
    const double along = dx * cos(hm) + dy * sin(hm);
    const double lateral = -dx * sin(hm) + dy * cos(hm);
    stats.max_lateral_ratio = std::max(stats.max_lateral_ratio, std::fabs(lateral) / length);
    stats.steps++;
    if (along < 0.0) {
      stats.reverse_steps++;
    }
  }
  return stats;
}

geometry_msgs::msg::PoseStamped makePose(double x, double y, double yaw)
{
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.position.x = x;
  pose.pose.position.y = y;
  pose.pose.orientation.z = sin(yaw / 2.0);
  pose.pose.orientation.w = cos(yaw / 2.0);
  return pose;
}

class FactorPlanner : public nav2_smac_planner::SmacPlannerHybrid
{
public:
  using nav2_smac_planner::SmacPlannerHybrid::dynamicParametersCallback;
};

}  // namespace

TEST(SmacFactors, reference_offset_checks_footprint_ahead_of_planned_point)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsA");
  nav2_costmap_2d::Costmap2D raw(100, 100, 0.05, 0.0, 0.0, 0);
  for (unsigned int i = 50; i < 53; ++i) {
    for (unsigned int j = 40; j < 60; ++j) {
      raw.setCost(i, j, 254);
    }
  }
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  *costmap_ros->getCostmap() = raw;

  nav2_smac_planner::GridCollisionChecker shifted(costmap_ros, 72, node);
  nav2_smac_planner::GridCollisionChecker plain(costmap_ros, 72, node);
  shifted.setFootprint(boxFootprint(), false, 0.0);
  plain.setFootprint(boxFootprint(), false, 0.0);
  const float offset = 4.0f;  // cells
  shifted.setReferenceOffset(offset);

  int collisions = 0;
  for (unsigned int bin = 0; bin < 72; bin += 9) {
    const float angle = bin * 2.0f * static_cast<float>(M_PI) / 72.0f;
    for (float x = 30.0f; x <= 70.0f; x += 3.0f) {
      for (float y = 35.0f; y <= 65.0f; y += 5.0f) {
        const bool a = shifted.inCollision(x, y, static_cast<float>(bin), false);
        const bool b = plain.inCollision(
          x + offset * std::cos(angle), y + offset * std::sin(angle), static_cast<float>(bin), false);
        EXPECT_EQ(a, b) << "bin " << bin << " x " << x << " y " << y;
        EXPECT_FLOAT_EQ(shifted.getCost(), plain.getCost());
        collisions += a ? 1 : 0;
      }
    }
  }
  EXPECT_GT(collisions, 0);

  // Offset 0 restores the stock check
  shifted.setReferenceOffset(0.0f);
  EXPECT_EQ(
    shifted.inCollision(46.0f, 50.0f, 0.0f, false),
    plain.inCollision(46.0f, 50.0f, 0.0f, false));
}

TEST(SmacFactors, length_direction_score_weights_reverse_and_cusps)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsB");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 2.0;
  info.non_straight_penalty = 1.01;
  info.reverse_penalty = 3.6;
  info.minimum_turning_radius = 10;
  info.cost_penalty = 3.5;
  info.analytic_score_length_direction = true;
  unsigned int size_x = 50;
  unsigned int size_y = 50;
  unsigned int size_theta = 72;
  nav2_smac_planner::NodeHybrid::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_theta, info);

  nav2_costmap_2d::Costmap2D raw(size_x, size_y, 0.05, 0.0, 0.0, 0);
  for (unsigned int i = 0; i < size_x; ++i) {
    raw.setCost(i, 20, 126);  // normalized cost 0.5 on row y = 20
  }
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  *costmap_ros->getCostmap() = raw;
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, 72, node);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  using Expansion = nav2_smac_planner::AnalyticExpansion<nav2_smac_planner::NodeHybrid>;
  Expansion expander(nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, 72);

  std::vector<std::unique_ptr<nav2_smac_planner::NodeHybrid>> nodes;
  auto build = [&](const std::vector<float> & xs, float y = 10.0f) {
      Expansion::AnalyticExpansionNodes expansion;
      for (float x : xs) {
        nodes.push_back(std::make_unique<nav2_smac_planner::NodeHybrid>(0));
        nav2_smac_planner::NodeHybrid * n = nodes.back().get();
        n->setPose(nav2_smac_planner::NodeHybrid::Coordinates(x, y, 0.0f));
        EXPECT_TRUE(n->isNodeValid(false, &checker));
        nav2_smac_planner::NodeHybrid::Coordinates coords(x, y, 0.0f);
        expansion.emplace_back(n, coords, coords);
      }
      return expansion;
    };

  // Heading is bin 0 (+x) throughout, free space (cost 0)
  EXPECT_NEAR(expander.lengthDirectionScore(build({10, 11, 12, 13})), 3.0f, 1e-4);
  EXPECT_NEAR(expander.lengthDirectionScore(build({13, 12, 11, 10})), 3.0f * 3.6f, 1e-4);
  // forward 2, then reverse 1: the first reverse step carries the change penalty
  EXPECT_NEAR(
    expander.lengthDirectionScore(build({10, 11, 12, 11})),
    2.0f + 3.6f * (1.0f + 2.0f), 1e-4);
  // uneven spacing is measured, not first-step chord x count
  EXPECT_NEAR(expander.lengthDirectionScore(build({10, 10.5f, 13})), 3.0f, 1e-4);
  // cost penalty, linear: each step x (1 + 3.5 * 0.5)
  EXPECT_NEAR(expander.lengthDirectionScore(build({10, 11, 12}, 20.0f)), 2.0f * 2.75f, 1e-3);

  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(SmacFactors, rear_axle_offset_plans_for_the_rear_axle)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsPlanner");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  nav2_costmap_2d::Costmap2D raw(100, 100, 0.05, 0.0, 0.0, 0);
  *costmap_ros->getCostmap() = raw;

  node->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  node->declare_parameter("test.minimum_turning_radius", 0.5);
  node->declare_parameter("test.downsample_costmap", false);
  node->declare_parameter("test.smooth_path", false);
  node->declare_parameter("test.analytic_expansion_max_length", 3.0);
  node->declare_parameter("test.rear_axle_offset", 0.2);
  node->declare_parameter("test.analytic_score", std::string("length_direction"));

  auto planner = std::make_unique<FactorPlanner>();
  planner->configure(node, "test", nullptr, costmap_ros);
  planner->activate();
  auto no_cancel = []() {return false;};

  // A 180 about the rear axle: rear (1.3, 1.5, 0) -> (1.3, 2.5, pi), radius 0.5.
  const double d = 0.2;
  const auto start = makePose(1.5, 1.5, 0.0);
  const auto goal = makePose(1.3 - d, 2.5, M_PI);
  nav_msgs::msg::Path with_axle = planner->createPlan(start, goal, no_cancel);
  ASSERT_GT(with_axle.poses.size(), 10u);
  EXPECT_NEAR(with_axle.poses.front().pose.position.x, 1.5, 0.06);
  EXPECT_NEAR(with_axle.poses.front().pose.position.y, 1.5, 0.06);
  EXPECT_NEAR(with_axle.poses.back().pose.position.x, 1.3 - d, 0.06);
  EXPECT_NEAR(with_axle.poses.back().pose.position.y, 2.5, 0.06);
  const TraceStats rear_on = traceStats(with_axle, -d);
  const TraceStats base_on = traceStats(with_axle, 0.0);
  EXPECT_LT(rear_on.max_lateral_ratio, 0.12);  // heading bins are 5 degrees
  EXPECT_GT(base_on.max_lateral_ratio, 0.25);  // d / R = 0.4 on the arc
  // One gear throughout. At exactly 2R the forward L(pi) and the reverse R(pi) semicircles tie in
  // length, so OMPL's word order picks the gear; either is the same rear-axle arc.
  EXPECT_TRUE(rear_on.reverse_steps == 0 || rear_on.reverse_steps == rear_on.steps)
    << rear_on.reverse_steps << " of " << rear_on.steps;

  // Back to stock: the returned path is kinematically consistent for the robot frame instead
  auto results = planner->dynamicParametersCallback(
    {rclcpp::Parameter("test.rear_axle_offset", 0.0)});
  EXPECT_TRUE(results.successful);
  nav_msgs::msg::Path stock = planner->createPlan(start, goal, no_cancel);
  ASSERT_GT(stock.poses.size(), 10u);
  EXPECT_LT(traceStats(stock, 0.0).max_lateral_ratio, 0.12);
  EXPECT_GT(traceStats(stock, -d).max_lateral_ratio, 0.25);

  // Rear axles in the same cell but robot frames 2d apart: no single-pose shortcut
  const auto turned = makePose(1.5 - 2.0 * d, 1.5, M_PI);
  planner->dynamicParametersCallback({rclcpp::Parameter("test.rear_axle_offset", d)});
  try {
    EXPECT_GT(planner->createPlan(start, turned, no_cancel).poses.size(), 1u);
  } catch (const std::exception &) {
    // no path is acceptable; a one-pose plan 2d away from the goal is not
  }
  planner->dynamicParametersCallback({rclcpp::Parameter("test.rear_axle_offset", 0.0)});

  // An unknown score name is ignored (warning) and planning still works
  planner->dynamicParametersCallback(
    {rclcpp::Parameter("test.analytic_score", std::string("bogus"))});
  EXPECT_GT(planner->createPlan(start, goal, no_cancel).poses.size(), 10u);

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}
