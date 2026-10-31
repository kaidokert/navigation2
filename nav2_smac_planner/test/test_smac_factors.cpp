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
// rear-axle point with exact unquantized start/goal heading continuity) and
// analytic_score: length_direction / analytic_directional_scoring: true (score
// analytic shots by true length with search-primitive weights).

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
  int gear_changes{0};
  int steps{0};
};

// Kinematic consistency of the point `offset` metres ahead of each pose along its heading:
// for a point with no lateral velocity, each step's chord is aligned with the mean heading.
TraceStats traceStats(const nav_msgs::msg::Path & plan, double offset)
{
  TraceStats stats;
  int prev_gear = 0;
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
    const int gear = (along < 0.0) ? -1 : 1;
    if (gear < 0) {
      stats.reverse_steps++;
    }
    if (prev_gear != 0 && gear != prev_gear) {
      stats.gear_changes++;
    }
    prev_gear = gear;
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
          x + offset * std::cos(angle),
          y + offset * std::sin(angle),
          static_cast<float>(bin), false);
        EXPECT_EQ(a, b);
        if (a) {
          collisions++;
        }
      }
    }
  }
  EXPECT_GT(collisions, 0);

  shifted.setReferenceOffset(0.0f);
  EXPECT_EQ(
    shifted.inCollision(48.0f, 50.0f, 0.0f, false),
    plain.inCollision(48.0f, 50.0f, 0.0f, false));
}

TEST(SmacFactors, length_direction_score_weights_reverse_and_cusps)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsB");
  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 14.0f;
  info.non_straight_penalty = 1.05f;
  info.change_penalty = 2.0f;
  info.reverse_penalty = 3.6f;
  info.cost_penalty = 3.5f;
  info.retrospective_penalty = 0.0f;
  info.analytic_expansion_ratio = 2.0f;
  info.analytic_expansion_max_length = 75.0f;
  info.analytic_expansion_max_cost = 252.0f;
  info.use_quadratic_cost_penalty = false;
  info.analytic_score_length_direction = true;

  unsigned int size_x = 100, size_y = 100, size_theta = 72;
  nav2_smac_planner::NodeHybrid::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_theta, info);

  nav2_costmap_2d::Costmap2D raw(100, 100, 0.05, 0.0, 0.0, 0);
  for (unsigned int x = 0; x < 100; ++x) {
    raw.setCost(x, 20, 126);  // normalized cost = 0.5
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
  // start_coords scores the initial step from the expansion origin into expansion[0]
  nav2_smac_planner::NodeHybrid::Coordinates start_at_10(10.0f, 10.0f, 0.0f);
  EXPECT_NEAR(
    expander.lengthDirectionScore(build({9, 10, 11}), &start_at_10),
    3.6f + 1.0f * (1.0f + 2.0f) + 1.0f, 1e-4);

  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(SmacFactors, rear_axle_offset_and_off_bin_start_yaw_have_zero_endpoint_sliver)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsPlanner");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  nav2_costmap_2d::Costmap2D raw(150, 150, 0.04, 0.0, 0.0, 0);
  *costmap_ros->getCostmap() = raw;

  node->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  node->declare_parameter("test.minimum_turning_radius", 0.56);
  node->declare_parameter("test.reverse_penalty", 3.6);
  node->declare_parameter("test.change_penalty", 2.0);
  node->declare_parameter("test.downsample_costmap", false);
  node->declare_parameter("test.smooth_path", false);
  node->declare_parameter("test.analytic_expansion_max_length", 3.0);
  node->declare_parameter("test.rear_axle_offset", 0.10636);
  node->declare_parameter("test.analytic_directional_scoring", true);

  auto planner = std::make_unique<FactorPlanner>();
  planner->configure(node, "test", nullptr, costmap_ros);
  planner->activate();
  auto no_cancel = []() {return false;};

  // Off-bin start yaw (1.535 rad ~= 87.95 deg, midway between 5-deg bins 17 and 18)
  // and feasible 180-deg turn at R_goal = 0.60 m around the rear axle (d = 0.10636 m).
  const double d = 0.10636;
  const double r_goal = 0.60;
  const double start_yaw = 1.535;
  const double goal_yaw = start_yaw - M_PI;
  const double rx0 = 3.0;
  const double ry0 = 3.0;
  const double sx = rx0 + d * std::cos(start_yaw);
  const double sy = ry0 + d * std::sin(start_yaw);
  // Left 180 turn of radius r_goal about rear axle: rear axle moves by 2 * r_goal along (-sin, cos)
  const double rx1 = rx0 - 2.0 * r_goal * std::sin(start_yaw);
  const double ry1 = ry0 + 2.0 * r_goal * std::cos(start_yaw);
  const double gx = rx1 + d * std::cos(goal_yaw);
  const double gy = ry1 + d * std::sin(goal_yaw);

  const auto start = makePose(sx, sy, start_yaw);
  const auto goal = makePose(gx, gy, goal_yaw);
  nav_msgs::msg::Path plan = planner->createPlan(start, goal, no_cancel);
  ASSERT_GT(plan.poses.size(), 10u);

  // Exact start and goal pose continuity (zero yaw-bin endpoint sliver)
  EXPECT_NEAR(plan.poses.front().pose.position.x, sx, 1e-4);
  EXPECT_NEAR(plan.poses.front().pose.position.y, sy, 1e-4);
  EXPECT_NEAR(tf2::getYaw(plan.poses.front().pose.orientation), start_yaw, 1e-4);
  EXPECT_NEAR(plan.poses.back().pose.position.x, gx, 1e-4);
  EXPECT_NEAR(plan.poses.back().pose.position.y, gy, 1e-4);
  EXPECT_NEAR(tf2::getYaw(plan.poses.back().pose.orientation), goal_yaw, 1e-4);

  const TraceStats rear_on = traceStats(plan, -d);
  const TraceStats base_on = traceStats(plan, 0.0);
  EXPECT_LT(rear_on.max_lateral_ratio, 0.05);
  EXPECT_GT(base_on.max_lateral_ratio, 0.15);
  // Zero gear changes (no 1-9 mm yaw-bin endpoint slivers at start or goal).
  // At exact dx_local = 0 and dtheta = pi, L+S+L+ and L-S-L- tie in Euclidean length.
  EXPECT_EQ(rear_on.gear_changes, 0);
  EXPECT_TRUE(rear_on.reverse_steps == 0 || rear_on.reverse_steps == rear_on.steps)
    << rear_on.reverse_steps << " of " << rear_on.steps;

  // On the exact left circle of radius r_goal at a turn angle slightly under 180 deg
  // (e.g. 178.85 deg), the forward single-gear arc is strictly shorter than reverse and
  // has zero gear changes and zero reverse steps.
  const double goal_yaw_under_pi = start_yaw + M_PI - 0.02;
  const double rx1_fwd = rx0 + r_goal * (std::sin(goal_yaw_under_pi) - std::sin(start_yaw));
  const double ry1_fwd = ry0 - r_goal * (std::cos(goal_yaw_under_pi) - std::cos(start_yaw));
  const double gx_fwd = rx1_fwd + d * std::cos(goal_yaw_under_pi);
  const double gy_fwd = ry1_fwd + d * std::sin(goal_yaw_under_pi);
  const auto goal_fwd = makePose(gx_fwd, gy_fwd, goal_yaw_under_pi);
  nav_msgs::msg::Path plan_fwd = planner->createPlan(start, goal_fwd, no_cancel);
  ASSERT_GT(plan_fwd.poses.size(), 10u);
  const TraceStats rear_fwd = traceStats(plan_fwd, -d);
  EXPECT_EQ(rear_fwd.gear_changes, 0);
  EXPECT_EQ(rear_fwd.reverse_steps, 0);

  // Back to stock: the returned path is kinematically consistent for the robot frame instead
  auto results = planner->dynamicParametersCallback(
    {rclcpp::Parameter("test.rear_axle_offset", 0.0),
      rclcpp::Parameter("test.analytic_directional_scoring", false)});
  EXPECT_TRUE(results.successful);
  nav_msgs::msg::Path stock = planner->createPlan(start, goal, no_cancel);
  ASSERT_GT(stock.poses.size(), 10u);
  EXPECT_LT(traceStats(stock, 0.0).max_lateral_ratio, 0.05);
  EXPECT_GT(traceStats(stock, -d).max_lateral_ratio, 0.15);

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}
