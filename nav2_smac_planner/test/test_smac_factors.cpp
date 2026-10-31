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
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "ompl/base/spaces/DubinsStateSpace.h"
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
          x + offset * std::cos(angle), y + offset * std::sin(angle), static_cast<float>(bin),
          false);
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

  // length_direction_cusp: each change of gear adds change_penalty x radius (2 x 10 cells), so a
  // sliver of reverse no longer scores like a single-gear path
  info.analytic_score_cusp_cost = true;
  Expansion cusp_expander(nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, 72);
  EXPECT_NEAR(cusp_expander.lengthDirectionScore(build({10, 11, 12, 13})), 3.0f, 1e-4);
  EXPECT_NEAR(
    cusp_expander.lengthDirectionScore(build({10, 11, 12, 11})),
    2.0f + 3.6f * (1.0f + 2.0f) + 2.0f * 10.0f, 1e-3);
  EXPECT_GT(
    cusp_expander.lengthDirectionScore(build({10, 11, 12, 13, 12.99f})),
    cusp_expander.lengthDirectionScore(build({10, 11, 12, 13})) + 19.0f);

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

namespace
{

// Robot-frame pose of a rear-axle U-turn: the rear axle ends `lateral` to the left, level with
// its start, heading reversed. offset_lon / offset_yaw perturb the goal (start frame).
geometry_msgs::msg::PoseStamped uTurnGoal(
  const geometry_msgs::msg::PoseStamped & start, double axle, double lateral,
  double offset_lon = 0.0, double offset_yaw = 0.0)
{
  const double yaw = tf2::getYaw(start.pose.orientation);
  const double rx = start.pose.position.x - axle * cos(yaw);
  const double ry = start.pose.position.y - axle * sin(yaw);
  const double gx = rx + offset_lon * cos(yaw) - lateral * sin(yaw);
  const double gy = ry + offset_lon * sin(yaw) + lateral * cos(yaw);
  const double gyaw = yaw + M_PI + offset_yaw;
  return makePose(gx + axle * cos(gyaw), gy + axle * sin(gyaw), gyaw);
}

nav_msgs::msg::Path subPath(const nav_msgs::msg::Path & plan, size_t first, size_t count)
{
  nav_msgs::msg::Path out;
  for (size_t i = first; i < first + count && i < plan.poses.size(); ++i) {
    out.poses.push_back(plan.poses[i]);
  }
  return out;
}

std::unique_ptr<FactorPlanner> makeRobotPlanner(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  // The robot settings that matter here: 0.56 m radius, 180 heading bins, rear axle 0.106 m
  // behind base_link, no smoothing.
  node->declare_parameter("test.motion_model_for_search", std::string("REEDS_SHEPP"));
  node->declare_parameter("test.minimum_turning_radius", 0.56);
  node->declare_parameter("test.angle_quantization_bins", 180);
  node->declare_parameter("test.lookup_table_size", 1.0);
  node->declare_parameter("test.downsample_costmap", false);
  node->declare_parameter("test.smooth_path", false);
  node->declare_parameter("test.analytic_expansion_max_length", 3.0);
  node->declare_parameter("test.reverse_penalty", 3.6);
  node->declare_parameter("test.change_penalty", 2.0);
  node->declare_parameter("test.rear_axle_offset", 0.106);
  node->declare_parameter("test.analytic_score", std::string("length_direction"));
  node->declare_parameter("test.analytic_single_gear", false);
  auto planner = std::make_unique<FactorPlanner>();
  planner->configure(node, "test", nullptr, costmap_ros);
  planner->activate();
  return planner;
}

}  // namespace

TEST(SmacFactors, rear_axle_plan_ends_on_the_exact_start_and_goal)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsEndpoints");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  *costmap_ros->getCostmap() = nav2_costmap_2d::Costmap2D(200, 200, 0.02, 0.0, 0.0, 0);
  auto planner = makeRobotPlanner(node, costmap_ros);
  auto no_cancel = []() {return false;};

  // The plan starts and ends on the exact requested poses, also for start headings between the
  // 2 degree bins (the start node heading is quantized). On a bin, the U-turn is one gear end to
  // end: no millimetre end step against the path.
  for (const double yaw : {M_PI / 2.0, 0.0123, 1.5654, -0.4}) {
    const auto start = makePose(1.6, 1.4, yaw);
    const auto goal = uTurnGoal(start, 0.106, 1.12);
    const nav_msgs::msg::Path plan = planner->createPlan(start, goal, no_cancel);
    ASSERT_GT(plan.poses.size(), 10u);
    EXPECT_DOUBLE_EQ(plan.poses.front().pose.position.x, start.pose.position.x);
    EXPECT_DOUBLE_EQ(plan.poses.front().pose.position.y, start.pose.position.y);
    EXPECT_NEAR(tf2::getYaw(plan.poses.front().pose.orientation), yaw, 1e-9);
    EXPECT_DOUBLE_EQ(plan.poses.back().pose.position.x, goal.pose.position.x);
    EXPECT_DOUBLE_EQ(plan.poses.back().pose.position.y, goal.pose.position.y);
    if (yaw == M_PI / 2.0) {
      const TraceStats all = traceStats(plan, -0.106);
      EXPECT_TRUE(all.reverse_steps == 0 || all.reverse_steps == all.steps)
        << all.reverse_steps << " of " << all.steps;
    }
  }

  // Between bins the quantized headings alone make the shortest Reeds-Shepp word add a
  // centimetre stub. With a few centimetres to spare across the turn (1.16 m for 2R = 1.12 m) and
  // the single-gear candidates, the whole plan, ends included, is one gear. At exactly 2R the
  // rotated goal can fall inside 2R, where a stub is genuinely shorter than any one-gear path.
  planner->dynamicParametersCallback({rclcpp::Parameter("test.analytic_single_gear", true)});
  for (const double yaw : {0.0123, 1.5654, -0.4}) {
    const auto start = makePose(1.6, 1.4, yaw);
    const auto goal = uTurnGoal(start, 0.106, 1.16);
    const nav_msgs::msg::Path plan = planner->createPlan(start, goal, no_cancel);
    ASSERT_GT(plan.poses.size(), 10u);
    const size_t n = plan.poses.size();
    const TraceStats first = traceStats(subPath(plan, 0, 3), -0.106);
    const TraceStats last = traceStats(subPath(plan, n - 3, 3), -0.106);
    const TraceStats all = traceStats(plan, -0.106);
    EXPECT_TRUE(all.reverse_steps == 0 || all.reverse_steps == all.steps) << "yaw " << yaw;
    EXPECT_EQ(first.reverse_steps == 0, all.reverse_steps == 0) << "yaw " << yaw;
    EXPECT_EQ(last.reverse_steps == 0, all.reverse_steps == 0) << "yaw " << yaw;
    EXPECT_DOUBLE_EQ(plan.poses.front().pose.position.x, start.pose.position.x);
    EXPECT_DOUBLE_EQ(plan.poses.back().pose.position.y, goal.pose.position.y);
  }
  planner->dynamicParametersCallback({rclcpp::Parameter("test.analytic_single_gear", false)});

  // Stock frame: endpoints are not restored, the start node is used as is
  planner->dynamicParametersCallback({rclcpp::Parameter("test.rear_axle_offset", 0.0)});
  const auto start = makePose(1.6, 1.4, 0.0123);
  const nav_msgs::msg::Path stock = planner->createPlan(
    start, uTurnGoal(start, 0.106, 1.12), no_cancel);
  ASSERT_GT(stock.poses.size(), 1u);
  EXPECT_NEAR(stock.poses.front().pose.position.x, 1.6, 1e-6);

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}

TEST(SmacFactors, single_gear_candidates_remove_stubs_from_perturbed_u_turns)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsSingleGear");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  *costmap_ros->getCostmap() = nav2_costmap_2d::Costmap2D(200, 200, 0.02, 0.0, 0.0, 0);
  auto planner = makeRobotPlanner(node, costmap_ros);
  auto no_cancel = []() {return false;};

  // U-turns with room (rear axle 1.20 m across) but the goal 1-2 cm and 2-3 degrees off the
  // exact turn: the shortest Reeds-Shepp word adds a centimetre opposite-gear stub.
  struct Case {double lon, yaw;};
  const std::vector<Case> cases = {{0.02, 0.0}, {-0.02, 0.0}, {0.0, 0.035}, {0.01, -0.05}};
  int stubbed_without = 0;
  for (const auto & c : cases) {
    const auto start = makePose(1.6, 1.4, 0.0);
    const auto goal = uTurnGoal(start, 0.106, 1.20, c.lon, c.yaw);
    planner->dynamicParametersCallback({rclcpp::Parameter("test.analytic_single_gear", false)});
    const TraceStats without = traceStats(planner->createPlan(start, goal, no_cancel), -0.106);
    stubbed_without += (without.reverse_steps > 0 && without.reverse_steps < without.steps);
    planner->dynamicParametersCallback({rclcpp::Parameter("test.analytic_single_gear", true)});
    const nav_msgs::msg::Path plan = planner->createPlan(start, goal, no_cancel);
    const TraceStats with = traceStats(plan, -0.106);
    EXPECT_EQ(with.reverse_steps, 0) << "lon " << c.lon << " yaw " << c.yaw;
    EXPECT_DOUBLE_EQ(plan.poses.back().pose.position.x, goal.pose.position.x);
  }
  EXPECT_GT(stubbed_without, 0);

  planner->deactivate();
  planner->cleanup();
  planner.reset();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}

TEST(SmacFactors, reverse_only_candidate_drives_backwards)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("factorsReverse");
  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 14;  // cells
  info.reverse_penalty = 3.6;
  info.change_penalty = 2.0;
  info.analytic_expansion_max_length = 200;
  unsigned int size_x = 200;
  unsigned int size_y = 200;
  unsigned int size_theta = 180;
  nav2_smac_planner::NodeHybrid::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_theta, info);
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  *costmap_ros->getCostmap() = nav2_costmap_2d::Costmap2D(size_x, size_y, 0.04, 0.0, 0.0, 0);
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, size_theta, node);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  using Expansion = nav2_smac_planner::AnalyticExpansion<nav2_smac_planner::NodeHybrid>;
  Expansion expander(nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  expander.setCollisionChecker(&checker);
  std::unordered_map<uint64_t, std::unique_ptr<nav2_smac_planner::NodeHybrid>> graph;
  Expansion::NodeGetter getter =
    [&](const uint64_t & index, nav2_smac_planner::NodeHybrid * & n) {
      auto & slot = graph[index];
      if (!slot) {
        slot = std::make_unique<nav2_smac_planner::NodeHybrid>(index);
      }
      n = slot.get();
      return true;
    };
  nav2_smac_planner::NodeHybrid start(0), goal(1);
  start.setPose(nav2_smac_planner::NodeHybrid::Coordinates(80.0f, 100.0f, 0.0f));
  // A U-turn to the left, 30 cells across, heading reversed (bin 90 of 180)
  goal.setPose(nav2_smac_planner::NodeHybrid::Coordinates(80.0f, 130.0f, 90.0f));
  auto space = std::make_shared<ompl::base::DubinsStateSpace>(info.minimum_turning_radius);
  for (const bool reverse_only : {false, true}) {
    const auto nodes = expander.getAnalyticPath(&start, &goal, getter, space, reverse_only);
    ASSERT_GT(nodes.size(), 10u);
    int forward = 0;
    int reverse = 0;
    float px = start.pose.x;
    float py = start.pose.y;
    float ptheta = start.pose.theta;
    for (const auto & n : nodes) {
      const float heading = n.node->motion_table.getAngleFromBin(ptheta);
      const float along =
        (n.proposed_coords.x - px) * cosf(heading) + (n.proposed_coords.y - py) * sinf(heading);
      if (along < 0.0f) {
        reverse++;
      } else {
        forward++;
      }
      px = n.proposed_coords.x;
      py = n.proposed_coords.y;
      ptheta = n.proposed_coords.theta;
    }
    EXPECT_EQ(reverse_only ? forward : reverse, 0) << "reverse_only " << reverse_only;
    EXPECT_NEAR(nodes.back().proposed_coords.x, 80.0f, 0.5f);
    EXPECT_NEAR(nodes.back().proposed_coords.y, 130.0f, 0.5f);
  }
  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}
