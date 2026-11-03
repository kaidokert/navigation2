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

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
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

  // start_coords precedes expansion[0]: a first-step gear reversal from start_coords is also scored
  const nav2_smac_planner::NodeHybrid::Coordinates start_pose(10.01f, 10.0f, 0.0f);
  EXPECT_GT(
    cusp_expander.lengthDirectionScore(build({10, 11, 12, 13}), &start_pose),
    cusp_expander.lengthDirectionScore(build({10, 11, 12, 13})) + 19.0f);

  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}
