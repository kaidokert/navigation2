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
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "ompl/base/spaces/DubinsStateSpace.h"
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
