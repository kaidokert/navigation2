// Copyright (c) 2020 Shivang Patel
// Copyright (c) 2020 Samsung Research
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

#include <string>
#include <vector>
#include <memory>
#include <limits>

#include "gtest/gtest.h"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/utils.hpp"
#include "nav2_util/lifecycle_node.hpp"

using namespace nav2_costmap_2d;  // NOLINT

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

TEST(collision_footprint, test_basic)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testA");
  nav2_costmap_2d::Costmap2D * costmap_ = new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0, 0, 0);

  geometry_msgs::msg::Point p1;
  p1.x = -0.5;
  p1.y = 0.0;
  geometry_msgs::msg::Point p2;
  p2.x = 0.0;
  p2.y = 0.5;
  geometry_msgs::msg::Point p3;
  p3.x = 0.5;
  p3.y = 0.0;
  geometry_msgs::msg::Point p4;
  p4.x = 0.0;
  p4.y = -0.5;

  nav2_costmap_2d::Footprint footprint = {p1, p2, p3, p4};

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmap_;

  nav2_smac_planner::GridCollisionChecker collision_checker(costmap_ros, 72, node);
  collision_checker.setFootprint(footprint, false /*use footprint*/, 0.0);
  collision_checker.inCollision(5.0, 5.0, 0.0, false);
  float cost = collision_checker.getCost();
  EXPECT_NEAR(cost, 0.0, 0.001);
  delete costmap_;
}

TEST(collision_footprint, test_point_cost)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testB");
  nav2_costmap_2d::Costmap2D * costmap_ = new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0, 0, 0);

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmap_;

  nav2_smac_planner::GridCollisionChecker collision_checker(costmap_ros, 72, node);
  nav2_costmap_2d::Footprint footprint;
  collision_checker.setFootprint(footprint, true /*radius / pointcose*/, 0.0);

  collision_checker.inCollision(5.0, 5.0, 0.0, false);
  float cost = collision_checker.getCost();
  EXPECT_NEAR(cost, 0.0, 0.001);
  delete costmap_;
}

TEST(collision_footprint, test_world_to_map)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testC");
  nav2_costmap_2d::Costmap2D * costmap_ = new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0, 0, 0);

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmap_;

  nav2_smac_planner::GridCollisionChecker collision_checker(costmap_ros, 72, node);
  nav2_costmap_2d::Footprint footprint;
  collision_checker.setFootprint(footprint, true /*radius / point cost*/, 0.0);

  unsigned int x, y;

  collision_checker.worldToMap(1.0, 1.0, x, y);

  collision_checker.inCollision(x, y, 0.0, false);
  float cost = collision_checker.getCost();

  EXPECT_NEAR(cost, 0.0, 0.001);

  costmap->setCost(50, 50, 200);
  collision_checker.worldToMap(5.0, 5.0, x, y);

  collision_checker.inCollision(x, y, 0.0, false);
  EXPECT_NEAR(collision_checker.getCost(), 200.0, 0.001);
  delete costmap_;
}

TEST(collision_footprint, test_footprint_at_pose_with_movement)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testD");
  nav2_costmap_2d::Costmap2D * costmap_ = new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0, 0, 254);

  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmap_->setCost(i, j, 128);
    }
  }

  geometry_msgs::msg::Point p1;
  p1.x = -1.0;
  p1.y = 1.0;
  geometry_msgs::msg::Point p2;
  p2.x = 1.0;
  p2.y = 1.0;
  geometry_msgs::msg::Point p3;
  p3.x = 1.0;
  p3.y = -1.0;
  geometry_msgs::msg::Point p4;
  p4.x = -1.0;
  p4.y = -1.0;

  nav2_costmap_2d::Footprint footprint = {p1, p2, p3, p4};

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmap_;

  nav2_smac_planner::GridCollisionChecker collision_checker(costmap_ros, 72, node);
  collision_checker.setFootprint(footprint, false /*use footprint*/, 0.0);

  EXPECT_FALSE(collision_checker.inCollision(50, 50, 0.0, false));
  float cost = collision_checker.getCost();
  EXPECT_NEAR(cost, 128.0, 0.001);

  EXPECT_TRUE(collision_checker.inCollision(50, 49, 0.0, false));
  float up_value = collision_checker.getCost();
  EXPECT_NEAR(up_value, 128.0, 0.001);  // center cost

  EXPECT_TRUE(collision_checker.inCollision(50, 52, 0.0, false));
  float down_value = collision_checker.getCost();
  EXPECT_NEAR(down_value, 128.0, 0.001);  // center cost

  EXPECT_TRUE(collision_checker.inCollision(11, 11, 0.0, false));
  float other_value = collision_checker.getCost();
  EXPECT_NEAR(other_value, 254.0, 0.001);  // center cost

  delete costmap_;
}

TEST(collision_footprint, test_point_and_line_cost)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("testE");
  nav2_costmap_2d::Costmap2D * costmap_ = new nav2_costmap_2d::Costmap2D(
    100, 100, 0.10000, 0, 0.0, 128.0);

  costmap_->setCost(62, 50, 254);
  costmap_->setCost(39, 60, 254);

  geometry_msgs::msg::Point p1;
  p1.x = -1.0;
  p1.y = 1.0;
  geometry_msgs::msg::Point p2;
  p2.x = 1.0;
  p2.y = 1.0;
  geometry_msgs::msg::Point p3;
  p3.x = 1.0;
  p3.y = -1.0;
  geometry_msgs::msg::Point p4;
  p4.x = -1.0;
  p4.y = -1.0;

  nav2_costmap_2d::Footprint footprint = {p1, p2, p3, p4};

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmap_;

  nav2_smac_planner::GridCollisionChecker collision_checker(costmap_ros, 72, node);
  collision_checker.setFootprint(footprint, false /*use footprint*/, 0.0);

  EXPECT_FALSE(collision_checker.inCollision(50, 50, 0.0, false));
  float value = collision_checker.getCost();
  EXPECT_NEAR(value, 128.0, 0.001);

  EXPECT_TRUE(collision_checker.inCollision(49, 50, 0.0, false));
  float left_value = collision_checker.getCost();
  EXPECT_NEAR(left_value, 128.0, 0.001);  // center cost

  EXPECT_TRUE(collision_checker.inCollision(52, 50, 0.0, false));
  float right_value = collision_checker.getCost();
  EXPECT_NEAR(right_value, 128.0, 0.001);  // center cost

  EXPECT_TRUE(collision_checker.inCollision(39, 60, 0.0, false));
  float other_value = collision_checker.getCost();
  EXPECT_NEAR(other_value, 254.0, 0.001);  // center cost

  delete costmap_;
}

TEST(collision_footprint, recorded_continuous_pose_matches_published_pose)
{
  // wp2plan032: the published path hits global cell (61,355), while the old
  // checker translated the footprint to floor(map coordinate) + half a cell.
  Costmap2D grid(100, 450, 0.02, -0.846, -9.168, 0);
  grid.setCost(61, 355, LETHAL_OBSTACLE);
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("continuous_pose");
  nav2_smac_planner::GridCollisionChecker checker(nullptr, 180, node);
  checker.setCostmap(&grid);
  Footprint footprint;
  for (const auto & xy : std::vector<std::pair<double, double>>{
    {0.316, 0.191}, {0.316, -0.191}, {-0.09, -0.191}, {-0.09, 0.191}})
  {
    geometry_msgs::msg::Point point;
    point.x = xy.first;
    point.y = xy.second;
    footprint.push_back(point);
  }
  checker.setFootprint(footprint, false, 0.0);
  const float heading = 76.283357f;
  const double yaw = heading * (2.0 * M_PI / 180);
  const auto published = nav2_smac_planner::getWorldCoords(79.1441f, 358.0372f, &grid);
  ASSERT_EQ(checker.footprintCostAtPose(
      published.position.x, published.position.y, yaw, footprint), LETHAL_OBSTACLE);
  EXPECT_TRUE(checker.inCollision(79.1441f, 358.0372f, heading, false));
  const auto clear = nav2_smac_planner::getWorldCoords(79.8f, 358.01f, &grid);
  ASSERT_LT(checker.footprintCostAtPose(
      clear.position.x, clear.position.y, yaw, footprint), LETHAL_OBSTACLE);
  EXPECT_FALSE(checker.inCollision(79.8f, 358.01f, heading, false));
  grid.setCost(61, 355, FREE_SPACE);
  EXPECT_FALSE(checker.inCollision(79.1441f, 358.0372f, heading, false));
}

TEST(collision_footprint, continuous_reference_uses_containing_cell)
{
  Costmap2D grid(4, 4, 0.1, 1.0, 2.0, 0);
  grid.setCost(2, 1, LETHAL_OBSTACLE);
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("continuous_center");
  nav2_smac_planner::GridCollisionChecker checker(nullptr, 180, node);
  checker.setCostmap(&grid);
  checker.setFootprint({}, true, 0.0);
  EXPECT_FALSE(checker.inCollision(1.75f, 1.1f, 0.0f, false));
  EXPECT_TRUE(checker.inCollision(2.01f, 1.1f, 0.0f, false));
  EXPECT_TRUE(checker.inCollision(4.0f, 1.0f, 0.0f, false));
  EXPECT_TRUE(checker.inCollision(1.0f, 4.0f, 0.0f, false));
  EXPECT_TRUE(checker.inCollision(std::numeric_limits<float>::quiet_NaN(), 1.0f, 0.0f, false));
}

TEST(collision_footprint, recorded_inflation_shortcut_misses_polygon)
{
  // wp2plan042: center cost 122 is below the configured circumscribed cost 128,
  // but the rasterized footprint intersects the recorded lethal cell.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("recorded_shortcut");
  Costmap2D grid(100, 200, 0.05, -0.85, -9.2, 0);
  grid.setCost(39, 142, 122);
  grid.setCost(39, 135, LETHAL_OBSTACLE);
  nav2_smac_planner::GridCollisionChecker checker(nullptr, 180, node);
  checker.setCostmap(&grid);
  Footprint footprint;
  for (const auto & xy : std::vector<std::pair<double, double>>{
    {0.288, 0.163}, {0.288, -0.163}, {-0.062, -0.163}, {-0.062, 0.163}})
  {
    geometry_msgs::msg::Point point;
    point.x = xy.first;
    point.y = xy.second;
    footprint.push_back(point);
  }
  const float x = (1.1107204491 + 0.85) / 0.05;
  const float y = (-2.0713005911 + 9.2) / 0.05;
  checker.setFootprint(footprint, false, 0.0);
  ASSERT_TRUE(checker.inCollision(x, y, 149.0f, false));
  checker.setFootprint(footprint, false, 128.0);
  EXPECT_TRUE(checker.inCollision(x, y, 149.0f, false));
  grid.setCost(39, 135, FREE_SPACE);
  EXPECT_FALSE(checker.inCollision(x, y, 149.0f, false));
  EXPECT_FLOAT_EQ(checker.getCost(), 122.0f);
}
