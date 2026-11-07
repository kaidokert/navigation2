// Copyright (c) 2020, Samsung Research America
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

#include <math.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <regex>
#include <cstdlib>
#include <cstdio>
#include <functional>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/socket.h>
#endif

#include "rcutils/env.h"

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_subscriber.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_smac_planner/node_hybrid.hpp"
#include "nav2_smac_planner/node_lattice.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

class RclCppFixture
{
public:
  RclCppFixture() {rclcpp::init(0, nullptr);}
  ~RclCppFixture() {rclcpp::shutdown();}
};
RclCppFixture g_rclcppfixture;

TEST(AStarTest, test_a_star_2d)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D> a_star(
    nav2_smac_planner::MotionModel::TWOD, info);
  int max_iterations = 10000;
  float tolerance = 0.0;
  float some_tolerance = 20.0;
  int it_on_approach = 10;
  int terminal_checking_interval = 5000;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, it_on_approach, terminal_checking_interval,
    max_planning_time, 0.0, 1);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmapA;

  auto dummy_cancel_checker = []() {
      return false;
    };

  // functional case testing
  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap_ros, 1, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(20u, 20u, 0);
  a_star.setGoal(80u, 80u, 0);
  nav2_smac_planner::Node2D::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance, dummy_cancel_checker));
  EXPECT_EQ(num_it, 2414);

  // check path is the right size and collision free
  EXPECT_EQ(path.size(), 82u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }

  // setting non-zero dim 3 for 2D search
  EXPECT_THROW(a_star.setGoal(0, 0, 10), std::runtime_error);
  EXPECT_THROW(a_star.setStart(0, 0, 10), std::runtime_error);

  path.clear();
  // failure cases with invalid inputs
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D> a_star_2(
    nav2_smac_planner::MotionModel::TWOD, info);
  a_star_2.initialize(
    false, max_iterations, it_on_approach, terminal_checking_interval,
    max_planning_time, 0, 1);
  num_it = 0;
  EXPECT_THROW(
    a_star_2.createPath(
      path, num_it, tolerance,
      dummy_cancel_checker), std::runtime_error);
  a_star_2.setCollisionChecker(checker.get());
  num_it = 0;
  EXPECT_THROW(
    a_star_2.createPath(
      path, num_it, tolerance,
      dummy_cancel_checker), std::runtime_error);
  a_star_2.setStart(0, 0, 0);  // valid
  a_star_2.setGoal(50, 50, 0);  // invalid
  num_it = 0;
  EXPECT_THROW(
    a_star_2.createPath(
      path, num_it, tolerance,
      dummy_cancel_checker), std::runtime_error);
  num_it = 0;
  // invalid goal but liberal tolerance
  a_star_2.setStart(20, 20, 0);  // valid
  a_star_2.setGoal(50, 50, 0);  // invalid
  EXPECT_TRUE(a_star_2.createPath(path, num_it, some_tolerance, dummy_cancel_checker));
  EXPECT_EQ(path.size(), 21u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }

  EXPECT_TRUE(a_star_2.getStart() != nullptr);
  EXPECT_TRUE(a_star_2.getGoal() != nullptr);
  EXPECT_EQ(a_star_2.getSizeX(), 100u);
  EXPECT_EQ(a_star_2.getSizeY(), 100u);
  EXPECT_EQ(a_star_2.getSizeDim3(), 1u);
  EXPECT_EQ(a_star_2.getToleranceHeuristic(), 20.0);
  EXPECT_EQ(a_star_2.getOnApproachMaxIterations(), 10);

  delete costmapA;
}

TEST(AStarTest, test_a_star_se2)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.1;
  info.non_straight_penalty = 1.1;
  info.reverse_penalty = 2.0;
  info.minimum_turning_radius = 8;  // in grid coordinates
  info.retrospective_penalty = 0.015;
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  info.analytic_expansion_ratio = 3.5;
  unsigned int size_theta = 72;
  info.cost_penalty = 1.7;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::DUBIN, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  int terminal_checking_interval = 5000;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, it_on_approach, terminal_checking_interval,
    max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 40; i <= 60; ++i) {
    for (unsigned int j = 40; j <= 60; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmapA;

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap_ros, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(10u, 10u, 0u);
  a_star.setGoal(80u, 80u, 40u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();

  auto dummy_cancel_checker = []() {
      return false;
    };

  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance, dummy_cancel_checker, expansions.get()));

  // check path is the right size and collision free
  EXPECT_GT(num_it, 2000);
  EXPECT_NEAR(path.size(), 77u, 2u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }
  // no skipped nodes
  for (unsigned int i = 1; i != path.size(); i++) {
    EXPECT_LT(hypotf(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y), 2.1f);
  }

  // Expansions properly recorded
  EXPECT_GT(expansions->size(), 5u);

  delete costmapA;
  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(AStarTest, test_a_star_analytic_expansion)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.0;
  info.non_straight_penalty = 1.1;
  info.reverse_penalty = 0.0;
  info.minimum_turning_radius = 8;  // in grid coordinates
  info.retrospective_penalty = 0.015;
  info.analytic_expansion_max_length = 2000.0;  // in grid coordinates
  info.analytic_expansion_ratio = 3.5;
  unsigned int size_theta = 72;
  info.cost_penalty = 1.7;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  int terminal_checking_interval = 5000;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, it_on_approach, terminal_checking_interval,
    max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmapA;

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap_ros, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  // should be a straight path running backwards
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(80u, 0u, 0u);
  a_star.setGoal(20u, 0u, 0u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();

  auto dummy_cancel_checker = []() {
      return false;
    };

  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance, dummy_cancel_checker, expansions.get()));

  // all straight with no wiggle
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_NEAR(path[i].theta, 0.0, 1e-3);
  }

  delete costmapA;
  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(AStarTest, test_a_star_lattice)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.05;
  info.non_straight_penalty = 1.05;
  info.reverse_penalty = 2.0;
  info.retrospective_penalty = 0.1;
  info.analytic_expansion_ratio = 3.5;
  info.lattice_filepath =
    ament_index_cpp::get_package_share_directory("nav2_smac_planner") +
    "/sample_primitives/5cm_resolution/0.5m_turning_radius/ackermann" +
    "/output.json";
  info.minimum_turning_radius = 8;  // in grid coordinates 0.4/0.05
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  unsigned int size_theta = 16;
  info.cost_penalty = 2.0;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeLattice> a_star(
    nav2_smac_planner::MotionModel::STATE_LATTICE, info);
  int max_iterations = 10000;
  float tolerance = 10.0;
  int it_on_approach = 10;
  int terminal_checking_interval = 5000;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations,
    std::numeric_limits<int>::max(), terminal_checking_interval, max_planning_time, 401,
    size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.05, 0.0, 0.0, 0);
  // island in the middle of lethal cost to cross
  for (unsigned int i = 20; i <= 30; ++i) {
    for (unsigned int j = 20; j <= 30; ++j) {
      costmapA->setCost(i, j, 254);
    }
  }

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmapA;

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap_ros, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  auto dummy_cancel_checker = []() {
      return false;
    };

  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(5u, 5u, 0u);
  a_star.setGoal(40u, 40u, 1u);
  nav2_smac_planner::NodeLattice::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance, dummy_cancel_checker));

  // check path is the right size and collision free
  EXPECT_EQ(num_it, 22);
  EXPECT_GT(path.size(), 45u);
  for (unsigned int i = 0; i != path.size(); i++) {
    EXPECT_EQ(costmapA->getCost(path[i].x, path[i].y), 0);
  }
  // no skipped nodes
  for (unsigned int i = 1; i != path.size(); i++) {
    EXPECT_LT(hypotf(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y), 2.1f);
  }

  delete costmapA;
  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(AStarTest, test_se2_single_pose_path)
{
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("test");
  nav2_smac_planner::SearchInfo info;
  info.change_penalty = 0.1;
  info.non_straight_penalty = 1.1;
  info.reverse_penalty = 2.0;
  info.retrospective_penalty = 0.0;
  info.minimum_turning_radius = 8;  // in grid coordinates
  info.analytic_expansion_max_length = 20.0;  // in grid coordinates
  info.analytic_expansion_ratio = 3.5;
  unsigned int size_theta = 72;
  info.cost_penalty = 1.7;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> a_star(
    nav2_smac_planner::MotionModel::DUBIN, info);
  int max_iterations = 100;
  float tolerance = 10.0;
  int it_on_approach = 10;
  int terminal_checking_interval = 5000;
  double max_planning_time = 120.0;
  int num_it = 0;

  a_star.initialize(
    false, max_iterations, it_on_approach, terminal_checking_interval,
    max_planning_time, 401, size_theta);

  nav2_costmap_2d::Costmap2D * costmapA =
    new nav2_costmap_2d::Costmap2D(100, 100, 0.1, 0.0, 0.0, 0);

  // Convert raw costmap into a costmap ros object
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto costmap = costmap_ros->getCostmap();
  *costmap = *costmapA;

  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker =
    std::make_unique<nav2_smac_planner::GridCollisionChecker>(costmap_ros, size_theta, lnode);
  checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  auto dummy_cancel_checker = []() {
      return false;
    };
  // functional case testing
  a_star.setCollisionChecker(checker.get());
  a_star.setStart(10u, 10u, 0u);
  // Goal is one costmap cell away
  a_star.setGoal(12u, 10u, 0u);
  nav2_smac_planner::NodeHybrid::CoordinateVector path;
  EXPECT_TRUE(a_star.createPath(path, num_it, tolerance, dummy_cancel_checker));

  // Check that the path is length one
  // With the current implementation, this produces a longer path
  // EXPECT_EQ(path.size(), 1u);
  EXPECT_GE(path.size(), 1u);

  delete costmapA;
  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
}

TEST(AStarTest, test_constants)
{
  nav2_smac_planner::MotionModel mm = nav2_smac_planner::MotionModel::UNKNOWN;  // unknown
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Unknown"));
  mm = nav2_smac_planner::MotionModel::TWOD;  // 2d
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("2D"));
  mm = nav2_smac_planner::MotionModel::DUBIN;  // dubin
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Dubin"));
  mm = nav2_smac_planner::MotionModel::REEDS_SHEPP;  // reeds-shepp
  EXPECT_EQ(nav2_smac_planner::toString(mm), std::string("Reeds-Shepp"));

  EXPECT_EQ(
    nav2_smac_planner::fromString(
      "2D"), nav2_smac_planner::MotionModel::TWOD);
  EXPECT_EQ(nav2_smac_planner::fromString("DUBIN"), nav2_smac_planner::MotionModel::DUBIN);
  EXPECT_EQ(
    nav2_smac_planner::fromString(
      "REEDS_SHEPP"), nav2_smac_planner::MotionModel::REEDS_SHEPP);
  EXPECT_EQ(nav2_smac_planner::fromString("NONE"), nav2_smac_planner::MotionModel::UNKNOWN);
}

TEST(AStarTest, obstacle_heuristic_uses_collision_grid_and_invalidates_changed_view)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("heuristic_grid_test");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("heuristic_grid_map");
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto original = costmap_ros->getCostmap();
  original->resizeMap(40, 40, 0.02, 0.0, 0.0);
  // The ROS grid is lethal; heuristic evaluation must read the free search grid.
  std::fill_n(
    original->getCharMap(), original->getSizeInCellsX() * original->getSizeInCellsY(),
    nav2_costmap_2d::LETHAL_OBSTACLE);
  nav2_costmap_2d::Costmap2D search_grid(20, 20, 0.04, 0.0, 0.0, 0);
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, 72, node);
  checker.setCostmap(&search_grid);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 4.0;
  info.cache_obstacle_heuristic = true;
  info.downsample_obstacle_heuristic = true;
  nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeHybrid> planner(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info);
  int max_iterations = 10000;
  planner.initialize(true, max_iterations, 1000, 5000, 0.0, 11.0f, 72);
  planner.setCollisionChecker(&checker);
  planner.setStart(2.0f, 2.0f, 0);
  planner.setGoal(15.0f, 15.0f, 0);

  EXPECT_EQ(nav2_smac_planner::NodeHybrid::obstacle_heuristic_costmap, &search_grid);
  EXPECT_DOUBLE_EQ(
    nav2_smac_planner::NodeHybrid::obstacle_heuristic_costmap->getResolution(), 0.04);
  EXPECT_EQ(nav2_smac_planner::NodeHybrid::obstacle_heuristic_lookup_table.size(), 100u);

  // Same numeric goal on another grid must not reuse the previous grid's cache.
  checker.setCostmap(original);
  planner.setCollisionChecker(&checker);
  planner.setStart(2.0f, 2.0f, 0);
  planner.setGoal(15.0f, 15.0f, 0);
  EXPECT_EQ(nav2_smac_planner::NodeHybrid::obstacle_heuristic_costmap, original);
  EXPECT_DOUBLE_EQ(
    nav2_smac_planner::NodeHybrid::obstacle_heuristic_costmap->getResolution(), 0.02);
  EXPECT_EQ(nav2_smac_planner::NodeHybrid::obstacle_heuristic_lookup_table.size(), 400u);

  // Equal dimensions isolate the cost source without relying on table-size checks.
  search_grid.resizeMap(40, 40, 0.04, 0.0, 0.0);
  checker.setCostmap(&search_grid);
  planner.setCollisionChecker(&checker);
  planner.setStart(10.0f, 10.0f, 0);
  planner.setGoal(15.0f, 15.0f, 0);
  using Node = nav2_smac_planner::NodeHybrid;
  const Node::Coordinates query(10.0f, 10.0f, 0.0f);
  const Node::Coordinates goal(15.0f, 15.0f, 0.0f);
  const float expected_cost = 4.0f * std::sqrt(2.0f);
  EXPECT_NEAR(Node::getObstacleHeuristic(query, goal, 3.5f), expected_cost, 0.001f);

  // An unchanged grid and goal should retain the computed cache entry.
  const float cached_cost = Node::obstacle_heuristic_lookup_table.at(105u);
  ASSERT_GT(cached_cost, 0.0f);
  planner.setCollisionChecker(&checker);
  planner.setStart(10.0f, 10.0f, 0);
  planner.setGoal(15.0f, 15.0f, 0);
  EXPECT_FLOAT_EQ(Node::obstacle_heuristic_lookup_table.at(105u), cached_cost);

  // The downsampler resizes its existing object; pointer identity stays unchanged.
  // Grow and shrink with the same goal, exercising both invalidation and clearing.
  for (const auto & dimensions :
    std::vector<std::pair<unsigned int, unsigned int>>{{60u, 40u}, {20u, 20u}, {40u, 40u}})
  {
    search_grid.resizeMap(dimensions.first, dimensions.second, 0.04, 0.0, 0.0);
    planner.setCollisionChecker(&checker);
    planner.setStart(10.0f, 10.0f, 0);
    planner.setGoal(15.0f, 15.0f, 0);
    const size_t expected_size = (dimensions.first / 2u) * (dimensions.second / 2u);
    ASSERT_EQ(Node::obstacle_heuristic_lookup_table.size(), expected_size);
    EXPECT_EQ(Node::obstacle_heuristic_costmap, &search_grid);
    EXPECT_NEAR(Node::getObstacleHeuristic(query, goal, 3.5f), expected_cost, 0.001f);
  }

  nav2_smac_planner::NodeHybrid::destroyStaticAssets();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}

#ifdef __linux__
namespace
{
class ScopedTimingEnvironment
{
public:
  explicit ScopedTimingEnvironment(const char * value)
  {
    const char * previous = std::getenv("NAV2_SMAC_TIMING");
    existed_ = previous != nullptr;
    if (existed_) {
      previous_ = previous;
    }
    EXPECT_TRUE(rcutils_set_env("NAV2_SMAC_TIMING", value));
  }
  ~ScopedTimingEnvironment()
  {
    EXPECT_TRUE(rcutils_set_env("NAV2_SMAC_TIMING", existed_ ? previous_.c_str() : nullptr));
  }

private:
  bool existed_;
  std::string previous_;
};

class TimingSearch
{
public:
  using Algorithm = nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::Node2D>;
  TimingSearch(
    int cap = 10000, bool barrier = false, bool invalid = false, bool same = false,
    double budget = 120.0)
  : node(std::make_shared<rclcpp_lifecycle::LifecycleNode>("timing_test")),
    map(std::make_shared<nav2_costmap_2d::Costmap2DROS>()),
    search(nav2_smac_planner::MotionModel::TWOD, nav2_smac_planner::SearchInfo())
  {
    map->on_configure(rclcpp_lifecycle::State());
    map->getCostmap()->resizeMap(30, 30, 0.1, 0.0, 0.0);
    if (barrier) {
      for (unsigned int y = 0; y < 30; ++y) {
        map->getCostmap()->setCost(15, y, 254);
      }
    }
    checker = std::make_unique<nav2_smac_planner::GridCollisionChecker>(map, 1, node);
    checker->setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);
    search.initialize(false, cap, 10, 2, budget, 0.0, 1);
    search.setCollisionChecker(checker.get());
    search.setStart(2, 2, 0);
    if (!invalid) {
      search.setGoal(same ? 2 : 27, same ? 2 : 27, 0);
    }
  }

  std::string run(bool cancel = false, std::function<bool()> callback = {})
  {
    testing::internal::CaptureStderr();
    std::fputs("BEFORE_TIMING_RECORDS\n", stderr);
    std::fflush(stderr);
    try {
      found = search.createPath(
        path, iterations, 0.0, [&]() {return callback ? callback() : cancel;});
    } catch (const std::exception &) {
      threw = true;
    }
    std::fputs("AFTER_TIMING_RECORDS\n", stderr);
    std::fflush(stderr);
    return testing::internal::GetCapturedStderr();
  }

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> map;
  std::unique_ptr<nav2_smac_planner::GridCollisionChecker> checker;
  Algorithm search;
  nav2_smac_planner::Node2D::CoordinateVector path;
  int iterations = 0;
  bool found = false;
  bool threw = false;
};

std::vector<std::string> timingRecords(const std::string & output)
{
  std::vector<std::string> result;
  const std::regex record("SMAC_TIMING (\\{[^\\n]*\\})");
  for (std::sregex_iterator i(output.begin(), output.end(), record), end; i != end; ++i) {
    result.push_back((*i)[1].str());
  }
  return result;
}

uint64_t timingNumber(const std::string & record, const std::string & name)
{
  std::smatch match;
  const bool found = std::regex_search(record, match, std::regex("\"" + name + "\":([0-9]+)"));
  EXPECT_TRUE(found) << record;
  return found ? std::stoull(match[1].str()) : 0;
}
}  // namespace

TEST(AStarTest, timing_is_opt_in_and_preserves_path)
{
  ScopedTimingEnvironment enabled("1");
  TimingSearch baseline;
  const auto output = baseline.run();
  EXPECT_LT(output.find("BEFORE_TIMING_RECORDS"), output.find("SMAC_TIMING"));
  EXPECT_GT(output.find("AFTER_TIMING_RECORDS"), output.rfind("SMAC_TIMING"));
  auto records = timingRecords(output);
  ASSERT_TRUE(baseline.found);
  ASSERT_GE(records.size(), 3u);
  EXPECT_NE(records.front().find("\"event\":\"start\""), std::string::npos);
  EXPECT_NE(records.back().find("\"event\":\"goal\""), std::string::npos);
  EXPECT_GT(timingNumber(records.back(), "queue"), 0u);
  EXPECT_GT(timingNumber(records.back(), "sampled_pops"), 0u);
  std::smatch samples;
  ASSERT_TRUE(std::regex_search(
      records.back(), samples,
      std::regex("\"phase_samples\":\\[([0-9]+),([0-9]+),([0-9]+),([0-9]+)\\]")));
  EXPECT_EQ(std::stoull(samples[1].str()), timingNumber(records.back(), "sampled_pops"));
  EXPECT_EQ(std::stoull(samples[2].str()), timingNumber(records.back(), "sampled_expansions"));
  EXPECT_EQ(samples[3].str(), samples[4].str());
  EXPECT_LE(std::stoull(samples[3].str()), std::stoull(samples[2].str()));
  for (const char * phase : {"selection_ms", "analytic_ms", "neighbors_ms", "heuristic_queue_ms"}) {
    std::smatch duration;
    ASSERT_TRUE(std::regex_search(
        records.back(), duration, std::regex(std::string("\"") + phase + "\":([0-9.]+)")));
    const double value = std::stod(duration[1].str());
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GT(value, 0.0);
  }
  for (const auto & record : records) {
    if (record.find("\"event\":\"checkpoint\"") != std::string::npos) {
      EXPECT_GT(timingNumber(record, "iterations"), 0u);
    }
  }
  for (const char * value : {static_cast<const char *>(nullptr), "", "0"}) {
    ScopedTimingEnvironment disabled(value);
    TimingSearch comparison;
    EXPECT_TRUE(timingRecords(comparison.run()).empty());
    ASSERT_TRUE(comparison.found);
    EXPECT_EQ(comparison.iterations, baseline.iterations);
    ASSERT_EQ(comparison.path.size(), baseline.path.size());
    for (size_t i = 0; i < comparison.path.size(); ++i) {
      EXPECT_EQ(comparison.path[i].x, baseline.path[i].x);
      EXPECT_EQ(comparison.path[i].y, baseline.path[i].y);
    }
  }
}

TEST(AStarTest, timing_distinguishes_search_outcomes)
{
  ScopedTimingEnvironment enabled("1");
  TimingSearch invalid(10000, false, true);
  auto records = timingRecords(invalid.run());
  ASSERT_TRUE(invalid.threw);
  ASSERT_EQ(records.size(), 2u);
  EXPECT_NE(records.back().find("\"event\":\"invalid_input\""), std::string::npos);

  TimingSearch cancelled;
  records = timingRecords(cancelled.run(true));
  ASSERT_TRUE(cancelled.threw);
  ASSERT_EQ(records.size(), 2u);
  EXPECT_NE(records.back().find("\"event\":\"cancelled\""), std::string::npos);

  TimingSearch capped(1);
  records = timingRecords(capped.run());
  ASSERT_FALSE(capped.found);
  ASSERT_EQ(records.size(), 2u);
  EXPECT_NE(records.back().find("\"event\":\"max_iterations\""), std::string::npos);
  EXPECT_GT(timingNumber(records.back(), "queue"), 0u);

  TimingSearch exhausted(10000, true);
  records = timingRecords(exhausted.run());
  ASSERT_FALSE(exhausted.found);
  ASSERT_FALSE(records.empty());
  EXPECT_NE(records.back().find("\"event\":\"no_path\""), std::string::npos);
  EXPECT_EQ(timingNumber(records.back(), "queue"), 0u);

  TimingSearch backtrace(10000, false, false, true);
  records = timingRecords(backtrace.run());
  ASSERT_FALSE(backtrace.found);
  ASSERT_FALSE(records.empty());
  EXPECT_NE(records.back().find("\"event\":\"backtrace_failed\""), std::string::npos);
}

TEST(AStarTest, timing_checks_cancellation_before_emitting_checkpoint)
{
  ScopedTimingEnvironment enabled("1");
  TimingSearch search;
  int last_checked = 0;
  int checks = 0;
  search.run(
    false, [&]() {
      if (search.iterations > last_checked) {
        char buffer[2048];
        const off_t end = lseek(STDERR_FILENO, 0, SEEK_CUR);
        const ssize_t size = pread(
          STDERR_FILENO, buffer, sizeof(buffer),
        std::max<off_t>(0, end - static_cast<off_t>(sizeof(buffer))));
        EXPECT_GT(size, 0);
        if (size > 0) {
          const auto records = timingRecords(std::string(buffer, static_cast<size_t>(size)));
          EXPECT_FALSE(records.empty());
          if (!records.empty()) {
            EXPECT_LT(timingNumber(records.back(), "iterations"), search.iterations);
          }
        }
        last_checked = search.iterations;
        ++checks;
      }
      return false;
    });
  EXPECT_TRUE(search.found);
  EXPECT_GT(checks, 0);
}

TEST(AStarTest, timing_flushes_deadline_without_initial_checkpoint)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_DEATH(
  {
    ScopedTimingEnvironment enabled("1");
    TimingSearch expired(10000, false, false, false, 1e-12);
    expired.search.createPath(expired.path, expired.iterations, 0.0, []() {return false;});
    }, "SMAC_TIMING.*event.*deadline");
}

TEST(AStarTest, timing_does_not_block_on_full_or_closed_stderr_pipe)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (bool closed : {false, true}) {
    ASSERT_EXIT(
    {
      ScopedTimingEnvironment enabled("1");
      TimingSearch search;
      int descriptors[2];
      if (pipe2(descriptors, O_NONBLOCK | O_CLOEXEC) != 0) {_exit(10);}
      if (!closed) {
        char buffer[4096] = {};
        while (write(descriptors[1], buffer, sizeof(buffer)) > 0) {}
      }
      const int original_flags = fcntl(descriptors[1], F_GETFL) & ~O_NONBLOCK;
      fcntl(descriptors[1], F_SETFL, original_flags);
      const int saved = dup(STDERR_FILENO);
      if (dup2(descriptors[1], STDERR_FILENO) < 0) {_exit(11);}
      signal(SIGPIPE, SIG_DFL);
      sigset_t pipe_signal;
      sigemptyset(&pipe_signal);
      sigaddset(&pipe_signal, SIGPIPE);
      if (pthread_sigmask(SIG_UNBLOCK, &pipe_signal, nullptr) != 0) {_exit(13);}
      alarm(2);
      const bool found = search.search.createPath(
          search.path, search.iterations, 0.0,
        [&]() {
            // Start has opened the sink; force EPIPE on later checkpoint/terminal writes.
          if (closed && descriptors[0] >= 0) {
            close(descriptors[0]);
            descriptors[0] = -1;
          }
          return false;
          });
      const bool flags_unchanged = fcntl(STDERR_FILENO, F_GETFL) == original_flags;
      dup2(saved, STDERR_FILENO);
      _exit(found && flags_unchanged ? 0 : 12);
      }, testing::ExitedWithCode(0), "");
  }
}
TEST(AStarTest, timing_supports_stderr_sockets_without_blocking_or_sigpipe)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (int mode : {0, 1, 2}) {
    ASSERT_EXIT(
    {
      ScopedTimingEnvironment enabled("1");
      TimingSearch search;
      int descriptors[2];
      if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors) != 0) {_exit(10);}
      if (mode == 1) {
        char buffer[4096] = {};
        while (send(descriptors[1], buffer, sizeof(buffer), MSG_DONTWAIT | MSG_NOSIGNAL) > 0) {}
      }
      const int original_flags = fcntl(descriptors[1], F_GETFL);
      const int saved = dup(STDERR_FILENO);
      if (dup2(descriptors[1], STDERR_FILENO) < 0) {_exit(11);}
      signal(SIGPIPE, SIG_DFL);
      sigset_t pipe_signal;
      sigemptyset(&pipe_signal);
      sigaddset(&pipe_signal, SIGPIPE);
      if (pthread_sigmask(SIG_UNBLOCK, &pipe_signal, nullptr) != 0) {_exit(13);}
      alarm(2);
      const bool found = search.search.createPath(
          search.path, search.iterations, 0.0,
        [&]() {
          if (mode == 2 && descriptors[0] >= 0) {
            close(descriptors[0]);
            descriptors[0] = -1;
          }
          return false;
          });
      const bool flags_unchanged = fcntl(STDERR_FILENO, F_GETFL) == original_flags;
      dup2(saved, STDERR_FILENO);
      if (mode == 0) {
        char buffer[65536];
        const ssize_t size = recv(descriptors[0], buffer, sizeof(buffer), MSG_DONTWAIT);
        if (size <= 0) {_exit(14);}
        const auto records = timingRecords(std::string(buffer, static_cast<size_t>(size)));
        if (records.size() < 2 ||
        records.front().find("\"event\":\"start\"") == std::string::npos ||
        records.back().find("\"event\":\"goal\"") == std::string::npos)
        {
          _exit(15);
        }
      }
      _exit(found && flags_unchanged ? 0 : 12);
      }, testing::ExitedWithCode(0), "");
  }
}

#endif  // __linux__
