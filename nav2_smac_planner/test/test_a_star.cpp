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
  for (size_t i = 1; i <= 4; ++i) {
    EXPECT_GT(std::stoull(samples[i].str()), 0u) << "phase " << i - 1 << " was never sampled";
  }
  // Phase times are sums of a few sampled intervals (one pop in 64). A phase shorter than
  // one steady_clock tick reads exactly 0: the RK3588's arch timer runs at 24 MHz (41.7 ns),
  // so "> 0" failed there at random while ~1 ns clocks (x86, Thor) passed. Node2D's analytic
  // expansion is a no-op, the usual zero. The count checks above prove every phase was sampled;
  // per phase, require only sane values, and require the durations to accumulate in total.
  std::smatch elapsed_match;
  ASSERT_TRUE(std::regex_search(
      records.back(), elapsed_match, std::regex("\"elapsed_ms\":([0-9.]+)")));
  const double elapsed_ms = std::stod(elapsed_match[1].str());
  EXPECT_GT(elapsed_ms, 0.0);
  double phase_total_ms = 0.0;
  for (const char * phase : {"selection_ms", "analytic_ms", "neighbors_ms", "heuristic_queue_ms"}) {
    std::smatch duration;
    ASSERT_TRUE(std::regex_search(
        records.back(), duration, std::regex(std::string("\"") + phase + "\":([0-9.]+)")));
    const double value = std::stod(duration[1].str());
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GE(value, 0.0);
    phase_total_ms += value;
  }
  // Neighbor expansion alone (8 neighbors, collision checks, queue pushes) spans many ticks,
  // so a zero total means add() stopped accumulating, not clock resolution.
  EXPECT_GT(phase_total_ms, 0.0);
  // Sampled phases are a subset of the search, so they cannot exceed its wall time.
  EXPECT_LE(phase_total_ms, elapsed_ms);
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

#include "ompl/base/spaces/DubinsStateSpace.h"
#include "ompl/base/spaces/ReedsSheppStateSpace.h"

TEST(AStarTest, analytic_handoff_gear_seeding_and_single_gear_fallback)
{
  using Node = nav2_smac_planner::NodeHybrid;
  using TurnDir = nav2_smac_planner::TurnDirection;
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("handoff_gear_test");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("handoff_gear_map");
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto * costmap = costmap_ros->getCostmap();
  costmap->resizeMap(100, 100, 0.1, 0.0, 0.0);
  std::fill_n(
    costmap->getCharMap(), costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    nav2_costmap_2d::FREE_SPACE);

  const unsigned int size_theta = 72;
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, size_theta, lnode);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 6.0f;
  info.non_straight_penalty = 1.05f;
  info.change_penalty = 0.0f;
  info.reverse_penalty = 1.0f;
  info.cost_penalty = 2.0f;
  info.analytic_expansion_ratio = 3.5f;
  info.analytic_expansion_max_length = 500.0f;
  info.analytic_expansion_max_cost = 200.0f;
  info.gear_change_penalty = 15.0f;
  info.analytic_gear_change_deferrals = 2;

  unsigned int size_x = 100u;
  unsigned int size_y = 100u;
  unsigned int size_th = size_theta;
  Node::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_th, info);
  Node::precomputeDistanceHeuristic(
    401.0f, nav2_smac_planner::MotionModel::REEDS_SHEPP, size_theta, info);
  Node::resetObstacleHeuristic(costmap_ros, 50u, 50u, 35u, 50u, costmap);
  EXPECT_FLOAT_EQ(Node::motion_table.gear_change_penalty, 15.0f);

  // Verify NodeHybrid::getTraversalCost does not charge gear_change_penalty from UNKNOWN turn_dir
  // or same-gear REVERSE -> REVERSE transition, but DOES charge gear_change_penalty on a
  // FORWARD -> REVERSE transition.
  Node unknown_parent(Node::getIndex(50u, 50u, 0u));
  unknown_parent.setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  unknown_parent.setMotionPrimitiveIndex(0u, TurnDir::UNKNOWN);
  ASSERT_TRUE(unknown_parent.isNodeValid(false, &checker));
  Node rev_parent(Node::getIndex(50u, 50u, 0u));
  rev_parent.setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  rev_parent.setMotionPrimitiveIndex(3u, TurnDir::REVERSE);
  ASSERT_TRUE(rev_parent.isNodeValid(false, &checker));
  Node fwd_child(Node::getIndex(51u, 50u, 0u));
  fwd_child.setPose(Node::Coordinates(51.0f, 50.0f, 0.0f));
  fwd_child.setMotionPrimitiveIndex(0u, TurnDir::FORWARD);
  ASSERT_TRUE(fwd_child.isNodeValid(false, &checker));
  Node rev_child(Node::getIndex(49u, 50u, 0u));
  rev_child.setPose(Node::Coordinates(49.0f, 50.0f, 0.0f));
  rev_child.setMotionPrimitiveIndex(3u, TurnDir::REVERSE);
  ASSERT_TRUE(rev_child.isNodeValid(false, &checker));

  const float unknown_to_rev = unknown_parent.getTraversalCost(&rev_child);
  const float rev_to_rev = rev_parent.getTraversalCost(&rev_child);
  const float fwd_to_rev = fwd_child.getTraversalCost(&rev_child);
  EXPECT_NEAR(unknown_to_rev, rev_to_rev, 1e-4f);
  EXPECT_NEAR(fwd_to_rev - rev_to_rev, info.gear_change_penalty, 1e-4f);
  EXPECT_NEAR(fwd_to_rev - unknown_to_rev, info.gear_change_penalty, 1e-4f);

  nav2_smac_planner::AStarAlgorithm<Node>::Graph graph;
  const uint64_t max_index =
    static_cast<uint64_t>(size_x) * static_cast<uint64_t>(size_y) * static_cast<uint64_t>(size_th);
  auto getter = [&graph, max_index](const uint64_t & index, Node * & node_rtn) -> bool {
      if (index >= max_index) {
        return false;
      }
      auto it = graph.find(index);
      if (it == graph.end()) {
        node_rtn = &(graph.emplace(index, Node(index)).first->second);
        return true;
      }
      node_rtn = &(it->second);
      return true;
    };

  nav2_smac_planner::AnalyticExpansion<Node> expansion(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  expansion.setCollisionChecker(&checker);

  std::vector<std::string> trace_records;
  expansion.setTraceCallback([&trace_records](const std::string & rec) {
      trace_records.push_back(rec);
    });

  auto has_record_with = [](const std::vector<std::string> & recs, const std::string & substr) {
      for (const auto & r : recs) {
        if (r.find(substr) != std::string::npos) {
          return true;
        }
      }
      return false;
    };

  Node * goal_node = nullptr;
  getter(Node::getIndex(35u, 50u, 0u), goal_node);
  goal_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));

  // Case A: Start node (TurnDirection::UNKNOWN -> initial_gear = 0) to pure-reverse goal.
  // Must count 0 gear changes and accept immediately without deferral.
  Node * start_node = nullptr;
  getter(Node::getIndex(50u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  int iterations = 0;
  int best_cost = 1000;
  trace_records.clear();
  expansion.setTraceRequestId(1);
  Node * res_start = expansion.tryAnalyticExpansion(
    start_node, goal_node, getter, iterations, best_cost);
  ASSERT_NE(res_start, nullptr);
  EXPECT_TRUE(has_record_with(trace_records, "\"gear_changes\":0"));
  EXPECT_TRUE(has_record_with(trace_records, "\"selected_by\":\"gear_aware_min_score\""));

  // Case B: Forward A* node (TurnDirection::FORWARD -> initial_gear = +1, parent != nullptr)
  // on open map. The direct reverse shot has 1 handoff gear change (+1 -> -1) and lower score,
  // while the forward_only Dubins loop has 0 gear changes (+1 -> +1) and higher score.
  // On deferral 1, tryAnalyticExpansion must fall back to best_single_gear (single_gear_forward).
  graph.clear();
  getter(Node::getIndex(49u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(49.0f, 50.0f, 0.0f));
  getter(Node::getIndex(35u, 50u, 0u), goal_node);
  goal_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
  Node * fwd_node = nullptr;
  getter(Node::getIndex(50u, 50u, 0u), fwd_node);
  fwd_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  fwd_node->setMotionPrimitiveIndex(0u, TurnDir::FORWARD);
  fwd_node->parent = start_node;

  trace_records.clear();
  expansion.setTraceRequestId(2);
  iterations = 0;
  best_cost = 1000;
  Node * res_fallback = expansion.tryAnalyticExpansion(
    fwd_node, goal_node, getter, iterations, best_cost);
  ASSERT_NE(res_fallback, nullptr);
  EXPECT_TRUE(
    has_record_with(trace_records, "\"selected_by\":\"gear_aware_single_gear_fallback\""));
  EXPECT_TRUE(has_record_with(trace_records, "\"kind\":\"single_gear_forward\""));
  // Verify the selected path steps forward from fwd_node (0 gear changes).
  std::vector<Node::Coordinates> chain;
  for (Node * p = res_fallback; p != nullptr && p != start_node; p = p->parent) {
    chain.push_back(p->pose);
  }
  std::reverse(chain.begin(), chain.end());
  ASSERT_GE(chain.size(), 4u);
  for (size_t i = 1; i < chain.size(); ++i) {
    const float dx = chain[i].x - chain[i - 1].x;
    const float dy = chain[i].y - chain[i - 1].y;
    if (std::hypot(dx, dy) > 1e-3f) {
      const float h = Node::motion_table.getAngleFromBin(chain[i - 1].theta);
      EXPECT_GE(dx * std::cos(h) + dy * std::sin(h), -1e-3f)
        << "Fallback single-gear forward path reversed at step " << i;
    }
  }

  // Case C: Block all forward motion at x in [51, 54] so no 0-gear-change path exists.
  // Now deferral must return nullptr twice (deferrals_used: 0->1, 1->2) and accept on call 3.
  for (unsigned int x = 51u; x <= 54u; ++x) {
    for (unsigned int y = 10u; y <= 90u; ++y) {
      costmap->setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }
  expansion.setTraceRequestId(3);

  for (int attempt = 1; attempt <= 2; ++attempt) {
    graph.clear();
    getter(Node::getIndex(49u, 50u, 0u), start_node);
    start_node->setPose(Node::Coordinates(49.0f, 50.0f, 0.0f));
    getter(Node::getIndex(35u, 50u, 0u), goal_node);
    goal_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
    getter(Node::getIndex(50u, 50u, 0u), fwd_node);
    fwd_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
    fwd_node->setMotionPrimitiveIndex(0u, TurnDir::FORWARD);
    fwd_node->parent = start_node;
    trace_records.clear();
    iterations = 0;
    best_cost = 1000;
    Node * deferred_res = expansion.tryAnalyticExpansion(
      fwd_node, goal_node, getter, iterations, best_cost);
    EXPECT_EQ(deferred_res, nullptr) << "Attempt " << attempt << " should have been deferred";
    EXPECT_TRUE(has_record_with(trace_records, "\"gear_changes\":1"));
    EXPECT_FALSE(has_record_with(trace_records, "\"event\":\"selection\""));
  }

  graph.clear();
  getter(Node::getIndex(49u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(49.0f, 50.0f, 0.0f));
  getter(Node::getIndex(35u, 50u, 0u), goal_node);
  goal_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
  getter(Node::getIndex(50u, 50u, 0u), fwd_node);
  fwd_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  fwd_node->setMotionPrimitiveIndex(0u, TurnDir::FORWARD);
  fwd_node->parent = start_node;
  trace_records.clear();
  iterations = 0;
  best_cost = 1000;
  Node * accepted_res = expansion.tryAnalyticExpansion(
    fwd_node, goal_node, getter, iterations, best_cost);
  ASSERT_NE(accepted_res, nullptr) << "Attempt 3 after exhausting 2 deferrals must succeed";
  EXPECT_TRUE(has_record_with(trace_records, "\"selected_by\":\"gear_aware_min_score\""));

  Node::destroyStaticAssets();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}

TEST(AStarTest, analytic_reverse_only_dubins_and_strict_tie_break_and_quadratic_cost)
{
  using Node = nav2_smac_planner::NodeHybrid;
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("rev_dubins_test");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("rev_dubins_map");
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto * costmap = costmap_ros->getCostmap();
  costmap->resizeMap(100, 100, 0.1, 0.0, 0.0);
  std::fill_n(
    costmap->getCharMap(), costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    nav2_costmap_2d::FREE_SPACE);

  const unsigned int size_theta = 72;
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, size_theta, lnode);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 6.0f;
  info.non_straight_penalty = 1.0f;
  info.change_penalty = 0.0f;
  info.reverse_penalty = 1.0f;
  info.cost_penalty = 4.0f;
  info.analytic_expansion_ratio = 3.5f;
  info.analytic_expansion_max_length = 500.0f;
  info.analytic_expansion_max_cost = 250.0f;
  info.gear_change_penalty = 100.0f;
  info.analytic_gear_change_deferrals = 0;

  unsigned int size_x = 100u;
  unsigned int size_y = 100u;
  unsigned int size_th = size_theta;
  Node::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_th, info);
  Node::precomputeDistanceHeuristic(
    401.0f, nav2_smac_planner::MotionModel::REEDS_SHEPP, size_theta, info);
  Node::resetObstacleHeuristic(costmap_ros, 50u, 50u, 50u, 56u, costmap);

  nav2_smac_planner::AStarAlgorithm<Node>::Graph graph;
  const uint64_t max_index =
    static_cast<uint64_t>(size_x) * static_cast<uint64_t>(size_y) * static_cast<uint64_t>(size_th);
  auto getter = [&graph, max_index](const uint64_t & index, Node * & node_rtn) -> bool {
      if (index >= max_index) {
        return false;
      }
      auto it = graph.find(index);
      if (it == graph.end()) {
        node_rtn = &(graph.emplace(index, Node(index)).first->second);
        return true;
      }
      node_rtn = &(it->second);
      return true;
    };

  nav2_smac_planner::AnalyticExpansion<Node> expansion(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  expansion.setCollisionChecker(&checker);

  // 1. Direct test of getAnalyticPath(..., dubins_space, reverse_only = true vs false)
  ompl::base::StateSpacePtr dubins_space =
    std::make_shared<ompl::base::DubinsStateSpace>(info.minimum_turning_radius);
  Node * start_node = nullptr;
  Node * behind_goal = nullptr;
  getter(Node::getIndex(50u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  getter(Node::getIndex(38u, 50u, 0u), behind_goal);
  behind_goal->setPose(Node::Coordinates(38.0f, 50.0f, 0.0f));

  const auto fwd_dubins_nodes =
    expansion.getAnalyticPath(start_node, behind_goal, getter, dubins_space, false);
  graph.clear();
  getter(Node::getIndex(50u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  getter(Node::getIndex(38u, 50u, 0u), behind_goal);
  behind_goal->setPose(Node::Coordinates(38.0f, 50.0f, 0.0f));
  const auto rev_dubins_nodes =
    expansion.getAnalyticPath(start_node, behind_goal, getter, dubins_space, true);

  ASSERT_FALSE(fwd_dubins_nodes.empty());
  ASSERT_FALSE(rev_dubins_nodes.empty());
  EXPECT_GT(fwd_dubins_nodes.size(), rev_dubins_nodes.size());
  float prev_x = 50.0f;
  for (const auto & n : rev_dubins_nodes) {
    EXPECT_LT(n.proposed_coords.x, prev_x);
    prev_x = n.proposed_coords.x;
  }

  // 2. Strict '<' tie-breaking: from (50, 50, bin 0) to (50, 56, bin 36 = pi, dy = R = 6.0):
  // Reeds-Shepp uses a 1-gear-change cusp (penalized by gear_change_penalty = 100.0),
  // whereas forward_only Dubins and reverse_only Dubins are symmetric 0-gear-change curves
  // with identical length and score. Strict '<' must keep single_gear_forward over reverse.
  graph.clear();
  Node * sym_start = nullptr;
  Node * sym_goal = nullptr;
  getter(Node::getIndex(50u, 50u, 0u), sym_start);
  sym_start->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  getter(Node::getIndex(50u, 56u, 36u), sym_goal);
  sym_goal->setPose(Node::Coordinates(50.0f, 56.0f, 36.0f));

  std::vector<std::string> trace_records;
  expansion.setTraceCallback([&trace_records](const std::string & rec) {
      trace_records.push_back(rec);
    });
  expansion.setTraceRequestId(10);
  int iterations = 0;
  int best_cost = 1000;
  Node * sym_res = expansion.tryAnalyticExpansion(
    sym_start, sym_goal, getter, iterations, best_cost);
  ASSERT_NE(sym_res, nullptr);
  // Check that the first step from sym_start moves forward (+x), not backward (-x).
  std::vector<Node::Coordinates> sym_chain;
  for (Node * p = sym_res; p != nullptr; p = p->parent) {
    sym_chain.push_back(p->pose);
  }
  std::reverse(sym_chain.begin(), sym_chain.end());
  ASSERT_GE(sym_chain.size(), 3u);
  EXPECT_GT(sym_chain[1].x, sym_chain[0].x)
    << "Strict '<' tie-break must select forward Dubins (+x), not reverse Dubins (-x)";

  // 3. Quadratic vs linear cost penalty in gear_aware scoring:
  // Fill uniform cost = 126 (normalized = 126/252 = 0.5) across the map.
  // With cost_penalty = 4.0:
  //   linear multiplier per step    = 1 + 4.0 * 0.5  = 3.0
  //   quadratic multiplier per step = 1 + 4.0 * 0.25 = 2.0
  std::fill_n(
    costmap->getCharMap(), costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    static_cast<unsigned char>(126));

  auto extract_first_score = [](const std::vector<std::string> & recs) -> double {
      std::smatch m;
      for (const auto & r : recs) {
        if (r.find("\"event\":\"score\"") != std::string::npos &&
          std::regex_search(r, m, std::regex("\"score\":([0-9.]+)")))
        {
          return std::stod(m[1].str());
        }
      }
      return -1.0;
    };

  info.gear_change_penalty = 1.0f;
  info.use_quadratic_cost_penalty = false;
  nav2_smac_planner::AnalyticExpansion<Node> exp_lin(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  exp_lin.setCollisionChecker(&checker);
  trace_records.clear();
  exp_lin.setTraceCallback([&trace_records](const std::string & rec) {
      trace_records.push_back(rec);
    });
  graph.clear();
  getter(Node::getIndex(35u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
  getter(Node::getIndex(50u, 50u, 0u), behind_goal);
  behind_goal->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  iterations = 0;
  best_cost = 1000;
  ASSERT_NE(
    exp_lin.tryAnalyticExpansion(start_node, behind_goal, getter, iterations, best_cost),
    nullptr);
  const double score_linear = extract_first_score(trace_records);
  ASSERT_GT(score_linear, 0.0);

  info.use_quadratic_cost_penalty = true;
  nav2_smac_planner::AnalyticExpansion<Node> exp_quad(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  exp_quad.setCollisionChecker(&checker);
  trace_records.clear();
  exp_quad.setTraceCallback([&trace_records](const std::string & rec) {
      trace_records.push_back(rec);
    });
  graph.clear();
  getter(Node::getIndex(35u, 50u, 0u), start_node);
  start_node->setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
  getter(Node::getIndex(50u, 50u, 0u), behind_goal);
  behind_goal->setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  iterations = 0;
  best_cost = 1000;
  ASSERT_NE(
    exp_quad.tryAnalyticExpansion(start_node, behind_goal, getter, iterations, best_cost),
    nullptr);
  const double score_quad = extract_first_score(trace_records);
  ASSERT_GT(score_quad, 0.0);

  EXPECT_NEAR(score_quad * 3.0, score_linear * 2.0, 1e-3)
    << "score_linear=" << score_linear << " score_quad=" << score_quad;

  // 4. Default-off (gear_change_penalty = 0.0, analytic_gear_change_deferrals = 0) per-sample
  // cell_cost preservation: when two samples in getAnalyticPath map to the same NodeHybrid*
  // (e.g. a node alias / shared bin) and the second sample's isNodeValid overwrites node->_cost,
  // scoringFn must still sum each sample's saved cell_cost rather than the final overwritten
  // node->getCost().
  info.gear_change_penalty = 0.0f;
  info.analytic_gear_change_deferrals = 0;
  info.use_quadratic_cost_penalty = false;
  nav2_smac_planner::AnalyticExpansion<Node> exp_default_off(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  exp_default_off.setCollisionChecker(&checker);
  trace_records.clear();
  exp_default_off.setTraceCallback([&trace_records](const std::string & rec) {
      trace_records.push_back(rec);
    });
  std::fill_n(
    costmap->getCharMap(), costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    nav2_costmap_2d::FREE_SPACE);
  // Paint moderate cost (189 <= max_cost 250) on x in [36, 42] and free space (0) on x > 42,
  // and alias all intermediate lookups to a single shared NodeHybrid instance so later
  // free-space samples overwrite shared_node._cost = 0.
  for (unsigned int x = 36u; x <= 42u; ++x) {
    costmap->setCost(x, 50u, 189);
  }
  Node shared_node(Node::getIndex(36u, 50u, 0u));
  Node start_35(Node::getIndex(35u, 50u, 0u));
  start_35.setPose(Node::Coordinates(35.0f, 50.0f, 0.0f));
  Node goal_50(Node::getIndex(50u, 50u, 0u));
  goal_50.setPose(Node::Coordinates(50.0f, 50.0f, 0.0f));
  auto aliasing_getter = [&shared_node](const uint64_t & /*index*/, Node * & node_rtn) -> bool {
      node_rtn = &shared_node;
      return true;
    };
  iterations = 0;
  best_cost = 1000;
  ASSERT_NE(
    exp_default_off.tryAnalyticExpansion(
      &start_35, &goal_50, aliasing_getter, iterations, best_cost),
    nullptr);
  EXPECT_FLOAT_EQ(shared_node.getCost(), 0.0f);
  const double score_aliased = extract_first_score(trace_records);
  // Pure distance score (if all samples had cost 0) is ~15.0; with 7 samples at cost 252
  // (weight = 4.0), preserved cell_cost adds ~28.0 * step_distance (> 30.0 total).
  EXPECT_GT(score_aliased, 30.0)
    << "Default-off scoringFn must use per-sample saved cell_cost, not overwritten node->getCost()";

  Node::destroyStaticAssets();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}

TEST(AStarTest, analytic_path_rejects_negative_and_upper_bounds_and_wraps_two_pi)
{
  using Node = nav2_smac_planner::NodeHybrid;
  auto lnode = std::make_shared<rclcpp_lifecycle::LifecycleNode>("bounds_wrap_test");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("bounds_wrap_map");
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->on_configure(rclcpp_lifecycle::State());
  auto * costmap = costmap_ros->getCostmap();
  costmap->resizeMap(30, 30, 0.1, 0.0, 0.0);
  std::fill_n(
    costmap->getCharMap(), costmap->getSizeInCellsX() * costmap->getSizeInCellsY(),
    nav2_costmap_2d::FREE_SPACE);

  const unsigned int size_theta = 72;
  nav2_smac_planner::GridCollisionChecker checker(costmap_ros, size_theta, lnode);
  checker.setFootprint(nav2_costmap_2d::Footprint(), true, 0.0);

  nav2_smac_planner::SearchInfo info;
  info.minimum_turning_radius = 8.0f;
  info.analytic_expansion_max_length = 500.0f;
  info.analytic_expansion_ratio = 3.5f;

  unsigned int size_x = 30u;
  unsigned int size_y = 30u;
  unsigned int size_th = size_theta;
  Node::initMotionModel(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, size_x, size_y, size_th, info);

  nav2_smac_planner::AStarAlgorithm<Node>::Graph graph;
  const uint64_t max_index =
    static_cast<uint64_t>(size_x) * static_cast<uint64_t>(size_y) * static_cast<uint64_t>(size_th);
  auto getter = [&graph, max_index](const uint64_t & index, Node * & node_rtn) -> bool {
      if (index >= max_index) {
        return false;
      }
      auto it = graph.find(index);
      if (it == graph.end()) {
        node_rtn = &(graph.emplace(index, Node(index)).first->second);
        return true;
      }
      node_rtn = &(it->second);
      return true;
    };

  nav2_smac_planner::AnalyticExpansion<Node> expansion(
    nav2_smac_planner::MotionModel::REEDS_SHEPP, info, false, size_theta);
  expansion.setCollisionChecker(&checker);
  ompl::base::StateSpacePtr dubins_space =
    std::make_shared<ompl::base::DubinsStateSpace>(info.minimum_turning_radius);

  // 1a. Negative x excursion in (-1.0, 0.0):
  // Start at (0.3, 10.0, bin 27 = 135 deg) -> Goal at (0.3, 15.0, bin 9 = 45 deg).
  // The arc dips to x ~ -0.45. Without reals[0] < 0.0, static_cast<unsigned int>(-0.45)
  // truncates toward zero to 0u (which is inside [0, 30) and free!) and falsely succeeds.
  Node * neg_x_start = nullptr;
  Node * neg_x_goal = nullptr;
  getter(Node::getIndex(0u, 10u, 27u), neg_x_start);
  neg_x_start->setPose(Node::Coordinates(0.3f, 10.0f, 27.0f));
  getter(Node::getIndex(0u, 15u, 9u), neg_x_goal);
  neg_x_goal->setPose(Node::Coordinates(0.3f, 15.0f, 9.0f));

  const auto neg_x_nodes =
    expansion.getAnalyticPath(neg_x_start, neg_x_goal, getter, dubins_space, false);
  EXPECT_TRUE(neg_x_nodes.empty())
    << "Curve dipping into negative x (-0.45) must be rejected before unsigned cast";

  // 1b. Deep negative x excursion (x <= -1.0, e.g. x ~ -1.8):
  // Start at (0.2, 8.0, bin 27 = 135 deg) -> Goal at (0.2, 18.0, bin 9 = 45 deg).
  // Without reals[0] < 0.0, static_cast<unsigned int>(-1.8) invokes UB / wraps to UINT_MAX.
  graph.clear();
  Node * deep_neg_x_start = nullptr;
  Node * deep_neg_x_goal = nullptr;
  getter(Node::getIndex(0u, 8u, 27u), deep_neg_x_start);
  deep_neg_x_start->setPose(Node::Coordinates(0.2f, 8.0f, 27.0f));
  getter(Node::getIndex(0u, 18u, 9u), deep_neg_x_goal);
  deep_neg_x_goal->setPose(Node::Coordinates(0.2f, 18.0f, 9.0f));

  const auto deep_neg_x_nodes =
    expansion.getAnalyticPath(deep_neg_x_start, deep_neg_x_goal, getter, dubins_space, false);
  EXPECT_TRUE(deep_neg_x_nodes.empty())
    << "Curve dipping into deep negative x (-1.8) must be rejected before unsigned cast";
  EXPECT_EQ(graph.size(), 2u)
    << "No out-of-bounds or wrapped graph nodes may be allocated during negative excursion";

  // 2. Negative y excursion in (-1.0, 0.0):
  // Start at (10.0, 0.3, bin 63 = 315 deg) -> Goal at (15.0, 0.3, bin 9 = 45 deg).
  graph.clear();
  Node * neg_y_start = nullptr;
  Node * neg_y_goal = nullptr;
  getter(Node::getIndex(10u, 0u, 63u), neg_y_start);
  neg_y_start->setPose(Node::Coordinates(10.0f, 0.3f, 63.0f));
  getter(Node::getIndex(15u, 0u, 9u), neg_y_goal);
  neg_y_goal->setPose(Node::Coordinates(15.0f, 0.3f, 9.0f));

  const auto neg_y_nodes =
    expansion.getAnalyticPath(neg_y_start, neg_y_goal, getter, dubins_space, false);
  EXPECT_TRUE(neg_y_nodes.empty())
    << "Curve dipping into negative y (-0.45) must be rejected before unsigned cast";

  // 3. Heading wrap across 2 * M_PI (bin 71 = 355 deg -> bin 1 = 5 deg, and reverse_only +M_PI):
  // Ensure every node's orientation bin is strictly < size_theta (72) and in [0, size_theta).
  graph.clear();
  Node * wrap_start = nullptr;
  Node * wrap_goal = nullptr;
  getter(Node::getIndex(10u, 15u, 71u), wrap_start);
  wrap_start->setPose(Node::Coordinates(10.0f, 15.0f, 71.0f));
  getter(Node::getIndex(20u, 15u, 1u), wrap_goal);
  wrap_goal->setPose(Node::Coordinates(20.0f, 15.0f, 1.0f));

  const auto wrap_nodes =
    expansion.getAnalyticPath(wrap_start, wrap_goal, getter, dubins_space, false);
  ASSERT_FALSE(wrap_nodes.empty());
  for (const auto & n : wrap_nodes) {
    const uint64_t idx = n.node->getIndex();
    const uint64_t theta_bin = idx % size_theta;
    EXPECT_LT(theta_bin, size_theta);
    EXPECT_GE(n.proposed_coords.theta, 0.0f);
    EXPECT_LT(n.proposed_coords.theta, static_cast<float>(size_theta));
    const Node::Coordinates decoded = Node::getCoords(idx, size_x, size_theta);
    EXPECT_LT(decoded.x, static_cast<float>(size_x));
    EXPECT_LT(decoded.y, static_cast<float>(size_y));
  }

  // Also test reverse_only = true backing straight North (+y) while facing South (bin 54 = 270 deg)
  // and backing straight West (-x) while facing East (bin 0 = 0 deg, where OMPL Dubins yaw is pi
  // so reals[2] + M_PI == 2 * M_PI and must wrap to bin 0, never bin 72).
  graph.clear();
  Node * rev_wrap_start = nullptr;
  Node * rev_wrap_goal = nullptr;
  getter(Node::getIndex(20u, 15u, 0u), rev_wrap_start);
  rev_wrap_start->setPose(Node::Coordinates(20.0f, 15.0f, 0.0f));
  getter(Node::getIndex(10u, 15u, 0u), rev_wrap_goal);
  rev_wrap_goal->setPose(Node::Coordinates(10.0f, 15.0f, 0.0f));
  const auto rev_wrap_nodes =
    expansion.getAnalyticPath(rev_wrap_start, rev_wrap_goal, getter, dubins_space, true);
  ASSERT_FALSE(rev_wrap_nodes.empty());
  for (const auto & n : rev_wrap_nodes) {
    EXPECT_GE(n.proposed_coords.theta, 0.0f);
    EXPECT_LT(n.proposed_coords.theta, static_cast<float>(size_theta));
  }

  Node::destroyStaticAssets();
  costmap_ros->on_cleanup(rclcpp_lifecycle::State());
}
