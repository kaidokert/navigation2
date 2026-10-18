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
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <unordered_map>
#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>

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

TEST(AStarTest, analytic_expansion_retains_recorded_reverse_cusp)
{
  using namespace nav2_smac_planner;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("cusp_test");
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>();
  cm->on_configure(rclcpp_lifecycle::State());
  cm->getCostmap()->resizeMap(200,200,0.02,0,0);
  GridCollisionChecker checker(cm,72,node);
  checker.setFootprint(nav2_costmap_2d::Footprint(),true,0.0);
  SearchInfo info;
  info.minimum_turning_radius = 27.5;info.analytic_expansion_max_length = 1000;
  unsigned int bins = 72, width = 200, height = 200;
  NodeHybrid::motion_table.initReedsShepp(width,height,bins,info);
  NodeHybrid start(NodeHybrid::getIndex(100,100,56));
  NodeHybrid goal(NodeHybrid::getIndex(108,71,51));
  start.pose = {100.0f,100.0f,56.0f};
  goal.pose = {static_cast<float>(100+(2.939681662343105-2.767846644211261)/.02),
    static_cast<float>(100+(-1.0526122190517526+0.4875475344163078)/.02),51.0f};
  std::unordered_map<uint64_t,std::unique_ptr<NodeHybrid>> graph;
  const auto getter = [&](const uint64_t & index, NodeHybrid * & result) {
      if (!graph.count(index)) {graph[index] = std::make_unique<NodeHybrid>(index);}
      result = graph[index].get();return true;
    };
  AnalyticExpansion<NodeHybrid> expansion(MotionModel::REEDS_SHEPP,info,false,bins);
  expansion.setCollisionChecker(&checker);
  auto samples = expansion.getAnalyticPath(&start,&goal,getter,NodeHybrid::motion_table.state_space);
  ASSERT_FALSE(samples.empty());
  auto rs = std::dynamic_pointer_cast<ompl::base::ReedsSheppStateSpace>(NodeHybrid::motion_table.state_space);
  ompl::base::ScopedState<ompl::base::SE2StateSpace> from(rs),to(rs),cusp(rs);
  from->setXY(start.pose.x,start.pose.y);from->setYaw(56*2*M_PI/72);
  to->setXY(goal.pose.x,goal.pose.y);to->setYaw(51*2*M_PI/72);
  const auto path = rs->reedsShepp(from(),to());
  auto tail = expansion.setAnalyticPath(&start,&goal,samples);
  std::vector<NodeHybrid::Coordinates> linked_poses;
  for (size_t steps=0;tail && steps<=samples.size()+2;++steps) {
    linked_poses.push_back(tail->pose);
    if (tail == &start) {tail = nullptr;break;}
    tail = tail->parent;
  }
  EXPECT_EQ(tail,nullptr) << "Analytic parent chain must terminate without a cycle";
  double progress = 0;size_t cusps = 0;
  for (int k=0;k<4;++k) {
    progress += std::abs(path.length_[k]);
    if (path.length_[k]*path.length_[k+1] >= 0) {continue;}
    ++cusps;rs->interpolate(from(),to(),progress/path.length(),cusp());
    double closest = std::numeric_limits<double>::max();
    for (const auto & s : samples) {
      closest = std::min(closest,std::hypot(s.proposed_coords.x-cusp->getX(),
        s.proposed_coords.y-cusp->getY()));
    }
    EXPECT_LT(closest,1e-3) << "Exact reversal cusp must survive analytic sampling";
    closest = std::numeric_limits<double>::max();
    for (const auto & pose : linked_poses) {
      closest = std::min(closest,std::hypot(pose.x-cusp->getX(),pose.y-cusp->getY()));
    }
    EXPECT_LT(closest,1e-3) << "Exact reversal cusp must survive graph linking";
    // A lethal obstacle at the cusp must reject the analytic shortcut,
    // including on a second probe against graph nodes from a free-space probe.
    // The collision checker rounds cells; this is not a claim that the cusp
    // necessarily occupies a different cell from its neighboring samples.
    const auto cx = static_cast<unsigned int>(cusp->getX()+0.5);
    const auto cy = static_cast<unsigned int>(cusp->getY()+0.5);
    cm->getCostmap()->setCost(cx,cy,nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_TRUE(expansion.getAnalyticPath(&start,&goal,getter,
      NodeHybrid::motion_table.state_space).empty());
    cm->getCostmap()->setCost(cx,cy,nav2_costmap_2d::FREE_SPACE);
  }
  EXPECT_GT(cusps,0u);
  NodeHybrid::destroyStaticAssets();
}

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
  EXPECT_NEAR(path.size(), 63u, 2u);
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
