// Copyright (c) 2026 kaidokert
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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{
using mppi::critics::ObstaclesCritic;

class FootprintWitness : public ObstaclesCritic
{
public:
  float threshold() const {return possible_collision_cost_;}
  mppi::critics::CollisionCost sample(float x, float y, float yaw)
  {
    return costAtPose(x, y, yaw);
  }
  void buildScoreCache()
  {
    if (consider_footprint_ || collision_check_footprint_only_) {
      possible_collision_cost_ = findCircumscribedCost(costmap_ros_);
    }
    prepareScoreCache();
  }
  mppi::critics::CollisionCost sampleCached(float x, float y, float yaw)
  {
    score_cache_active_ = true;
    const auto res = costAtPose(x, y, yaw);
    score_cache_active_ = false;
    return res;
  }
  float distanceUncached(const mppi::critics::CollisionCost & c)
  {
    score_cache_active_ = false;
    return distanceToObstacle(c);
  }
  float distanceCached(const mppi::critics::CollisionCost & c)
  {
    score_cache_active_ = true;
    const float res = distanceToObstacle(c);
    score_cache_active_ = false;
    return res;
  }
};

class CollisionCriticTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("collision_regression");
    node->declare_parameter("critic.consider_footprint", false);
    node->declare_parameter("critic.collision_check_footprint_only", true);
    node->declare_parameter("critic.repulsion_weight", 0.0);
    node->declare_parameter("critic.collision_margin_distance", 0.0);
    node->declare_parameter("critic.goal_truncation_distance", 0.25);
    cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "collision_map", "", "collision_map", true);
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{"inflation_layer"}));
    cm->declare_parameter("inflation_layer.plugin", std::string("nav2_costmap_2d::InflationLayer"));
    cm->declare_parameter("inflation_layer.inflation_radius", 0.20);
    cm->declare_parameter("inflation_layer.cost_scaling_factor", 6.0);
    cm->set_parameter(rclcpp::Parameter("footprint", std::string(
        "[[0.170,0.151],[0.170,-0.151],[-0.156,-0.151],[-0.156,0.151]]")));
    cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.012));
    ASSERT_EQ(cm->on_configure(lifecycle), nav2_util::CallbackReturn::SUCCESS);
    cm->pause();
    cm->getLayeredCostmap()->resizeMap(
      60, 60, 0.05000000074505806, -1.1000000003725297, -1.4500000003725289);
    auto * grid = cm->getCostmap();
    std::fill(grid->getCharMap(), grid->getCharMap() + 3600, 0);
    handler = std::make_unique<mppi::ParametersHandler>(node);
  }

  void TearDown() override
  {
    handler.reset();
    cm->on_cleanup(lifecycle);
    cm.reset();
    node.reset();
  }

  void configure(FootprintWitness & critic)
  {
    critic.on_configure(node, "mppi", "critic", cm, handler.get());
  }

  void cornerObstacle()
  {
    // Physical robot bag run_20261018_195510, output at 1792353513.3058.
    // The centre cell is free; this cell is touched by a rasterized corner.
    cm->getCostmap()->setCost(30, 42, nav2_costmap_2d::LETHAL_OBSTACLE);
  }

  void score(const std::vector<float> & xs, bool collision, bool all_colliding = false)
  {
    FootprintWitness critic;
    configure(critic);
    mppi::models::State state;
    state.reset(2, xs.size());
    mppi::models::Trajectories trajectories;
    trajectories.reset(2, xs.size());
    trajectories.y.fill(0.0f);
    trajectories.yaws.fill(0.0f);
    for (size_t j = 0; j < xs.size(); ++j) {
      trajectories.x(0, j) = xs[j];
      trajectories.x(1, j) = xs[j];
      trajectories.y(1, j) = all_colliding ? 0.0f : -0.75f;
    }
    mppi::models::Path path;
    path.reset(3);
    geometry_msgs::msg::Pose goal;
    goal.position.x = 0.5;
    goal.orientation.w = 1.0;
    xt::xtensor<float, 1> costs = xt::zeros<float>({2});
    float dt = 0.05;
    mppi::CriticData data{state, trajectories, path, goal, costs, dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    critic.score(data);
    if (collision) {
      EXPECT_GE(costs(0), 1000000.0f);
    } else {
      EXPECT_FLOAT_EQ(costs(0), 0.0f);
    }
    if (!all_colliding) {EXPECT_FLOAT_EQ(costs(1), 0.0f);}
    EXPECT_EQ(data.fail_flag, all_colliding && collision);
  }

  void wall()
  {
    auto * grid = cm->getCostmap();
    for (unsigned int y = 26; y <= 32; ++y) {
      // Column 36 centres around x=0.725, ahead of the goal at x=0.5.
      grid->setCost(36, y, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
  }

  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
  std::unique_ptr<mppi::ParametersHandler> handler;
  rclcpp_lifecycle::State lifecycle;
};

TEST_F(CollisionCriticTest, RecordedCornerVetoWithFreeCentre)
{
  cornerObstacle();
  FootprintWitness critic;
  configure(critic);
  ASSERT_GT(critic.threshold(), 0.0f);
  const auto cost = critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f);
  EXPECT_EQ(cost.cost, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_TRUE(cost.using_footprint);
}

TEST_F(CollisionCriticTest, RecordedCornerVetoWithFullFootprintSoftScoring)
{
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", true));
  cornerObstacle();
  FootprintWitness critic;
  configure(critic);
  EXPECT_EQ(critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f).cost,
    nav2_costmap_2d::LETHAL_OBSTACLE);
}

TEST_F(CollisionCriticTest, ClearPerimeterCannotEraseLethalCentre)
{
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", true));
  cm->getCostmap()->setCost(32, 37, nav2_costmap_2d::LETHAL_OBSTACLE);
  FootprintWitness critic;
  configure(critic);
  EXPECT_EQ(critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f).cost,
    nav2_costmap_2d::LETHAL_OBSTACLE);
}

TEST_F(CollisionCriticTest, DisabledHardVetoReproducesThresholdBlindSpot)
{
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", false));
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", true));
  cornerObstacle();
  FootprintWitness critic;
  configure(critic);
  EXPECT_FLOAT_EQ(critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f).cost, 0.0f);
}

TEST_F(CollisionCriticTest, HardVetoRetainsPointSoftCostWhenPerimeterIsClear)
{
  cm->getCostmap()->setCost(32, 37, 49);
  FootprintWitness critic;
  configure(critic);
  const auto cost = critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f);
  EXPECT_FLOAT_EQ(cost.cost, 49.0f);
  EXPECT_FALSE(cost.using_footprint);
}

TEST_F(CollisionCriticTest, HardVetoRequiresRectangle)
{
  // Radius mode is selected by configuration, not by supplying an empty polygon
  // to a running polygon costmap. Exercise it with a separately configured map.
  auto round = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "round_map", "", "round_map", true);
  round->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  round->on_configure(lifecycle);
  round->pause();
  FootprintWitness critic;
  EXPECT_THROW(critic.on_configure(node, "mppi", "critic", round, handler.get()),
    nav2_core::ControllerException);
  round->on_cleanup(lifecycle);
}

TEST_F(CollisionCriticTest, DynamicActivationOnRadiusMapIsRejected)
{
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", false));
  auto round = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dynamic_round_map", "", "dynamic_round_map", true);
  round->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  ASSERT_EQ(round->on_configure(lifecycle), nav2_util::CallbackReturn::SUCCESS);
  round->pause();
  FootprintWitness critic;
  critic.on_configure(node, "mppi", "critic", round, handler.get());
  handler->start();  // Reproduce the live controller's dynamic parameter callback.
  const auto result = node->set_parameter(
    rclcpp::Parameter("critic.collision_check_footprint_only", true));
  EXPECT_FALSE(result.successful);
  EXPECT_FALSE(node->get_parameter("critic.collision_check_footprint_only").as_bool());
  // Continue scoring safely in radius mode after the rejected activation.
  EXPECT_NO_THROW(critic.sample(1.0f, 1.0f, 0.0f));
  round->on_cleanup(lifecycle);
}

TEST_F(CollisionCriticTest, EnabledHardVetoCannotBeDynamicallyDisabled)
{
  cornerObstacle();
  FootprintWitness critic;
  configure(critic);
  handler->start();
  const auto result = node->set_parameter(
    rclcpp::Parameter("critic.collision_check_footprint_only", false));
  EXPECT_FALSE(result.successful);
  EXPECT_TRUE(node->get_parameter("critic.collision_check_footprint_only").as_bool());
  EXPECT_EQ(critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f).cost,
    nav2_costmap_2d::LETHAL_OBSTACLE);
}

TEST_F(CollisionCriticTest, StartupOverrideEnablesReadOnlyHardVeto)
{
  cornerObstacle();
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      rclcpp::Parameter("critic.collision_check_footprint_only", true)});
  auto configured_node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
    "startup_override", options);
  mppi::ParametersHandler configured_handler(configured_node);
  FootprintWitness critic;
  critic.on_configure(configured_node, "mppi", "critic", cm, &configured_handler);
  EXPECT_TRUE(configured_node->get_parameter("critic.collision_check_footprint_only").as_bool());
  EXPECT_FALSE(configured_node->set_parameter(
      rclcpp::Parameter("critic.collision_check_footprint_only", false)).successful);
  EXPECT_TRUE(configured_node->describe_parameter(
      "critic.collision_check_footprint_only").read_only);
  EXPECT_EQ(critic.sample(0.5028052926f, 0.4304492474f, -2.0596189499f).cost,
    nav2_costmap_2d::LETHAL_OBSTACLE);
}

TEST_F(CollisionCriticTest, OutOfMapFootprintIsVetoed)
{
  FootprintWitness critic;
  configure(critic);
  EXPECT_EQ(critic.sample(-1.05f, 0.0f, 0.0f).cost, nav2_costmap_2d::LETHAL_OBSTACLE);
}

TEST_F(CollisionCriticTest, InitialLethalFootprintInsideGoalRadiusIsVetoed)
{
  wall();
  score({0.625f, 0.6875f}, true);
}

TEST_F(CollisionCriticTest, EnteringGoalRadiusCannotHideLaterCollision)
{
  wall();
  score({0.25f, 0.4375f, 0.625f}, true);
}

TEST_F(CollisionCriticTest, PostGoalTailIsStillPartOfMotionForecast)
{
  wall();
  score({0.25f, 0.4375f, 0.5f, 0.625f}, true);
}

TEST_F(CollisionCriticTest, AllCollidingRolloutsRequestFallback)
{
  wall();
  score({0.4375f, 0.625f}, true, true);
}

TEST_F(CollisionCriticTest, ClearStoppingForecastAtGoalIsAccepted)
{
  wall();
  score({0.25f, 0.4375f, 0.5f, 0.5f}, false);
}

TEST_F(CollisionCriticTest, ReverseForecastCannotSkipLethalTail)
{
  wall();
  score({1.25f, 1.0f, 0.875f}, true);
}

TEST_F(CollisionCriticTest, PrefilterBitExactOn2cmGridWithFullFootprintSoftScoring)
{
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", true));
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", true));
  node->set_parameter(rclcpp::Parameter("critic.repulsion_weight", 2.0));
  node->set_parameter(rclcpp::Parameter("critic.collision_margin_distance", 0.08));

  cm->getLayeredCostmap()->resizeMap(150, 150, 0.02, -1.5, -1.5);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 150 * 150, 0);

  // Place a wall at x index 110 (x = +0.70 m) with inflation costs >= 220 (within the 0.20 m
  // configured inflation_radius so inflation_radius_ - dist_to_obj > 0 and dist_to_obj < margin).
  for (unsigned int y = 30; y <= 120; ++y) {
    grid->setCost(110, y, nav2_costmap_2d::LETHAL_OBSTACLE);
    for (unsigned int d = 1; d <= 17; ++d) {
      const unsigned char inf_cost = static_cast<unsigned char>(253 - d);
      grid->setCost(110 - d, y, inf_cost);
    }
  }
  grid->setCost(20, 20, nav2_costmap_2d::NO_INFORMATION);

  FootprintWitness critic;
  configure(critic);
  critic.buildScoreCache();

  // Verify LUT bit-equivalence across all valid obstacle costs (1..253).
  for (unsigned int c = 1; c <= 253; ++c) {
    mppi::critics::CollisionCost cc_fp{static_cast<float>(c), true};
    mppi::critics::CollisionCost cc_pt{static_cast<float>(c), false};
    EXPECT_FLOAT_EQ(critic.distanceCached(cc_fp), critic.distanceUncached(cc_fp));
    EXPECT_FLOAT_EQ(critic.distanceCached(cc_pt), critic.distanceUncached(cc_pt));
  }

  // Sweep poses across open space, the 0.36..0.68 m inflation band (where base_link is in
  // FREE_SPACE but the front bumper touches inflation), lethal wall, and map borders.
  const float yaws[] = {0.0f, 0.35f, -0.78f, 1.57f, -2.40f};
  for (int ix = 0; ix < 150; ix += 5) {
    for (int iy = 0; iy < 150; iy += 7) {
      const float wx = -1.49f + 0.02f * static_cast<float>(ix);
      const float wy = -1.49f + 0.02f * static_cast<float>(iy);
      for (const float yaw : yaws) {
        const auto expected = critic.sample(wx, wy, yaw);
        const auto actual = critic.sampleCached(wx, wy, yaw);
        EXPECT_FLOAT_EQ(actual.cost, expected.cost);
        EXPECT_EQ(actual.using_footprint, expected.using_footprint);
      }
    }
  }

  // Specifically verify a pose at x = 0.19 m (0.51 m from the lethal wall at x = 0.70 m):
  // base_link cell is in FREE_SPACE (0), but front bumper (+0.182 m -> x = 0.372 m, cell 93)
  // reaches into the inflation ramp (cells 93..109) and must NOT be skipped as zero cost.
  const auto band_uncached = critic.sample(0.19f, 0.0f, 0.0f);
  const auto band_cached = critic.sampleCached(0.19f, 0.0f, 0.0f);
  EXPECT_GT(band_uncached.cost, 0.0f);
  EXPECT_FLOAT_EQ(band_cached.cost, band_uncached.cost);
  EXPECT_TRUE(band_cached.using_footprint);

  // Verify full score() bit-equivalence against trajectories in open space, inflation band,
  // and lethal collision.
  mppi::models::State state;
  state.reset(3, 4);
  mppi::models::Trajectories trajectories;
  trajectories.reset(3, 4);
  trajectories.x.fill(0.0f);
  trajectories.y.fill(0.0f);
  trajectories.yaws.fill(0.0f);
  for (size_t j = 0; j < 4; ++j) {
    // Trajectory 0: open space (far from wall)
    trajectories.x(0, j) = -0.50f;
    // Trajectory 1: base_link in FREE_SPACE, front bumper inside inflation ramp
    trajectories.x(1, j) = 0.19f;
    // Trajectory 2: colliding with lethal wall
    trajectories.x(2, j) = 0.60f;
  }
  mppi::models::Path path;
  path.reset(3);
  geometry_msgs::msg::Pose goal;
  goal.position.x = 2.0;
  goal.orientation.w = 1.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({3});
  float dt = 0.05f;
  mppi::CriticData data{state, trajectories, path, goal, costs, dt,
    false, nullptr, nullptr, std::nullopt, std::nullopt};
  critic.score(data);

  // Compute exact uncached reference score for Trajectory 1 (4 steps at x = 0.19, y = 0, yaw = 0).
  const float inf_r = 0.20f;
  const float margin = 0.08f;
  const float dist_1 = critic.distanceUncached(band_uncached);
  float raw_1 = 0.0f;
  float rep_1 = 0.0f;
  for (size_t j = 0; j < 4; ++j) {
    if (dist_1 < margin) {
      raw_1 += (margin - dist_1);
    }
    rep_1 += inf_r - dist_1;
  }
  const float expected_cost_1 = (20.0f * raw_1) + (2.0f * (rep_1 / 4.0f));

  EXPECT_FLOAT_EQ(costs(0), 0.0f);
  EXPECT_FLOAT_EQ(costs(1), expected_cost_1);
  EXPECT_GT(costs(1), 0.0f);
  EXPECT_GE(costs(2), 1000000.0f);
  EXPECT_FALSE(data.fail_flag);
}

TEST_F(CollisionCriticTest, PrefilterBitExactOn2cmGridInHardOnlyMode)
{
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", false));
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", true));

  cm->getLayeredCostmap()->resizeMap(150, 150, 0.02, -1.5, -1.5);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 150 * 150, 0);
  grid->setCost(75, 75, 80);
  grid->setCost(95, 75, 253);
  grid->setCost(110, 80, nav2_costmap_2d::LETHAL_OBSTACLE);
  grid->setCost(40, 40, nav2_costmap_2d::NO_INFORMATION);

  FootprintWitness critic;
  configure(critic);
  critic.buildScoreCache();

  // Verify LUT bit-equivalence across all valid obstacle costs (1..253).
  for (unsigned int c = 1; c <= 253; ++c) {
    mppi::critics::CollisionCost cc_fp{static_cast<float>(c), true};
    mppi::critics::CollisionCost cc_pt{static_cast<float>(c), false};
    EXPECT_FLOAT_EQ(critic.distanceCached(cc_fp), critic.distanceUncached(cc_fp));
    EXPECT_FLOAT_EQ(critic.distanceCached(cc_pt), critic.distanceUncached(cc_pt));
  }

  const float yaws[] = {0.0f, 0.6f, -1.2f, 2.1f};
  for (int ix = 0; ix < 150; ix += 5) {
    for (int iy = 0; iy < 150; iy += 5) {
      const float wx = -1.49f + 0.02f * static_cast<float>(ix);
      const float wy = -1.49f + 0.02f * static_cast<float>(iy);
      for (const float yaw : yaws) {
        const auto expected = critic.sample(wx, wy, yaw);
        const auto actual = critic.sampleCached(wx, wy, yaw);
        EXPECT_FLOAT_EQ(actual.cost, expected.cost);
        EXPECT_EQ(actual.using_footprint, expected.using_footprint);
      }
    }
  }
}

TEST_F(CollisionCriticTest, PrefilterHandlesCostmapShrinkAndInPlaceMutation)
{
  node->set_parameter(rclcpp::Parameter("critic.consider_footprint", true));
  node->set_parameter(rclcpp::Parameter("critic.collision_check_footprint_only", true));

  // Reuse a SINGLE critic instance across all resizes and in-place costmap mutations.
  FootprintWitness critic;
  configure(critic);

  auto scoreWithReusedCritic = [&](const std::vector<float> & xs, bool expect_collision) {
      mppi::models::State state;
      state.reset(1, xs.size());
      mppi::models::Trajectories trajectories;
      trajectories.reset(1, xs.size());
      trajectories.y.fill(0.0f);
      trajectories.yaws.fill(0.0f);
      for (size_t j = 0; j < xs.size(); ++j) {
        trajectories.x(0, j) = xs[j];
      }
      mppi::models::Path path;
      path.reset(3);
      geometry_msgs::msg::Pose goal;
      goal.position.x = 0.5;
      goal.orientation.w = 1.0;
      xt::xtensor<float, 1> costs = xt::zeros<float>({1});
      float dt = 0.05f;
      mppi::CriticData data{state, trajectories, path, goal, costs, dt,
        false, nullptr, nullptr, std::nullopt, std::nullopt};
      critic.score(data);
      if (expect_collision) {
        EXPECT_GE(costs(0), 1000000.0f);
        EXPECT_TRUE(data.fail_flag);
      } else {
        EXPECT_FLOAT_EQ(costs(0), 0.0f);
        EXPECT_FALSE(data.fail_flag);
      }
    };

  // 1. Start on a 150x150 2 cm grid (22,500 cells) and score in open space.
  cm->getLayeredCostmap()->resizeMap(150, 150, 0.02, -1.5, -1.5);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 150 * 150, 0);
  scoreWithReusedCritic({0.0f, 0.05f}, false);

  // 2. Shrink in place to a 20x20 2 cm grid (smaller than 2 * prefilter_radius_) on the
  // SAME critic instance and verify no unsigned underflow or buffer overrun occurs.
  cm->getLayeredCostmap()->resizeMap(20, 20, 0.02, -0.2, -0.2);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 20 * 20, 0);
  scoreWithReusedCritic({0.0f}, false);

  // 3. Resize to 60x60 5 cm grid, score in open space (populating max_cost_in_radius_ with 0),
  // then mutate a PERIMETER cell (front bumper at x = +0.182 m, y = 0.0 m) in place to lethal
  // while the center reference cell at (0.0, 0.0) stays FREE_SPACE (0). If max_cost_in_radius_
  // were not recomputed on the next score() call, the prefilter would falsely skip as FREE_SPACE.
  cm->getLayeredCostmap()->resizeMap(60, 60, 0.05, -1.5, -1.5);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 60 * 60, 0);
  scoreWithReusedCritic({0.0f, 0.05f}, false);

  unsigned int cx = 0, cy = 0, px = 0, py = 0;
  ASSERT_TRUE(cm->getCostmap()->worldToMap(0.0, 0.0, cx, cy));
  ASSERT_TRUE(cm->getCostmap()->worldToMap(0.182, 0.0, px, py));
  ASSERT_NE(cx, px);
  cm->getCostmap()->setCost(px, py, nav2_costmap_2d::LETHAL_OBSTACLE);
  ASSERT_EQ(cm->getCostmap()->getCost(cx, cy), nav2_costmap_2d::FREE_SPACE);
  scoreWithReusedCritic({0.0f, 0.05f}, true);
}
}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
