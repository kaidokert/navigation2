// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_mppi_controller/optimizer.hpp"
#include "nav2_mppi_controller/tools/path_handler.hpp"
#include "utils/controller_scenario.hpp"

namespace
{
using controller_test::Atlas;
using controller_test::Pose;
using controller_test::Scenario;

// Access protected test state without casting a base object to a derived type.
struct NoiseAccess : mppi::NoiseGenerator
{
  static auto vx() {return &NoiseAccess::noises_vx_;}
  static auto wz() {return &NoiseAccess::noises_wz_;}
};
struct CriticAccess : mppi::CriticManager
{
  static auto critics() {return &CriticAccess::critics_;}
  static auto names() {return &CriticAccess::critic_names_;}
};

class ScenarioOptimizer : public mppi::Optimizer
{
public:
  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr node,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm, mppi::ParametersHandler * params)
  {
    initialize(node, "FollowPath", cm, params);
    // Bind critics compiled from this checkout, including in the standalone CI
    // target. Never load a different installed plugin via pluginlib for this proof.
    auto & critics = critic_manager_.*CriticAccess::critics();
    critics.emplace_back(std::make_unique<mppi::critics::PathFollowCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::PathAlignCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::ObstaclesCritic>());
    const std::vector<std::string> names = {
      "PathFollowCritic", "PathAlignCritic", "ObstaclesCritic"};
    critic_manager_.*CriticAccess::names() = names;
    for (size_t i = 0; i < critics.size(); ++i) {
      critics[i]->on_configure(node, "FollowPath", "FollowPath." + names[i], cm, params);
    }
    // Frozen, balanced bank; no RNG/thread dependence and no production seed API.
    auto & vx = noise_generator_.*NoiseAccess::vx();
    auto & wz = noise_generator_.*NoiseAccess::wz();
    for (size_t i = 0; i < vx.shape(0); ++i) {
      for (size_t j = 0; j < vx.shape(1); ++j) {
        const float sign = i % 2 == 0 ? 1.0f : -1.0f;
        vx(i, j) = sign * 0.033f * std::sin(0.7f * (i / 2 + j + 1));
        wz(i, j) = sign * 0.101f * std::cos(0.9f * (i / 2 + j + 1));
      }
    }
  }
  const auto & tape() const {return control_sequence_;}
  const auto & history() const {return control_history_;}
  void eraseTape()
  {
    control_sequence_.vx.fill(0.0f);
    control_sequence_.vy.fill(0.0f);
    control_sequence_.wz.fill(0.0f);
  }
  void eraseHistory()
  {
    control_history_.fill({0.0f, 0.0f, 0.0f});
  }
  void eraseMemory()
  {
    eraseTape();
    eraseHistory();
  }
  void copyMemoryFrom(const ScenarioOptimizer & other)
  {
    control_sequence_ = other.control_sequence_;
    control_history_ = other.control_history_;
  }
};

class ScenarioPathHandler : public mppi::PathHandler
{
public:
  bool reverse() const {return current_segment_is_reverse_;}
  const auto & activeEndpoint() const {return global_plan_up_to_inversion_.poses.back().pose;}
  size_t activeSize() const {return global_plan_up_to_inversion_.poses.size();}
  unsigned int inversion() const {return inversion_locale_;}
};

class ControllerScenarios : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    atlas = std::make_unique<Atlas>(CONTROLLER_SCENARIO_FILE);
    cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "scenario_map", "", "scenario_map", true);
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    cm->set_parameter(rclcpp::Parameter("global_frame", "map"));
    cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.0));
    std::string footprint_text = "[";
    for (const auto & p : atlas->document["footprint"]) {
      const auto xy = controller_test::numbers<2>(p);
      if (footprint_text.size() > 1) {footprint_text += ",";}
      footprint_text += "[" + std::to_string(xy[0]) + "," + std::to_string(xy[1]) + "]";
    }
    cm->set_parameter(rclcpp::Parameter("footprint", footprint_text + "]"));
    ASSERT_EQ(cm->on_configure(rclcpp_lifecycle::State{}), nav2_util::CallbackReturn::SUCCESS);
    cm->pause();
    std::vector<geometry_msgs::msg::Point> footprint;
    for (const auto & p : atlas->document["footprint"]) {
      geometry_msgs::msg::Point point;
      point.x = controller_test::number(p[0]);
      point.y = controller_test::number(p[1]);
      footprint.push_back(point);
    }
    cm->setRobotFootprint(footprint);
  }
  static void TearDownTestSuite()
  {
    cm->on_cleanup(rclcpp_lifecycle::State{});
    cm.reset();
    atlas.reset();
  }
  void grid(const Scenario & scenario)
  {
    const auto map = atlas->document["map"];
    cm->getLayeredCostmap()->resizeMap(
      map["size"][0].as<unsigned int>(), map["size"][1].as<unsigned int>(),
      controller_test::number(map["resolution"]), controller_test::number(map["origin"][0]),
      controller_test::number(map["origin"][1]));
    auto * map_ptr = cm->getCostmap();
    for (unsigned int y = 0; y < map_ptr->getSizeInCellsY(); ++y) {
      for (unsigned int x = 0; x < map_ptr->getSizeInCellsX(); ++x) {
        double wx, wy;
        map_ptr->mapToWorld(x, y, wx, wy);
        const double half = map_ptr->getResolution() / 2.0;
        unsigned char cost = nav2_costmap_2d::FREE_SPACE;
        for (const auto & b : scenario.obstacles) {
          if (wx + half > b[0] && wx - half<b[2] &&
            wy + half> b[1] && wy - half < b[3])
          {
            cost = nav2_costmap_2d::LETHAL_OBSTACLE;
          }
        }
        map_ptr->setCost(x, y, cost);
      }
    }
  }
  rclcpp_lifecycle::LifecycleNode::SharedPtr node(bool headings)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("controller_frequency", 20.0),
        rclcpp::Parameter("FollowPath.motion_model", "Ackermann"),
        rclcpp::Parameter("FollowPath.critics", std::vector<std::string>{}),
        rclcpp::Parameter("FollowPath.model_dt", 0.05),
        rclcpp::Parameter("FollowPath.batch_size", 64),
        rclcpp::Parameter("FollowPath.time_steps", 41),
        rclcpp::Parameter("FollowPath.iteration_count", 3),
        rclcpp::Parameter("FollowPath.regenerate_noises", false),
        rclcpp::Parameter("FollowPath.retry_attempt_limit", 0),
        rclcpp::Parameter("FollowPath.vx_std", 0.033),
        rclcpp::Parameter("FollowPath.wz_std", 0.101),
        rclcpp::Parameter("FollowPath.AckermannConstraints.min_turning_r", 0.558),
        rclcpp::Parameter("FollowPath.PathAlignCritic.threshold_to_consider", 0.01),
        rclcpp::Parameter("FollowPath.PathAlignCritic.offset_from_furthest", 1),
        rclcpp::Parameter("FollowPath.PathAlignCritic.cost_weight", 28.0),
        rclcpp::Parameter("FollowPath.PathAlignCritic.use_path_orientations", headings),
        rclcpp::Parameter("FollowPath.PathFollowCritic.threshold_to_consider", 0.01),
        rclcpp::Parameter("FollowPath.PathFollowCritic.cost_weight", 66.05),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.consider_footprint", true),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.collision_check_footprint_only", true),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.collision_cost", 1e9),
        rclcpp::Parameter("FollowPath.enforce_path_inversion", true),
        rclcpp::Parameter("FollowPath.terminal_stub_max_length", 0.0)});
    return std::make_shared<rclcpp_lifecycle::LifecycleNode>("scenario_controller", options);
  }
  geometry_msgs::msg::PoseStamped pose(const Pose & p)
  {
    Scenario one;
    one.path = {p};
    return one.message().poses.front();
  }
  inline static std::unique_ptr<Atlas> atlas;
  inline static std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
};

TEST_F(ControllerScenarios, GeometryGearAndMirrorContracts)
{
  ASSERT_EQ(atlas->cases.size(), 15u);
  for (const auto & scenario : atlas->cases) {
    SCOPED_TRACE(scenario.id);
    for (bool reverse : {false, true}) {
      const auto variant = scenario.variant(true, reverse);
      const auto restored = variant.variant(true, reverse);
      ASSERT_EQ(restored.path.size(), scenario.path.size());
      EXPECT_EQ(restored.gears, scenario.gears);
      for (size_t i = 0; i < scenario.path.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored.path[i][0], scenario.path[i][0]);
        EXPECT_DOUBLE_EQ(restored.path[i][1], scenario.path[i][1]);
        EXPECT_NEAR(controller_test::wrap(restored.path[i][2] - scenario.path[i][2]), 0, 1e-12);
      }
    }
  }
  const auto & full = atlas->at("m8_full");
  auto full_path = full.message();
  EXPECT_EQ(mppi::utils::findFirstPathInversion(full_path), 9u);  // First pose after cusp.
  EXPECT_NEAR(full.path.back()[0], 0.0, 1e-8);
  EXPECT_NEAR(full.path.back()[1], 0.4, 1e-8);
  EXPECT_THROW(controller_test::numbers<3>(YAML::Load("[0, 0, .nan]")), std::invalid_argument);
  EXPECT_THROW(controller_test::numbers<3>(YAML::Load("[0, 0]")), std::invalid_argument);
}

TEST_F(ControllerScenarios, RealPathAlignScoresAllStartsAndRepresentations)
{
  for (const auto & original : atlas->cases) {
    if (original.id == "m8_full") {continue;}  // PathHandler segments this in the test below.
    for (bool reverse : {false, true}) {
      const auto scenario = original.variant(false, reverse);
      grid(scenario);
      for (const auto & start : scenario.starts) {
        SCOPED_TRACE(scenario.id + "/" + start.first + (reverse ? "/reverse" : "/forward"));
        std::vector<float> position_costs;
        for (bool headings : {false, true}) {
          auto parent = node(headings);
          mppi::ParametersHandler params(parent);
          mppi::critics::PathAlignCritic critic;
          critic.on_configure(parent, "FollowPath", "FollowPath.PathAlignCritic", cm, &params);
          mppi::models::State state;
          state.reset(3, 41);
          state.pose = pose(start.second);
          mppi::models::Trajectories trajectories;
          trajectories.reset(3, 41);
          // Three physically feasible constant-curvature witnesses. Their scores
          // characterize the objective; no claim that a single tick predicts success.
          for (size_t i = 0; i < 3; ++i) {
            double x = start.second[0], y = start.second[1], yaw = start.second[2];
            const double v = scenario.gears.front() * 0.1;
            const double wz = (static_cast<int>(i) - 1) * 0.15;
            for (size_t j = 0; j < 41; ++j) {
              trajectories.x(i, j) = x;
              trajectories.y(i, j) = y;
              trajectories.yaws(i, j) = yaw;
              x += v * std::cos(yaw) * 0.05;
              y += v * std::sin(yaw) * 0.05;
              yaw += wz * 0.05;
            }
          }
          auto local_path = scenario.message();
          if (scenario.id.rfind("pruning_", 0) == 0) {
            // Match the production pipeline: PathAlign sees the local reference
            // after PathHandler has removed vertices behind this measured start.
            auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
            ScenarioPathHandler handler;
            handler.initialize(parent, "FollowPath", cm, buffer, &params);
            handler.setPath(local_path);
            local_path = handler.transformPath(state.pose);
            const auto closest = std::min_element(scenario.path.begin(), scenario.path.end(),
                [&start](const Pose & a, const Pose & b) {
                  return std::hypot(a[0] - start.second[0], a[1] - start.second[1]) <
                         std::hypot(b[0] - start.second[0], b[1] - start.second[1]);
              });
            ASSERT_NE(closest, scenario.path.begin());
            ASSERT_GE(local_path.poses.size(), 2u);
            EXPECT_NEAR(local_path.poses.front().pose.position.x, (*closest)[0], 1e-8);
            EXPECT_NEAR(local_path.poses.front().pose.position.y, (*closest)[1], 1e-8);
            EXPECT_LT(local_path.poses.size(), scenario.path.size());
          }
          auto path = mppi::utils::toTensor(local_path);
          auto goal = scenario.message().poses.back().pose;
          xt::xtensor<float, 1> costs = xt::zeros<float>({3});
          float dt = 0.05;
          mppi::CriticData data{state, trajectories, path, goal, costs, dt, false,
            nullptr, nullptr, std::nullopt, std::nullopt};
          // Production code computes the caches. These witnesses must reach the
          // scoring branch; a silent all-zero early return is not a passing test.
          critic.score(data);
          ASSERT_TRUE(data.furthest_reached_path_point.has_value());
          EXPECT_GT(*data.furthest_reached_path_point, 0u);
          for (size_t i = 0; i < 3; ++i) {
            EXPECT_GT(costs(i), 0.0f);
            EXPECT_LT(costs(i), 100.0f);
            if (!headings) {position_costs.push_back(costs(i));} else {
              EXPECT_GE(costs(i) + 1e-5f, position_costs[i]);
            }
          }
        }
      }
    }
  }
}

TEST_F(ControllerScenarios, PathAlignHasIndependentPositionAndHeadingSensitivity)
{
  for (bool mirror : {false, true}) {
    for (bool reverse : {false, true}) {
      const auto scenario = atlas->at("curved_capture").variant(mirror, reverse);
      grid(scenario);
      for (bool headings : {false, true}) {
        auto parent = node(headings);
        mppi::ParametersHandler params(parent);
        mppi::critics::PathAlignCritic critic;
        critic.on_configure(parent, "FollowPath", "FollowPath.PathAlignCritic", cm, &params);
        const auto count = scenario.path.size();
        mppi::models::State state;
        state.reset(3, count);
        state.pose = pose(scenario.path.front());
        mppi::models::Trajectories trajectories;
        trajectories.reset(3, count);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < count; ++j) {
            trajectories.x(i, j) = scenario.path[j][0];
            trajectories.y(i, j) = scenario.path[j][1] + (i == 1 ? 0.07 : 0.0);
            trajectories.yaws(i, j) = scenario.path[j][2] + (i == 2 ? 0.2 : 0.0);
          }
        }
        auto path = mppi::utils::toTensor(scenario.message());
        auto goal = scenario.message().poses.back().pose;
        xt::xtensor<float, 1> costs = xt::zeros<float>({3});
        float dt = 0.05;
        mppi::CriticData data{state, trajectories, path, goal, costs, dt, false,
          nullptr, nullptr, std::nullopt, std::nullopt};
        critic.score(data);
        // Strided arc-length sampling has a small chord approximation error.
        EXPECT_LT(costs(0), 0.1f);
        EXPECT_GT(costs(1), costs(0) + 1.0f);
        if (headings) {EXPECT_GT(costs(2), costs(0) + 4.0f);} else {
          EXPECT_NEAR(costs(2), costs(0), 1e-5f);
        }
      }
    }
  }
}

TEST_F(ControllerScenarios, WallFixtureExercisesHardCollisionScoring)
{
  const auto scenario = atlas->at("wall_capture");
  grid(scenario);
  auto parent = node(true);
  mppi::ParametersHandler params(parent);
  mppi::critics::ObstaclesCritic critic;
  critic.on_configure(parent, "FollowPath", "FollowPath.ObstaclesCritic", cm, &params);
  const auto count = scenario.path.size();
  mppi::models::State state;
  state.reset(2, count);
  state.pose = pose(scenario.path.front());
  mppi::models::Trajectories trajectories;
  trajectories.reset(2, count);
  for (size_t j = 0; j < count; ++j) {
    for (size_t i = 0; i < 2; ++i) {
      trajectories.x(i, j) = scenario.path[j][0];
      trajectories.y(i, j) = scenario.path[j][1] + i * 0.1;
      trajectories.yaws(i, j) = scenario.path[j][2];
      // Both reference points are free at EVERY sample. Only the displaced
      // body's perimeter intersects the wall; point-only collision must fail.
      unsigned int mx, my;
      ASSERT_TRUE(cm->getCostmap()->worldToMap(
          trajectories.x(i, j), trajectories.y(i, j), mx, my));
      ASSERT_EQ(cm->getCostmap()->getCost(mx, my), nav2_costmap_2d::FREE_SPACE);
    }
  }
  auto path = mppi::utils::toTensor(scenario.message());
  auto goal = scenario.message().poses.back().pose;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float dt = 0.05;
  mppi::CriticData data{state, trajectories, path, goal, costs, dt, false,
    nullptr, nullptr, std::nullopt, std::nullopt};
  critic.score(data);
  EXPECT_EQ(costs(0), 0.0f);
  EXPECT_GE(costs(1), 1e9f);
  EXPECT_FALSE(data.fail_flag);
}

TEST_F(ControllerScenarios, SequentialOptimizerPreservesMemoryAndIsRepeatable)
{
  const auto & scenario = atlas->at("curved_capture");
  grid(scenario);
  std::vector<std::array<double, 2>> reference;
  bool memory_changes_command = false;
  for (int repetition = 0; repetition < 3; ++repetition) {
    auto parent = node(true);
    mppi::ParametersHandler params(parent);
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params);
    auto path = scenario.message();
    auto actual = pose(scenario.starts.at("inside"));
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.1;
    for (size_t tick = 0; tick < 12; ++tick) {
      if (repetition == 2) {optimizer.eraseMemory();}
      // Exogenous poses stay fixed deliberately. This is a sequential decision
      // fixture, not a simulator or a claim about a closed-loop endpoint.
      const auto command = optimizer.evalControl(actual, speed, path, path.poses.back().pose,
          nullptr);
      const double vx = command.twist.linear.x, wz = command.twist.angular.z;
      EXPECT_LE(std::abs(wz), std::abs(vx) / 0.558 + 1e-5);
      EXPECT_NEAR(optimizer.history().back().vx, vx, 1e-6);
      EXPECT_NEAR(optimizer.history().back().wz, wz, 1e-6);
      if (repetition == 0) {reference.push_back({vx, wz});} else if (repetition == 1) {
        EXPECT_NEAR(vx, reference[tick][0], 1e-6);
        EXPECT_NEAR(wz, reference[tick][1], 1e-6);
      } else {
        EXPECT_NEAR(vx, reference.front()[0], 1e-6);
        EXPECT_NEAR(wz, reference.front()[1], 1e-6);
        memory_changes_command |= std::abs(vx - reference[tick][0]) > 1e-5 ||
          std::abs(wz - reference[tick][1]) > 1e-5;
      }
    }
    EXPECT_GT(xt::sum(xt::abs(optimizer.tape().vx))(), 0.001f);
  }
  EXPECT_TRUE(memory_changes_command);
}

TEST_F(ControllerScenarios, CompleteM8UsesProductionPathHandlerWithoutResetAtCusps)
{
  const auto & scenario = atlas->at("m8_full");
  grid(scenario);
  auto parent = node(true);
  mppi::ParametersHandler params(parent);
  auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
  ScenarioPathHandler handler;
  handler.initialize(parent, "FollowPath", cm, buffer, &params);
  handler.setPath(scenario.message());
  ScenarioOptimizer optimizer;
  optimizer.configure(parent, cm, &params);
  ScenarioOptimizer erased_at_cusps;
  erased_at_cusps.configure(parent, cm, &params);
  const auto goal = scenario.message().poses.back().pose;
  std::vector<size_t> handoffs;
  auto expected_history = optimizer.history();
  size_t commands = 0;
  // Prescribed poses exercise the actual segment handoff and carried optimizer
  // state. Default 1.5 cm along-track tolerance accepts the 1.27/1.22 cm
  // pre-cusp edges at indices 7 and 24; these are not exact-cusp pose arrivals.
  // FollowPath in Gazebo supplies the separate physical tracking test.
  for (size_t i = 0; i + 1 < scenario.path.size(); ++i) {
    SCOPED_TRACE(i);
    const auto actual = pose(scenario.path[i]);
    const bool was_reverse = handler.reverse();
    auto segment = handler.transformPath(actual);
    ASSERT_GE(segment.poses.size(), 2u);
    EXPECT_EQ(handler.reverse(), i >= 7 && i < 24);
    const size_t endpoint = i < 7 ? 8 : (i < 24 ? 25 : scenario.path.size() - 1);
    EXPECT_NEAR(handler.activeEndpoint().position.x, scenario.path[endpoint][0], 1e-8);
    EXPECT_NEAR(handler.activeEndpoint().position.y, scenario.path[endpoint][1], 1e-8);
    const bool handoff = handler.reverse() != was_reverse;
    if (handoff) {
      handoffs.push_back(i);
      // Check the returned optimizer reference, not just the handler's internals:
      // stale pre-handoff output must not survive a correct internal transition.
      EXPECT_NEAR(segment.poses.back().pose.position.x, scenario.path[endpoint][0], 1e-8);
      EXPECT_NEAR(segment.poses.back().pose.position.y, scenario.path[endpoint][1], 1e-8);
      const auto & a = segment.poses[0].pose;
      const auto & b = segment.poses[1].pose;
      const double yaw = tf2::getYaw(a.orientation);
      const double along_body = (b.position.x - a.position.x) * std::cos(yaw) +
        (b.position.y - a.position.y) * std::sin(yaw);
      EXPECT_GT(along_body * (handler.reverse() ? -1 : 1), 0.0);
      EXPECT_GT(xt::sum(xt::abs(optimizer.tape().vx) + xt::abs(optimizer.tape().wz))(),
        0.001f);
      // Identical SG history isolates tape carry. History carry is independently
      // asserted below, so either kind of unintended reset must fail this test.
      erased_at_cusps.copyMemoryFrom(optimizer);
      erased_at_cusps.eraseTape();
    }
    geometry_msgs::msg::Twist speed;
    // This prescribed-state probe follows the active segment, including the
    // tolerance-accepted early switches, instead of injecting the previous gear.
    speed.linear.x = handler.reverse() ? -0.1 : 0.1;
    const auto command = optimizer.evalControl(actual, speed, segment, goal, nullptr);
    for (size_t h = 0; h + 1 < expected_history.size(); ++h) {
      EXPECT_FLOAT_EQ(optimizer.history()[h].vx, expected_history[h + 1].vx);
      EXPECT_FLOAT_EQ(optimizer.history()[h].vy, expected_history[h + 1].vy);
      EXPECT_FLOAT_EQ(optimizer.history()[h].wz, expected_history[h + 1].wz);
    }
    EXPECT_NEAR(optimizer.history().back().vx, command.twist.linear.x, 1e-6);
    EXPECT_NEAR(optimizer.history().back().wz, command.twist.angular.z, 1e-6);
    if (handoff) {
      erased_at_cusps.evalControl(actual, speed, segment, goal, nullptr);
      const double tape_delta = xt::sum(
        xt::abs(optimizer.tape().vx - erased_at_cusps.tape().vx) +
        xt::abs(optimizer.tape().wz - erased_at_cusps.tape().wz))();
      EXPECT_GT(tape_delta, 1e-5);
    }
    expected_history = optimizer.history();
    ++commands;
  }
  EXPECT_EQ(commands, scenario.path.size() - 1);
  EXPECT_EQ(handoffs, (std::vector<size_t>{7, 24}));
  EXPECT_FALSE(handler.reverse());
  EXPECT_EQ(handler.inversion(), 0u);
  EXPECT_EQ(handler.getPath().poses.size(), scenario.path.size() - 26);
  EXPECT_EQ(handler.activeSize(), 2u);
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
