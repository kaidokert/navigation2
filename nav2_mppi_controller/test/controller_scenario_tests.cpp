// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <memory>
#include <random>
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
  void setSeededGaussianBank(uint32_t seed = 42u)
  {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dvx(0.0f, 0.033f);
    std::normal_distribution<float> dwz(0.0f, 0.101f);
    auto & vx = noise_generator_.*NoiseAccess::vx();
    auto & wz = noise_generator_.*NoiseAccess::wz();
    for (size_t i = 0; i < vx.shape(0); i += 2) {
      for (size_t j = 0; j < vx.shape(1); ++j) {
        const float nvx = dvx(gen);
        const float nwz = dwz(gen);
        vx(i, j) = nvx;
        wz(i, j) = nwz;
        if (i + 1 < vx.shape(0)) {
          vx(i + 1, j) = -nvx;
          wz(i + 1, j) = -nwz;
        }
      }
    }
  }
};

class ScenarioPathHandler : public mppi::PathHandler
{
public:
  bool reverse() const {return current_segment_is_reverse_;}
  const auto & activeEndpoint() const {return global_plan_up_to_inversion_.poses.back().pose;}
  size_t activeSize() const {return global_plan_up_to_inversion_.poses.size();}
  unsigned int inversion() const {return inversion_locale_;}
  bool latchedCapture() const {return has_latched_terminal_capture_;}
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
  rclcpp_lifecycle::LifecycleNode::SharedPtr node(
    bool headings, int batch_size = 64, double min_turning_r = 0.558, int pfc_offset = 6,
    bool terminal_capture = false, double replan_drift = 0.020,
    double max_residual_y = 0.040, double pullthrough_length = 0.10,
    double pullthrough_release = 0.015)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("controller_frequency", 20.0),
        rclcpp::Parameter("FollowPath.motion_model", "Ackermann"),
        rclcpp::Parameter("FollowPath.critics", std::vector<std::string>{}),
        rclcpp::Parameter("FollowPath.model_dt", 0.05),
        rclcpp::Parameter("FollowPath.batch_size", batch_size),
        rclcpp::Parameter("FollowPath.time_steps", 41),
        rclcpp::Parameter("FollowPath.iteration_count", 3),
        rclcpp::Parameter("FollowPath.regenerate_noises", false),
        rclcpp::Parameter("FollowPath.retry_attempt_limit", 0),
        rclcpp::Parameter("FollowPath.vx_std", 0.033),
        rclcpp::Parameter("FollowPath.wz_std", 0.101),
        rclcpp::Parameter("FollowPath.AckermannConstraints.min_turning_r", min_turning_r),
        rclcpp::Parameter("FollowPath.PathAlignCritic.threshold_to_consider", 0.01),
        rclcpp::Parameter("FollowPath.PathAlignCritic.offset_from_furthest", 1),
        rclcpp::Parameter("FollowPath.PathAlignCritic.cost_weight", 28.0),
        rclcpp::Parameter("FollowPath.PathAlignCritic.use_path_orientations", headings),
        rclcpp::Parameter("FollowPath.PathFollowCritic.threshold_to_consider", 0.01),
        rclcpp::Parameter("FollowPath.PathFollowCritic.offset_from_furthest", pfc_offset),
        rclcpp::Parameter("FollowPath.PathFollowCritic.cost_weight", 66.05),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.consider_footprint", true),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.collision_check_footprint_only", true),
        rclcpp::Parameter("FollowPath.ObstaclesCritic.collision_cost", 1e9),
        rclcpp::Parameter("FollowPath.enforce_path_inversion", true),
        rclcpp::Parameter("FollowPath.terminal_stub_max_length", 0.0),
        rclcpp::Parameter("FollowPath.terminal_capture_enabled", terminal_capture),
        rclcpp::Parameter("FollowPath.terminal_capture_replan_drift", replan_drift),
        rclcpp::Parameter("FollowPath.terminal_capture_max_residual_y", max_residual_y),
        rclcpp::Parameter("FollowPath.terminal_capture_pullthrough_length", pullthrough_length),
        rclcpp::Parameter("FollowPath.terminal_capture_pullthrough_release", pullthrough_release)});
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
  ASSERT_EQ(atlas->cases.size(), 16u);
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

struct ClosedLoopOutcome
{
  double x_lon{0.0};
  double e_right{0.0};
  double dth_rad{0.0};
  double xy_err{0.0};
  double final_vx{0.0};
  bool collision_free{true};
  bool latched_capture{false};
  size_t ticks{0u};
};

struct CaptureCandidateConfig
{
  std::string name{"C3"};
  bool enable_capture{true};
  double replan_drift{0.020};
  double max_residual_y{0.040};
  double pullthrough_length{0.10};
  double pullthrough_release{0.015};
  bool use_stopped_goal_checker{true};
  std::optional<std::pair<size_t, std::array<double, 3>>> inject_deflection{std::nullopt};
};

TEST_F(ControllerScenarios, ClosedLoopTerminalApproachBaselineAndCapture)
{
  auto run_scenario_closed_loop_cfg = [&](
    const Scenario & scenario, const std::string & label,
    const std::string & start_key, const CaptureCandidateConfig & cfg)
    {
      grid(scenario);
      auto parent = node(
        true, 500, 0.325, 7, cfg.enable_capture,
        cfg.replan_drift, cfg.max_residual_y, cfg.pullthrough_length, cfg.pullthrough_release);
      mppi::ParametersHandler params(parent);
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(scenario.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_pose = scenario.path.back();
      const auto goal_msg = scenario.message().poses.back().pose;
      auto actual = pose(scenario.starts.at(start_key));
      geometry_msgs::msg::Twist speed;
      speed.linear.x = scenario.gears.front() * 0.15;
      speed.angular.z = 0.0;

      const auto footprint = cm->getRobotFootprint();
      nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(
        cm->getCostmap());

      auto to_target_frame = [&](const geometry_msgs::msg::PoseStamped & p) {
          const double dx = p.pose.position.x - goal_pose[0];
          const double dy = p.pose.position.y - goal_pose[1];
          const double c = std::cos(-goal_pose[2]);
          const double s = std::sin(-goal_pose[2]);
          const double x_lon = c * dx - s * dy;
          const double y_lat = s * dx + c * dy;
          const double yaw = tf2::getYaw(p.pose.orientation);
          const double dth = controller_test::wrap(yaw - goal_pose[2]);
          return std::array<double, 3>{x_lon, -y_lat, dth};
        };

      ClosedLoopOutcome best;
      auto init_tf = to_target_frame(actual);
      best.x_lon = init_tf[0];
      best.e_right = init_tf[1];
      best.dth_rad = init_tf[2];
      best.xy_err = std::hypot(init_tf[0], init_tf[1]);

      const double stopped_vx_thresh = cfg.use_stopped_goal_checker ? 0.05 : 0.002;
      const double stopped_wz_thresh = cfg.use_stopped_goal_checker ? 0.25 : 0.02;

      for (size_t tick = 0; tick < 180; ++tick) {
        if (cfg.inject_deflection.has_value() && tick == cfg.inject_deflection->first) {
          actual = pose(cfg.inject_deflection->second);
        }
        const double cur_yaw = tf2::getYaw(actual.pose.orientation);
        const double fp_cost = checker.footprintCostAtPose(
          actual.pose.position.x, actual.pose.position.y, cur_yaw, footprint);
        if (fp_cost < 0.0 || fp_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
          best.collision_free = false;
        }

        auto segment = handler.transformPath(actual);
        best.latched_capture = handler.latchedCapture();
        const auto cmd = optimizer.evalControl(actual, speed, segment, goal_msg, nullptr);
        double vx = cmd.twist.linear.x;
        double wz = cmd.twist.angular.z;
        const double max_wz = std::abs(vx) / 0.558;
        if (std::abs(wz) > max_wz) {
          wz = std::copysign(max_wz, wz);
        }

        const double dt = 0.05;
        const double mid_yaw = cur_yaw + 0.5 * wz * dt;
        actual.pose.position.x += vx * std::cos(mid_yaw) * dt;
        actual.pose.position.y += vx * std::sin(mid_yaw) * dt;
        const double next_yaw = controller_test::wrap(cur_yaw + wz * dt);
        actual.pose.orientation.z = std::sin(next_yaw / 2.0);
        actual.pose.orientation.w = std::cos(next_yaw / 2.0);
        speed.linear.x = vx;
        speed.angular.z = wz;

        const auto tf = to_target_frame(actual);
        best.x_lon = tf[0];
        best.e_right = tf[1];
        best.dth_rad = tf[2];
        best.xy_err = std::hypot(tf[0], tf[1]);
        best.final_vx = vx;
        best.ticks = tick + 1;

        if (handler.inversion() == 0u &&
          (tf[0] * scenario.gears.back() >= 0.0 ||
          (tick >= 10 && best.xy_err <= 0.10 && std::abs(best.dth_rad) <= 0.25 &&
          std::abs(vx) <= stopped_vx_thresh && std::abs(wz) <= stopped_wz_thresh)))
        {
          break;
        }
      }
      const double final_yaw = tf2::getYaw(actual.pose.orientation);
      const double final_fp_cost = checker.footprintCostAtPose(
        actual.pose.position.x, actual.pose.position.y, final_yaw, footprint);
      if (final_fp_cost < 0.0 || final_fp_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        best.collision_free = false;
      }
      printf(
        "[CLOSED_LOOP][%-8s] %-22s / %-12s -> ticks=%3zu x_lon=%+6.1fmm e_right=%+6.1fmm "
        "dth=%+5.2fdeg xy=%5.1fmm vx=%+5.3fm/s coll_free=%d latched=%d\n",
        cfg.name.c_str(),
        label.c_str(), start_key.c_str(), best.ticks,
        best.x_lon * 1000.0, best.e_right * 1000.0, best.dth_rad * 180.0 / M_PI,
        best.xy_err * 1000.0, best.final_vx,
        best.collision_free ? 1 : 0, best.latched_capture ? 1 : 0);
      return best;
    };

  auto run_scenario_closed_loop = [&](
    const Scenario & scenario, const std::string & label,
    const std::string & start_key, bool enable_capture)
    {
      CaptureCandidateConfig cfg;
      cfg.name = enable_capture ? "C3_CAP" : "BASELINE";
      cfg.enable_capture = enable_capture;
      return run_scenario_closed_loop_cfg(scenario, label, start_key, cfg);
    };

  auto run_closed_loop = [&](
    const std::string & id, const std::string & start_key, bool enable_capture)
    {
      return run_scenario_closed_loop(atlas->at(id), id, start_key, enable_capture);
    };

  constexpr double kMaxYawErrRad = 5.0 * M_PI / 180.0;

  // 1. Nominal starts: cross-track offset is 0 (< 25 mm deadband), so terminal capture
  // must not latch and must be bit-for-bit identical to baseline (both under StoppedGoalChecker
  // and when run to full rest).
  for (const auto & [id, key] : std::vector<std::pair<std::string, std::string>>{
      {"curved_capture", "nominal"},
      {"wall_capture", "nominal"},
      {"early_lateral_capture", "nominal"},
      {"local_hazard_capture", "nominal"},
      {"short_cusp_exit", "nominal"}})
  {
    SCOPED_TRACE(id + "/" + key);
    const auto base = run_closed_loop(id, key, false);
    const auto cap = run_closed_loop(id, key, true);
    EXPECT_FALSE(cap.latched_capture);
    EXPECT_TRUE(cap.collision_free);
    EXPECT_DOUBLE_EQ(cap.x_lon, base.x_lon);
    EXPECT_DOUBLE_EQ(cap.e_right, base.e_right);
    EXPECT_DOUBLE_EQ(cap.dth_rad, base.dth_rad);

    CaptureCandidateConfig rest_base{"BASE_RST", false, 0.020, 0.040, 0.10, 0.015, false, {}};
    CaptureCandidateConfig rest_cap{"CAP_RST ", true, 0.020, 0.040, 0.10, 0.015, false, {}};
    const auto base_rest = run_scenario_closed_loop_cfg(atlas->at(id), id, key, rest_base);
    const auto cap_rest = run_scenario_closed_loop_cfg(atlas->at(id), id, key, rest_cap);
    EXPECT_FALSE(cap_rest.latched_capture);
    EXPECT_TRUE(cap_rest.collision_free);
    EXPECT_LT(std::abs(cap_rest.x_lon), 0.025);
    EXPECT_LT(std::abs(cap_rest.e_right), 0.010);
    EXPECT_LT(cap_rest.xy_err, 0.025);
    EXPECT_LT(std::abs(cap_rest.dth_rad), kMaxYawErrRad);
    EXPECT_DOUBLE_EQ(cap_rest.x_lon, base_rest.x_lon);
    EXPECT_DOUBLE_EQ(cap_rest.e_right, base_rest.e_right);
    EXPECT_DOUBLE_EQ(cap_rest.dth_rad, base_rest.dth_rad);
  }

  // 2. Displaced starts: baseline fails to close lateral offset (|e_right| > 35 mm, P1),
  // whereas terminal capture latches a collision-free 1-gear curve and achieves
  // |x_lon| < 25.0 mm, |e_right| < 25.0 mm, xy_err < 25.0 mm, and |dth| < 5.0 deg
  // with zero footprint collisions (P2 & P3).
  for (const auto & [id, key] : std::vector<std::pair<std::string, std::string>>{
      {"curved_capture", "inside"},
      {"curved_capture", "outside"},
      {"wall_capture", "inside"},
      {"early_lateral_capture", "offset"},
      {"local_hazard_capture", "displaced"}})
  {
    SCOPED_TRACE(id + "/" + key);
    const auto base = run_closed_loop(id, key, false);
    const auto cap = run_closed_loop(id, key, true);
    EXPECT_GT(std::abs(base.e_right), 0.035);
    EXPECT_TRUE(cap.latched_capture);
    EXPECT_TRUE(cap.collision_free);
    EXPECT_LT(std::abs(cap.x_lon), 0.025);
    EXPECT_LT(std::abs(cap.e_right), 0.025);
    EXPECT_LT(cap.xy_err, 0.025);
    EXPECT_LT(std::abs(cap.dth_rad), kMaxYawErrRad);
  }

  // 3a. Short-segment gate (short_cusp_exit has segment length 0.25 m < 0.40 m):
  // terminal capture must not activate on short post-cusp segments and must leave
  // behavior bit-for-bit identical to baseline.
  {
    const auto base = run_closed_loop("short_cusp_exit", "stress", false);
    const auto cap = run_closed_loop("short_cusp_exit", "stress", true);
    EXPECT_FALSE(cap.latched_capture);
    EXPECT_DOUBLE_EQ(cap.x_lon, base.x_lon);
    EXPECT_DOUBLE_EQ(cap.e_right, base.e_right);
    EXPECT_DOUBLE_EQ(cap.dth_rad, base.dth_rad);
  }

  // 3b. Falsifiable minimum-turning-radius (R < terminal_capture_min_turning_r) rejection
  // inside solveTerminalCapturePath on a >= 0.40 m segment (early_lateral_capture = 1.00 m),
  // plus dynamic-parameter latch invalidation when terminal_capture_min_turning_r increases:
  {
    const auto & clean = atlas->at("early_lateral_capture");
    grid(clean);
    auto parent = node(true, 500, 0.325, 7, true);
    mppi::ParametersHandler params(parent);
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    params.start();
    handler.setPath(clean.message());

    // At (-0.45, -0.22, 0.0), goal_dist = 0.501 m in [0.22, 1.00] m and cross_track = 0.22 m
    // (>= 0.025 m), so transformPath invokes solveTerminalCapturePath. The required symmetric
    // 1-gear S-curve radius is R = (0.45^2 + 0.22^2) / (4 * 0.22) = 0.2851 m < 0.561 m,
    // which must be rejected.
    const std::array<double, 3> tight_pose{-0.45, -0.22, 0.0};
    auto rejected_seg = handler.transformPath(pose(tight_pose));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(rejected_seg.poses.empty());
    EXPECT_NEAR(rejected_seg.poses.front().pose.position.y, 0.0, 1e-6);

    // Lowering terminal_capture_min_turning_r to 0.25 m (< 0.2851 m) on the exact same pose
    // proves that R < min_turning_r was the sole rejection gate:
    parent->set_parameter(rclcpp::Parameter("FollowPath.terminal_capture_min_turning_r", 0.25));
    auto accepted_seg = handler.transformPath(pose(tight_pose));
    EXPECT_TRUE(handler.latchedCapture());
    ASSERT_FALSE(accepted_seg.poses.empty());
    EXPECT_LT(accepted_seg.poses.front().pose.position.y, -0.15);

    // Raising terminal_capture_min_turning_r back to 0.561 m while latched at the exact same
    // pose (zero positional drift) must immediately invalidate the tighter latched curve and
    // restore the nominal centerline plan:
    parent->set_parameter(rclcpp::Parameter("FollowPath.terminal_capture_min_turning_r", 0.561));
    auto invalidated_seg = handler.transformPath(pose(tight_pose));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(invalidated_seg.poses.empty());
    EXPECT_NEAR(invalidated_seg.poses.front().pose.position.y, 0.0, 1e-6);
  }

  // 4. Obstacle-blocked capture rejection (perimeter-only collision where all rear-axle
  // center cells are FREE_SPACE), dynamic-obstacle unlatching & nominal restore,
  // drift-triggered replanning (both feasible re-latch and infeasible unlatch),
  // and dynamic parameter disable verification:
  {
    Scenario blocked = atlas->at("early_lateral_capture");
    // Place an obstacle box strictly at y in [0.05, 0.14] so every point (wx, wy) on the
    // capture curve from (-0.75, -0.11, 0) to (0, 0, 0) has wy in [-0.11, 0.00] in FREE_SPACE,
    // while only the left footprint perimeter (wy + 0.163 m) intersects the box:
    blocked.obstacles.push_back(controller_test::Box{-0.55, 0.05, -0.35, 0.14, 0.15});
    grid(blocked);
    auto parent = node(true, 500, 0.325, 7, true);
    mppi::ParametersHandler params(parent);
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    params.start();
    handler.setPath(blocked.message());
    auto blocked_seg = handler.transformPath(pose(blocked.starts.at("offset")));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(blocked_seg.poses.empty());
    EXPECT_NEAR(blocked_seg.poses.front().pose.position.y, 0.0, 1e-6);

    // Now test dynamic obstacle appearing after latching on early_lateral_capture:
    const auto & clean = atlas->at("early_lateral_capture");
    grid(clean);
    handler.setPath(clean.message());
    auto latched_seg = handler.transformPath(pose(clean.starts.at("offset")));
    EXPECT_TRUE(handler.latchedCapture());
    ASSERT_FALSE(latched_seg.poses.empty());
    EXPECT_LT(latched_seg.poses.front().pose.position.y, -0.05);

    // Paint the perimeter-only obstacle and verify unlatching + nominal restore
    grid(blocked);
    auto restored_seg = handler.transformPath(pose(clean.starts.at("offset")));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(restored_seg.poses.empty());
    EXPECT_NEAR(restored_seg.poses.front().pose.position.y, 0.0, 1e-6);

    // Test drift-triggered replanning when displaced beyond terminal_capture_replan_drift_:
    grid(clean);
    handler.setPath(clean.message());
    auto first_cap = handler.transformPath(pose(clean.starts.at("offset")));
    EXPECT_TRUE(handler.latchedCapture());
    ASSERT_FALSE(first_cap.poses.empty());
    EXPECT_LT(first_cap.poses.front().pose.position.y, -0.05);

    auto drifted_cap = handler.transformPath(pose({-0.75, 0.11, 0.0}));
    EXPECT_TRUE(handler.latchedCapture());
    ASSERT_FALSE(drifted_cap.poses.empty());
    EXPECT_GT(drifted_cap.poses.front().pose.position.y, 0.05);

    // Test drift to an infeasible pose (-0.45, -0.22, 0.0 where R = 0.2851 m < 0.561 m):
    // must unlatch the stale pre-drift curve and restore the nominal centerline plan:
    auto unlatched_drift_seg = handler.transformPath(pose({-0.45, -0.22, 0.0}));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(unlatched_drift_seg.poses.empty());
    EXPECT_NEAR(unlatched_drift_seg.poses.front().pose.position.y, 0.0, 1e-6);

    // Re-latch on clean start, then test dynamic disabling of FollowPath.terminal_capture_enabled:
    handler.setPath(clean.message());
    ASSERT_GE(handler.transformPath(pose({-0.75, 0.11, 0.0})).poses.size(), 2u);
    EXPECT_TRUE(handler.latchedCapture());
    parent->set_parameter(rclcpp::Parameter("FollowPath.terminal_capture_enabled", false));
    auto disabled_seg = handler.transformPath(pose({-0.75, 0.11, 0.0}));
    EXPECT_FALSE(handler.latchedCapture());
    ASSERT_FALSE(disabled_seg.poses.empty());
    EXPECT_NEAR(disabled_seg.poses.front().pose.position.y, 0.0, 1e-6);
  }

  // 5. Complete M8 3-stroke cusp sequence with terminal_capture_enabled = true:
  {
    const auto & scenario = atlas->at("m8_full");
    grid(scenario);
    auto parent = node(true, 64, 0.558, 6, true);
    mppi::ParametersHandler params(parent);
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    std::vector<size_t> handoffs;
    for (size_t i = 0; i + 1 < scenario.path.size(); ++i) {
      const bool was_reverse = handler.reverse();
      auto segment = handler.transformPath(pose(scenario.path[i]));
      ASSERT_GE(segment.poses.size(), 2u);
      if (handler.reverse() != was_reverse) {
        handoffs.push_back(i);
      }
      EXPECT_FALSE(handler.latchedCapture());
    }
    EXPECT_EQ(handoffs, (std::vector<size_t>{7, 24}));
    EXPECT_FALSE(handler.reverse());
    EXPECT_EQ(handler.inversion(), 0u);
  }

  // 6. Reverse-gear closed-loop terminal capture verification:
  {
    const auto rev_scenario = atlas->at("early_lateral_capture").variant(false, true);
    const auto base_rev = run_scenario_closed_loop(
      rev_scenario, "early_lateral_rev", "offset", false);
    const auto cap_rev = run_scenario_closed_loop(
      rev_scenario, "early_lateral_rev", "offset", true);
    EXPECT_GT(std::abs(base_rev.e_right), 0.035);
    EXPECT_TRUE(cap_rev.latched_capture);
    EXPECT_TRUE(cap_rev.collision_free);
    EXPECT_LT(std::abs(cap_rev.x_lon), 0.025);
    EXPECT_LT(std::abs(cap_rev.e_right), 0.025);
    EXPECT_LT(cap_rev.xy_err, 0.025);
    EXPECT_LT(std::abs(cap_rev.dth_rad), kMaxYawErrRad);
  }

  // 7. Narrowing lane candidate screening (C0, C1, C2, C3) under StoppedGoalChecker
  // on undisturbed displaced approaches and mid_approach_deflection (post-hazard states
  // measured in Start -> WP1 .mcap bags at x_lon = -0.391 m and x_lon = -0.299 m):
  {
    const CaptureCandidateConfig cfg_c0{"C0_BASE", true, 0.045, 0.000, 0.00, 0.015, true, {}};
    const CaptureCandidateConfig cfg_c1{"C1_M1M2", true, 0.020, 0.040, 0.00, 0.015, true, {}};
    const CaptureCandidateConfig cfg_c2{"C2_M3  ", true, 0.045, 0.000, 0.10, 0.015, true, {}};
    const CaptureCandidateConfig cfg_c3{"C3_ALL ", true, 0.020, 0.040, 0.10, 0.015, true, {}};

    const auto & early = atlas->at("early_lateral_capture");
    const auto c0_early =
      run_scenario_closed_loop_cfg(early, "early_lateral_capture", "offset", cfg_c0);
    const auto c1_early =
      run_scenario_closed_loop_cfg(early, "early_lateral_capture", "offset", cfg_c1);
    const auto c2_early =
      run_scenario_closed_loop_cfg(early, "early_lateral_capture", "offset", cfg_c2);
    const auto c3_early =
      run_scenario_closed_loop_cfg(early, "early_lateral_capture", "offset", cfg_c3);

    // Without M3 (tangent pull-through), StoppedGoalChecker terminates early near the 100 mm rim:
    EXPECT_GT(std::abs(c0_early.x_lon), 0.025);
    EXPECT_GT(std::abs(c1_early.x_lon), 0.025);
    // With M3, C2 and C3 drive through to |x_lon| < 25 mm:
    EXPECT_LT(std::abs(c2_early.x_lon), 0.025);
    EXPECT_LT(std::abs(c3_early.x_lon), 0.025);

    // (a) mid_deflect_500mm: 31 mm cross-track drift (> 20 mm M1 threshold, < 45 mm C0 threshold)
    // at x_lon = -0.500 m, y = -0.112 m, theta = 0.0 deg, where an exact Y_end = 0 curve
    // still exists (R = 0.586 m >= 0.561 m) if re-latched immediately (M1), whereas deferring
    // until drift > 45 mm leaves the robot displaced:
    // (b) mid_deflect_299mm: post-hazard state (-0.299 m, -0.1016 m, +15.0 deg) where exact
    // Y_end = 0 requires R = 0.318 m < 0.561 m, testing M2 bounded-R_min fallback:
    const std::array<double, 3> deflect_500{-0.500, -0.112, 0.0};
    const std::array<double, 3> deflect_299{-0.299, -0.1016, 15.0 * M_PI / 180.0};

    for (const auto & [sub_label, deflect_pose] :
      std::vector<std::pair<std::string, std::array<double, 3>>>{
        {"mid_deflect_500mm", deflect_500},
        {"mid_deflect_299mm", deflect_299}})
    {
      SCOPED_TRACE(sub_label);
      auto d_c0 = cfg_c0;
      d_c0.inject_deflection = std::make_pair(size_t{1u}, deflect_pose);
      auto d_c1 = cfg_c1;
      d_c1.inject_deflection = std::make_pair(size_t{1u}, deflect_pose);
      auto d_c2 = cfg_c2;
      d_c2.inject_deflection = std::make_pair(size_t{1u}, deflect_pose);
      auto d_c3 = cfg_c3;
      d_c3.inject_deflection = std::make_pair(size_t{1u}, deflect_pose);

      const auto out_c0 = run_scenario_closed_loop_cfg(early, sub_label, "offset", d_c0);
      const auto out_c1 = run_scenario_closed_loop_cfg(early, sub_label, "offset", d_c1);
      const auto out_c2 = run_scenario_closed_loop_cfg(early, sub_label, "offset", d_c2);
      const auto out_c3 = run_scenario_closed_loop_cfg(early, sub_label, "offset", d_c3);

      // Without M1+M2, C0 and C2 fail to recover from mid-approach deflection (|e_right| > 50 mm):
      EXPECT_GT(std::abs(out_c0.e_right), 0.050);
      EXPECT_GT(std::abs(out_c2.e_right), 0.050);
      if (sub_label == "mid_deflect_299mm") {
        EXPECT_FALSE(out_c0.latched_capture);
        EXPECT_FALSE(out_c2.latched_capture);
      }
      // C1 recovers laterally relative to C0/C2, but stops longitudinally short without M3
      // (so its projected goal-line lateral error is < 38 mm while |x_lon| > 25 mm):
      const double c1_projected_e_right =
        out_c1.e_right + out_c1.x_lon * std::tan(out_c1.dth_rad);
      EXPECT_TRUE(out_c1.latched_capture);
      EXPECT_LT(std::abs(c1_projected_e_right), 0.038);
      EXPECT_GT(std::abs(out_c1.x_lon), 0.025);
      // C3 (M1 + M2 + M3) recovers both laterally (|e_right| < 38 mm) and longitudinally
      // (|x_lon| < 25 mm, xy_err < 42 mm, |dth| < 5 deg):
      EXPECT_TRUE(out_c3.latched_capture);
      EXPECT_TRUE(out_c3.collision_free);
      EXPECT_LT(std::abs(out_c3.x_lon), 0.025);
      EXPECT_LT(std::abs(out_c3.e_right), 0.038);
      EXPECT_LT(out_c3.xy_err, 0.042);
      EXPECT_LT(std::abs(out_c3.dth_rad), kMaxYawErrRad);
    }
  }
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
