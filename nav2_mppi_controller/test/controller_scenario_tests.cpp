// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_controller/plugins/stopped_goal_checker.hpp"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_mppi_controller/controller.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_mppi_controller/critics/terminal_approach_critic.hpp"
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
struct OptimizerAccess : mppi::Optimizer
{
  static auto criticManager() {return &OptimizerAccess::critic_manager_;}
  static auto noiseGenerator() {return &OptimizerAccess::noise_generator_;}
};

class ScenarioOptimizer : public mppi::Optimizer
{
public:
  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr node,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm,
    mppi::ParametersHandler * params,
    bool include_terminal_critic = false)
  {
    initialize(node, "FollowPath", cm, params);
    // Bind critics compiled from this checkout, including in the standalone CI
    // target. Never load a different installed plugin via pluginlib for this proof.
    auto & critics = critic_manager_.*CriticAccess::critics();
    std::vector<std::string> names;
    terminal_critic_ = nullptr;
    if (include_terminal_critic) {
      auto terminal_critic = std::make_unique<mppi::critics::TerminalApproachCritic>();
      terminal_critic_ = terminal_critic.get();
      critics.emplace_back(std::move(terminal_critic));
      names.push_back("TerminalApproachCritic");
    }
    critics.emplace_back(std::make_unique<mppi::critics::PathFollowCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::PathAlignCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::ObstaclesCritic>());
    names.push_back("PathFollowCritic");
    names.push_back("PathAlignCritic");
    names.push_back("ObstaclesCritic");
    critic_manager_.*CriticAccess::names() = names;
    for (size_t i = 0; i < critics.size(); ++i) {
      critics[i]->on_configure(node, "FollowPath", "FollowPath." + names[i], cm, params);
    }
    on_activate();
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
  mppi::critics::TerminalApproachCritic * terminalCritic() const
  {
    return terminal_critic_;
  }
  void addCritic(
    std::unique_ptr<mppi::critics::CriticFunction> critic, const std::string & name)
  {
    auto & critics = critic_manager_.*CriticAccess::critics();
    auto & names = critic_manager_.*CriticAccess::names();
    critic->on_activate();
    critics.emplace_back(std::move(critic));
    names.push_back(name);
  }
  void prependCritic(
    std::unique_ptr<mppi::critics::CriticFunction> critic, const std::string & name)
  {
    auto & critics = critic_manager_.*CriticAccess::critics();
    auto & names = critic_manager_.*CriticAccess::names();
    critic->on_activate();
    critics.insert(critics.begin(), std::move(critic));
    names.insert(names.begin(), name);
  }

private:
  mppi::critics::TerminalApproachCritic * terminal_critic_{nullptr};
};

class ScenarioPathHandler : public mppi::PathHandler
{
public:
  bool reverse() const {return current_segment_is_reverse_;}
  const auto & activeEndpoint() const {return global_plan_up_to_inversion_.poses.back().pose;}
  size_t activeSize() const {return global_plan_up_to_inversion_.poses.size();}
  unsigned int inversion() const {return inversion_locale_;}
};

class ScenarioController : public nav2_mppi_controller::MPPIController
{
public:
  void configureWithCompiledCritics(
    rclcpp_lifecycle::LifecycleNode::SharedPtr node,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm)
  {
    configure(node, "FollowPath", tf, cm);
    auto & critic_mgr = optimizer_.*OptimizerAccess::criticManager();
    auto & critics = critic_mgr.*CriticAccess::critics();
    auto & names = critic_mgr.*CriticAccess::names();
    critics.clear();
    names.clear();
    auto tc = std::make_unique<mppi::critics::TerminalApproachCritic>();
    terminal_critic_ = tc.get();
    critics.emplace_back(std::move(tc));
    critics.emplace_back(std::make_unique<mppi::critics::PathFollowCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::PathAlignCritic>());
    critics.emplace_back(std::make_unique<mppi::critics::ObstaclesCritic>());
    names = {
      "TerminalApproachCritic", "PathFollowCritic", "PathAlignCritic", "ObstaclesCritic"};
    for (size_t i = 0; i < critics.size(); ++i) {
      critics[i]->on_configure(
        node, "FollowPath", "FollowPath." + names[i], cm, parameters_handler_.get());
    }
    activate();
    auto & noise_gen = optimizer_.*OptimizerAccess::noiseGenerator();
    auto & vx = noise_gen.*NoiseAccess::vx();
    auto & wz = noise_gen.*NoiseAccess::wz();
    for (size_t i = 0; i < vx.shape(0); ++i) {
      for (size_t j = 0; j < vx.shape(1); ++j) {
        const float sign = i % 2 == 0 ? 1.0f : -1.0f;
        vx(i, j) = sign * 0.033f * std::sin(0.7f * (i / 2 + j + 1));
        wz(i, j) = sign * 0.101f * std::cos(0.9f * (i / 2 + j + 1));
      }
    }
  }
  mppi::critics::TerminalApproachCritic * terminalCritic() const
  {
    return terminal_critic_;
  }

private:
  mppi::critics::TerminalApproachCritic * terminal_critic_{nullptr};
};

class ControllerScenarios : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    atlas = std::make_unique<Atlas>(CONTROLLER_SCENARIO_FILE);
    Scenario local_hazard;
    local_hazard.id = "local_hazard_capture";
    local_hazard.path = {
      {-0.62984755, -0.1386433, 0.433333333},
      {-0.584127524, -0.118408513, 0.400000007},
      {-0.537758516, -0.099708681, 0.366666673},
      {-0.490792054, -0.082564584, 0.33333334},
      {-0.443280318, -0.066995269, 0.300000006},
      {-0.395276094, -0.053018034, 0.266666672},
      {-0.346832715, -0.040648408, 0.233333338},
      {-0.298004002, -0.029900134, 0.200000004},
      {-0.248844204, -0.020785153, 0.16666667},
      {-0.199407938, -0.013313592, 0.133333336},
      {-0.149750128, -0.007493752, 0.100000002},
      {-0.099925944, -0.003332099, 0.066666668},
      {-0.049990742, -0.000833256, 0.033333334},
      {0.0, 0.0, 0.0}};
    local_hazard.gears.assign(local_hazard.path.size() - 1u, 1);
    local_hazard.starts = {
      {"nominal", {-0.537758516, -0.099708681, 0.366666673}},
      {"displaced", {-0.538, -0.2438, 0.543084}}};
    local_hazard.obstacles = {{-0.85, 0.04, -0.66, 0.24, 0.11}};
    local_hazard.diagnostic_only = false;
    atlas->cases.push_back(std::move(local_hazard));

    Scenario m5_rec;
    m5_rec.id = "m5_recorded";
    m5_rec.path = {
      {-0.004839, -0.013471, 0.000000},
      {-0.032711, -0.012776, -0.049792},
      {-0.060514, -0.010696, -0.099585},
      {-0.114030, -0.005349, -0.099585},
      {-0.167546, -0.000002, -0.099585},
      {-0.221062, 0.005345, -0.099585},
      {-0.274579, 0.010692, -0.099585},
      {-0.328095, 0.016039, -0.099585},
      {-0.381611, 0.021386, -0.099585},
      {-0.435127, 0.026733, -0.099585},
      {-0.488643, 0.032080, -0.099585},
      {-0.542159, 0.037427, -0.099585},
      {-0.595675, 0.042775, -0.099585},
      {-0.643630, 0.049672, -0.186126},
      {-0.690809, 0.060689, -0.272668},
      {-0.736860, 0.075742, -0.359210},
      {-0.781436, 0.094719, -0.445752},
      {-0.824206, 0.117479, -0.532294},
      {-0.864849, 0.143850, -0.618836},
      {-0.903060, 0.173635, -0.705378},
      {-0.938554, 0.206611, -0.791919},
      {-0.971064, 0.242532, -0.878461},
      {-1.000348, 0.281128, -0.965003},
      {-1.026187, 0.322111, -1.051545},
      {-1.048387, 0.365174, -1.138087},
      {-1.066781, 0.409995, -1.224629},
      {-1.081232, 0.456238, -1.311171},
      {-1.091633, 0.503556, -1.397713},
      {-1.097904, 0.551597, -1.484254},
      {-1.100000, 0.600000, -1.570796}};
    m5_rec.gears.assign(m5_rec.path.size() - 1u, -1);
    m5_rec.starts = {
      {"nominal", {-0.004839, -0.013471, 0.000000}},
      {"entry_c1", {-0.7401, 0.1327, -0.4681}},
      {"entry_c2", {-0.7381, 0.1345, -0.4085}},
      {"entry_c3", {-0.7939, 0.1238, -0.4713}}};
    m5_rec.obstacles = {};
    m5_rec.diagnostic_only = true;
    atlas->cases.push_back(std::move(m5_rec));

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
    double stock_cutoff = 0.01)
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
        rclcpp::Parameter("FollowPath.PathAlignCritic.threshold_to_consider", stock_cutoff),
        rclcpp::Parameter("FollowPath.PathAlignCritic.offset_from_furthest", 1),
        rclcpp::Parameter("FollowPath.PathAlignCritic.cost_weight", 28.0),
        rclcpp::Parameter("FollowPath.PathAlignCritic.use_path_orientations", headings),
        rclcpp::Parameter("FollowPath.PathFollowCritic.threshold_to_consider", stock_cutoff),
        rclcpp::Parameter("FollowPath.PathFollowCritic.offset_from_furthest", pfc_offset),
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
  // First-order lagged Ackermann approximation (tau_v, tau_steer, wheel_speed_gain, steering_gain)
  struct AckermannPlantState
  {
    double v{0.0};
    double kappa{0.0};
  };
  static void stepAckermannPose(
    geometry_msgs::msg::PoseStamped & actual,
    geometry_msgs::msg::Twist & speed,
    const geometry_msgs::msg::Twist & cmd,
    AckermannPlantState & plant,
    double dt = 0.05,
    double r_min = 0.5605,
    double tau_v = 0.0,
    double tau_steer = 0.0,
    double wheel_speed_gain = 1.0,
    double steering_gain = 1.0)
  {
    const double max_kappa = 1.0 / std::max(0.10, r_min);
    const double alpha_v = (tau_v > 0.0) ? (dt / (tau_v + dt)) : 1.0;
    const double alpha_k = (tau_steer > 0.0) ? (dt / (tau_steer + dt)) : 1.0;
    const double v_target = cmd.linear.x * wheel_speed_gain;
    plant.v += alpha_v * (v_target - plant.v);
    const double cmd_kappa = (std::fabs(cmd.linear.x) > 1e-4) ?
      std::clamp((cmd.angular.z / cmd.linear.x) * steering_gain, -max_kappa, max_kappa) :
      plant.kappa;
    plant.kappa = std::clamp(plant.kappa + alpha_k * (cmd_kappa - plant.kappa), -max_kappa,
      max_kappa);
    // Ackermann constraint: zero in-place rotation when v == 0, |wz| <= |vx| / r_min always
    const double v_eff = (std::fabs(plant.v) > 1e-5) ? plant.v : 0.0;
    const double wz_eff = v_eff * plant.kappa;
    speed.linear.x = v_eff;
    speed.angular.z = wz_eff;

    const double yaw = tf2::getYaw(actual.pose.orientation);
    const double dtheta = wz_eff * dt;
    const double mid_yaw = yaw + 0.5 * dtheta;
    actual.pose.position.x += v_eff * std::cos(mid_yaw) * dt;
    actual.pose.position.y += v_eff * std::sin(mid_yaw) * dt;
    const double next_yaw = controller_test::wrap(yaw + dtheta);
    actual.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(next_yaw);
  }
  inline static std::unique_ptr<Atlas> atlas;
  inline static std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
};

TEST_F(ControllerScenarios, GeometryGearAndMirrorContracts)
{
  ASSERT_EQ(atlas->cases.size(), 17u);
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
    if (original.id == "m8_full" || original.id == "m5_recorded") {continue;}
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

TEST_F(ControllerScenarios, TerminalApproachOptionA11ClosedLoopCaptureAndInvariance)
{
  struct RolloutMetrics
  {
    bool reached{false};
    size_t ticks{0u};
    double xy_err{0.0};
    double x_lon{0.0};
    double e_right{0.0};
    double dth_deg{0.0};
    double final_vx{0.0};
    double final_wz{0.0};
    bool latched{false};
    bool ref_latched{false};
    int redraws{0};
    double end_x{0.0};
    double end_y{0.0};
    double end_yaw{0.0};
  };

  auto runClosedLoop = [&](
    const Scenario & scenario,
    const std::string & start_name,
    bool enable_a11,
    bool include_a11_plugin = true,
    bool calibrated_sim_plant = false,
    double custom_stock_cutoff = -1.0) -> RolloutMetrics
    {
      grid(scenario);
      const double stock_cutoff =
        (custom_stock_cutoff >= 0.0) ? custom_stock_cutoff : (enable_a11 ? 0.35 : 0.01);
      auto parent = node(true, 500, 0.325, 7, stock_cutoff);
      if (include_a11_plugin) {
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(enable_a11));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.min_turning_radius", rclcpp::ParameterValue(0.5605));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.infeasible_policy",
          rclcpp::ParameterValue(std::string("abort")));
      }

      // Exact live nav2_params.yaml StoppedGoalChecker configuration
      parent->declare_parameter(
        "FollowPath.goal_checker.xy_goal_tolerance", rclcpp::ParameterValue(0.10));
      parent->declare_parameter(
        "FollowPath.goal_checker.yaw_goal_tolerance", rclcpp::ParameterValue(0.25));
      parent->declare_parameter(
        "FollowPath.goal_checker.trans_stopped_velocity", rclcpp::ParameterValue(0.04));
      parent->declare_parameter(
        "FollowPath.goal_checker.rot_stopped_velocity", rclcpp::ParameterValue(0.12));
      parent->declare_parameter(
        "FollowPath.goal_checker.stateful", rclcpp::ParameterValue(false));

      nav2_controller::StoppedGoalChecker goal_checker;
      goal_checker.initialize(parent, "FollowPath.goal_checker", cm);

      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(scenario.message());

      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, include_a11_plugin);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_msg = scenario.message().poses.back().pose;
      auto actual = pose(scenario.starts.at(start_name));
      geometry_msgs::msg::Twist speed;
      speed.linear.x = scenario.gears.front() * 0.15;
      speed.angular.z = 0.0;
      AckermannPlantState plant{speed.linear.x, 0.0};
      const double tau_v = calibrated_sim_plant ? 0.176 : 0.0;
      const double tau_steer = calibrated_sim_plant ? 0.05 : 0.0;
      const double wheel_gain = calibrated_sim_plant ? 0.945 : 1.0;
      const double steer_gain = calibrated_sim_plant ? 1.21 : 1.0;

      bool ever_latched = false;
      bool ever_ref_latched = false;
      int max_redraws_seen = 0;
      bool reached = false;
      size_t ticks = 0u;

      for (; ticks < 180u; ++ticks) {
        if (goal_checker.isGoalReached(actual.pose, goal_msg, speed)) {
          reached = true;
          break;
        }
        auto local_path = handler.transformPath(actual);
        const auto cmd = optimizer.evalControl(
          actual, speed, local_path, goal_msg, &goal_checker);
        if (optimizer.terminalCritic() != nullptr) {
          ever_latched |= optimizer.terminalCritic()->isLatched();
          ever_ref_latched |= optimizer.terminalCritic()->isReferencePathLatched();
          max_redraws_seen = std::max(
            max_redraws_seen, optimizer.terminalCritic()->redrawCount());
        }
        stepAckermannPose(
          actual, speed, cmd.twist, plant, 0.05, 0.5605,
          tau_v, tau_steer, wheel_gain, steer_gain);
      }
      if (!reached && goal_checker.isGoalReached(actual.pose, goal_msg, speed)) {
        reached = true;
      }

      const double gx = goal_msg.position.x;
      const double gy = goal_msg.position.y;
      const double gyaw = tf2::getYaw(goal_msg.orientation);
      const double rx = actual.pose.position.x;
      const double ry = actual.pose.position.y;
      const double ryaw = tf2::getYaw(actual.pose.orientation);
      const double dx = rx - gx;
      const double dy = ry - gy;
      const double x_lon = dx * std::cos(gyaw) + dy * std::sin(gyaw);
      const double y_lat = -dx * std::sin(gyaw) + dy * std::cos(gyaw);
      const double e_right = -y_lat;
      const double dth_deg =
        std::fabs(controller_test::wrap(ryaw - gyaw)) * 180.0 / M_PI;

      return RolloutMetrics{
      reached,
      ticks,
      std::hypot(dx, dy),
      x_lon,
      e_right,
      dth_deg,
      speed.linear.x,
      speed.angular.z,
      ever_latched,
      ever_ref_latched,
      max_redraws_seen,
      rx,
      ry,
      ryaw};
    };

  struct DisplacedCase
  {
    std::string id;
    bool reverse;
    std::string start;
  };
  const std::vector<DisplacedCase> displaced_cases = {
    {"curved_capture", false, "inside"},
    {"curved_capture", false, "outside"},
    {"wall_capture", false, "inside"},
    {"early_lateral_capture", false, "offset"},
    {"local_hazard_capture", false, "displaced"},
    {"early_lateral_capture", true, "offset"},
  };

  for (const auto & tc : displaced_cases) {
    SCOPED_TRACE(tc.id + ":" + tc.start + (tc.reverse ? ":rev" : ":fwd"));
    const auto scenario = atlas->at(tc.id).variant(false, tc.reverse);
    const auto base = runClosedLoop(scenario, tc.start, false, true, false);
    const auto cand = runClosedLoop(scenario, tc.start, true, true, false);

    EXPECT_TRUE(cand.reached)
      << "xy=" << cand.xy_err << " vx=" << cand.final_vx << " wz=" << cand.final_wz;
    EXPECT_TRUE(cand.latched);
    EXPECT_FALSE(cand.ref_latched);
    EXPECT_EQ(cand.redraws, 0);
    EXPECT_LE(std::fabs(cand.final_vx), 0.04 + 1e-6);
    EXPECT_LE(std::fabs(cand.final_wz), 0.12 + 1e-6);
    EXPECT_LT(cand.xy_err, 0.030)
      << "base_xy=" << base.xy_err << " cand_xy=" << cand.xy_err
      << " x_lon=" << cand.x_lon << " e_right=" << cand.e_right;
    EXPECT_LT(std::fabs(cand.x_lon), 0.025);
    EXPECT_LT(std::fabs(cand.e_right), 0.025);
    EXPECT_LT(cand.dth_deg, 5.0);
    EXPECT_LT(cand.xy_err, base.xy_err);
    EXPECT_LT(std::fabs(cand.e_right), std::fabs(base.e_right));

    // Calibrated sim_plant.yaml closed-loop verification
    // (motor_tau_s=0.176, tau_steer=0.05, wheel_speed_gain=0.945, steering_gain=1.21)
    const auto sim_base = runClosedLoop(scenario, tc.start, false, true, true);
    const auto sim_cand = runClosedLoop(scenario, tc.start, true, true, true);
    EXPECT_TRUE(sim_cand.reached)
      << "sim_plant xy=" << sim_cand.xy_err << " vx=" << sim_cand.final_vx;
    EXPECT_TRUE(sim_cand.latched);
    EXPECT_FALSE(sim_cand.ref_latched);
    EXPECT_EQ(sim_cand.redraws, 0);
    EXPECT_LE(std::fabs(sim_cand.final_vx), 0.04 + 1e-6);
    EXPECT_LE(std::fabs(sim_cand.final_wz), 0.12 + 1e-6);
    EXPECT_LT(sim_cand.xy_err, 0.035)
      << "sim_base_xy=" << sim_base.xy_err << " sim_cand_xy=" << sim_cand.xy_err
      << " x_lon=" << sim_cand.x_lon << " e_right=" << sim_cand.e_right;
    EXPECT_LT(sim_cand.xy_err, sim_base.xy_err);
    EXPECT_LT(std::fabs(sim_cand.e_right), std::fabs(sim_base.e_right));
  }

  // Nominal starts:
  // (a) When TerminalApproachCritic is disabled, critic-absent vs critic-present-disabled is
  //     bit-for-bit identical (0.0 mm diff, unlatched, 0 redraws).
  // (b) Under the continuous handoff (stock critics threshold_to_consider = 0.35 m,
  //     TerminalApproachCritic approach_radius = 0.65 m, cutoff_radius = 0.35 m), stock critics
  //     switch off unconditionally at 0.35 m and TerminalApproachCritic latches the reference
  //     path in the overlap ring (latched == true, ref_latched == true) to guide the robot
  //     cleanly into StoppedGoalChecker (< 30 mm XY error).
  const std::vector<DisplacedCase> nominal_cases = {
    {"curved_capture", false, "nominal"},
    {"wall_capture", false, "nominal"},
    {"early_lateral_capture", false, "nominal"},
    {"local_hazard_capture", false, "nominal"},
    {"early_lateral_capture", true, "nominal"},
    {"short_cusp_exit", false, "nominal"},
  };

  for (const auto & tc : nominal_cases) {
    SCOPED_TRACE("nominal/" + tc.id + (tc.reverse ? ":rev" : ":fwd"));
    const auto scenario = atlas->at(tc.id).variant(false, tc.reverse);
    const auto absent = runClosedLoop(scenario, tc.start, false, false);
    const auto base = runClosedLoop(scenario, tc.start, false, true);
    const auto cand = runClosedLoop(scenario, tc.start, true, true);

    // Critic-absent vs critic-present-disabled must be bit-for-bit identical
    EXPECT_EQ(base.ticks, absent.ticks);
    EXPECT_FLOAT_EQ(static_cast<float>(base.end_x), static_cast<float>(absent.end_x));
    EXPECT_FLOAT_EQ(static_cast<float>(base.end_y), static_cast<float>(absent.end_y));
    EXPECT_FLOAT_EQ(static_cast<float>(base.end_yaw), static_cast<float>(absent.end_yaw));
    EXPECT_DOUBLE_EQ(base.end_x, absent.end_x);
    EXPECT_DOUBLE_EQ(base.end_y, absent.end_y);
    EXPECT_DOUBLE_EQ(base.end_yaw, absent.end_yaw);
    EXPECT_FALSE(base.latched);
    EXPECT_EQ(base.redraws, 0);

    // Continuous handoff with 0.35 m stock-critic cutoff: TerminalApproachCritic latches the
    // collision-free reference path and guides the robot into the goal inside the cutoff.
    EXPECT_TRUE(cand.reached) << "cand_xy=" << cand.xy_err << " vx=" << cand.final_vx;
    EXPECT_TRUE(cand.latched);
    EXPECT_TRUE(cand.ref_latched);
    EXPECT_EQ(cand.redraws, 0);
    EXPECT_LT(cand.xy_err, 0.030);
    EXPECT_LE(std::fabs(cand.final_vx), 0.04 + 1e-6);
    EXPECT_LE(std::fabs(cand.final_wz), 0.12 + 1e-6);
  }

  // Overlap-ring progress de-duplication & fresh cutoff latch (0.35 m < d_goal <= 0.65 m):
  // (a) On a nominal approach in the overlap ring (d_goal = 0.50 m > 0.35 m),
  //     TerminalApproachCritic defers freezing the reference-path latch while stock
  //     PathFollowCritic and PathAlignCritic drive the reference path with bit-for-bit identical
  //     commanded velocity to the disabled baseline (zero overlap-ring vx spike, zero stale
  //     0.65 m latch, and zero drift aborts while passive), and latches "reference_path" freshly
  //     upon crossing cutoff_radius_ (d_goal = 0.30 m <= 0.35 m).
  // (b) On a displaced capture-curve latch in the overlap ring (d_goal = 0.553 m > 0.35 m),
  //     suppressing duplicate follow_weight_ progress scoring while goal_dist > cutoff_radius_
  //     prevents vx from spiking above the baseline (whereas cutoff_radius = 0.0 spikes vx).
  // (c) If a user enables TerminalApproachCritic (cutoff_radius = 0.35) while leaving stock
  //     PathFollowCritic.threshold_to_consider at 0.01 (main default), syncStockCriticThresholds
  //     detects the 0.01 threshold, logs [TERMINAL_APPROACH_CONFIG_WARN], and suppresses
  //     duplicate reference-path follow_weight_ inside 0.35 m (d_goal = 0.30 m).
  {
    const auto & scenario = atlas->at("early_lateral_capture");
    grid(scenario);
    auto evalOverlapCmd = [&](
      const geometry_msgs::msg::PoseStamped & actual, bool enable_a11, double cutoff_r = 0.35,
      bool * out_latched = nullptr, bool * out_ref_latched = nullptr, double stock_cutoff_r = 0.35)
      {
        auto parent = node(true, 500, 0.325, 7, stock_cutoff_r);
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(enable_a11));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(cutoff_r));
        mppi::ParametersHandler params(parent);
        params.start();
        auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
        ScenarioPathHandler handler;
        handler.initialize(parent, "FollowPath", cm, buffer, &params);
        handler.setPath(scenario.message());
        ScenarioOptimizer optimizer;
        optimizer.configure(parent, cm, &params, true);
        optimizer.setSeededGaussianBank(42u);
        const auto local_path = handler.transformPath(actual);
        geometry_msgs::msg::Twist speed;
        speed.linear.x = 0.25;
        const auto cmd = optimizer.evalControl(
          actual, speed, local_path, scenario.message().poses.back().pose, nullptr);
        if (out_latched != nullptr && optimizer.terminalCritic() != nullptr) {
          *out_latched = optimizer.terminalCritic()->isLatched();
        }
        if (out_ref_latched != nullptr && optimizer.terminalCritic() != nullptr) {
          *out_ref_latched = optimizer.terminalCritic()->isReferencePathLatched();
        }
        return cmd.twist;
      };

    bool overlap_latched = true, overlap_is_ref = true;
    const auto base_nom_cmd = evalOverlapCmd(pose({-0.50, 0.0, 0.0}), false, 0.35);
    const auto cand_nom_cmd =
      evalOverlapCmd(pose({-0.50, 0.0, 0.0}), true, 0.35, &overlap_latched, &overlap_is_ref);
    EXPECT_FALSE(overlap_latched);
    EXPECT_FALSE(overlap_is_ref);
    EXPECT_DOUBLE_EQ(cand_nom_cmd.linear.x, base_nom_cmd.linear.x);
    EXPECT_DOUBLE_EQ(cand_nom_cmd.angular.z, base_nom_cmd.angular.z);

    bool cutoff_latched = false, cutoff_is_ref = false;
    evalOverlapCmd(pose({-0.30, 0.0, 0.0}), true, 0.35, &cutoff_latched, &cutoff_is_ref);
    EXPECT_TRUE(cutoff_latched);
    EXPECT_TRUE(cutoff_is_ref);

    // (c) Misconfigured stock_cutoff_r = 0.01 inside cutoff_radius (d_goal = 0.30 m):
    //     1) On a nominal reference approach, effectiveRefCutoff() = 0.01 m defers reference_path
    //        latching while stock PathFollowCritic/PathAlignCritic are still active at 0.30 m,
    //        and a 55 mm (> 45 mm) corner-cutting step at 0.25 m does NOT self-abort with
    //        DRIFT_EXCEEDED while stock critics are steering.
    //     2) On a displaced capture curve (offset at d_goal = 0.306 m <= 0.35 m), the synthetic
    //        capture curve latches, and syncStockCriticThresholds() suppresses duplicate
    //        follow_weight_ acceleration while PathFollowCritic (0.01 m) is still active.
    bool mis_latched = true, mis_is_ref = true;
    const auto base_001_cmd = evalOverlapCmd(pose({-0.30, 0.0, 0.0}), false, 0.35, nullptr, nullptr,
        0.01);
    const auto cand_001_cmd = evalOverlapCmd(
      pose({-0.30, 0.0, 0.0}), true, 0.35, &mis_latched, &mis_is_ref, 0.01);
    EXPECT_FALSE(mis_latched);
    EXPECT_FALSE(mis_is_ref);
    EXPECT_DOUBLE_EQ(cand_001_cmd.linear.x, base_001_cmd.linear.x);
    EXPECT_DOUBLE_EQ(cand_001_cmd.angular.z, base_001_cmd.angular.z);

    bool cap_001_latched = false, cap_001_is_ref = true;
    const auto base_cap_001_cmd = evalOverlapCmd(
      pose({-0.30, -0.06, 0.0}), false, 0.35, nullptr, nullptr, 0.01);
    const auto cand_cap_001_cmd = evalOverlapCmd(
      pose({-0.30, -0.06, 0.0}), true, 0.35, &cap_001_latched, &cap_001_is_ref, 0.01);
    const auto spiked_cap_001_cmd = evalOverlapCmd(
      pose({-0.30, -0.06, 0.0}), true, 0.0, nullptr, nullptr, 0.01);
    EXPECT_TRUE(cap_001_latched);
    EXPECT_FALSE(cap_001_is_ref);
    EXPECT_LE(cand_cap_001_cmd.linear.x, base_cap_001_cmd.linear.x + 1e-4);
    EXPECT_GT(spiked_cap_001_cmd.linear.x, cand_cap_001_cmd.linear.x);

    bool cap_latched = false, cap_is_ref = true;
    const auto base_cap_cmd = evalOverlapCmd(pose({-0.55, -0.06, 0.0}), false, 0.35);
    const auto cand_cap_cmd =
      evalOverlapCmd(pose({-0.55, -0.06, 0.0}), true, 0.35, &cap_latched, &cap_is_ref);
    const auto spiked_cap_cmd = evalOverlapCmd(pose({-0.55, -0.06, 0.0}), true, 0.0);
    EXPECT_TRUE(cap_latched);
    EXPECT_FALSE(cap_is_ref);
    EXPECT_LE(cand_cap_cmd.linear.x, base_cap_cmd.linear.x + 1e-4);
    EXPECT_GT(spiked_cap_cmd.linear.x, cand_cap_cmd.linear.x);
  }

  // m8_full multi-cusp default-off invariance (critic absent vs critic present & disabled)
  {
    const auto & scenario = atlas->at("m8_full");
    grid(scenario);
    auto runM8 = [&](bool enable_a11, bool include_plugin = true) {
        auto parent = node(true);
        if (include_plugin) {
          parent->declare_parameter(
            "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(enable_a11));
        }
        mppi::ParametersHandler params(parent);
        params.start();
        auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
        ScenarioPathHandler handler;
        handler.initialize(parent, "FollowPath", cm, buffer, &params);
        handler.setPath(scenario.message());
        ScenarioOptimizer optimizer;
        optimizer.configure(parent, cm, &params, include_plugin);
        const auto goal = scenario.message().poses.back().pose;
        std::vector<std::pair<double, double>> cmds;
        for (size_t i = 0; i + 1 < scenario.path.size(); ++i) {
          const auto actual = pose(scenario.path[i]);
          auto segment = handler.transformPath(actual);
          geometry_msgs::msg::Twist speed;
          speed.linear.x = handler.reverse() ? -0.1 : 0.1;
          const auto cmd = optimizer.evalControl(actual, speed, segment, goal, nullptr);
          cmds.emplace_back(cmd.twist.linear.x, cmd.twist.angular.z);
        }
        return cmds;
      };
    const auto absent_cmds = runM8(false, false);
    const auto base_cmds = runM8(false, true);
    ASSERT_EQ(base_cmds.size(), absent_cmds.size());
    for (size_t i = 0; i < base_cmds.size(); ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(base_cmds[i].first),
          static_cast<float>(absent_cmds[i].first));
      EXPECT_FLOAT_EQ(
        static_cast<float>(base_cmds[i].second), static_cast<float>(absent_cmds[i].second));
      EXPECT_DOUBLE_EQ(base_cmds[i].first, absent_cmds[i].first);
      EXPECT_DOUBLE_EQ(base_cmds[i].second, absent_cmds[i].second);
    }
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11EscalationAndDriftPolicies)
{
  // 1. Entry Infeasibility Abort: late_lateral_capture: offset with infeasible_policy="abort"
  {
    const auto & scenario = atlas->at("late_lateral_capture");
    grid(scenario);
    auto parent = node(true);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);

    const auto actual = pose(scenario.starts.at("offset"));
    const auto local_path = handler.transformPath(actual);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    try {
      optimizer.evalControl(actual, speed, local_path, scenario.message().poses.back().pose,
        nullptr);
      FAIL() << "Expected nav2_core::InvalidPath on infeasible late_lateral_capture: offset";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos)
        << ex.what();
    }
  }

  // 2. Post-Latch Drift Abort: curved_capture: inside with injected > 45 mm drift & max_redraws=0
  {
    const auto & scenario = atlas->at("curved_capture");
    grid(scenario);
    auto parent = node(true);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);

    auto actual = pose(scenario.starts.at("inside"));
    auto local_path = handler.transformPath(actual);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    const auto goal = scenario.message().poses.back().pose;

    // Tick 0 latches the capture curve
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    ASSERT_NE(optimizer.terminalCritic(), nullptr);
    EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);

    // Inject a 65 mm lateral perturbation off the latched curve
    actual.pose.position.y -= 0.065;
    local_path = handler.transformPath(actual);
    try {
      optimizer.evalControl(actual, speed, local_path, goal, nullptr);
      FAIL() <<
        "Expected nav2_core::InvalidPath when drift > max_tracking_drift with max_redraws=0";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos)
        << ex.what();
    }
  }

  // 3. Continuous-Handoff Replan Signal Policy (1 replacement per attempt):
  //    - In the overlap ring (0.35 m < d_goal <= 0.65 m), requests at most one replacement
  //      plan via Bool(true) on ~/terminal_replan_requested while stock critics keep guiding
  //      the robot.
  //    - Duplicate ticks, controller.reset(), and setPlan() cannot replenish the 1-replacement
  //      allowance.
  //    - If the replacement is not ready by cutoff_radius (d_goal <= 0.35 m), or if the
  //      replacement approach fails, the attempt fails immediately (throws InvalidPath) with
  //      no further replans or hidden recovery loop.
  {
    const auto & late_scenario = atlas->at("late_lateral_capture");
    const auto & early_scenario = atlas->at("early_lateral_capture");
    grid(late_scenario);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("replan_signal")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    auto * tc = controller.terminalCritic();
    ASSERT_NE(tc, nullptr);

    bool received_replan_true = false;
    bool received_replan_false = false;
    int true_msg_count = 0;
    auto sub = parent->create_subscription<std_msgs::msg::Bool>(
      "FollowPath/terminal_replan_requested",
      rclcpp::QoS(1).transient_local(),
      [&received_replan_true, &received_replan_false, &true_msg_count](
        const std_msgs::msg::Bool::SharedPtr msg)
      {
        if (msg->data) {
          received_replan_true = true;
          ++true_msg_count;
        } else {
          received_replan_false = true;
        }
      });

    auto spinForFlag = [&](const bool & flag) {
        for (int i = 0; i < 20 && !flag; ++i) {
          rclcpp::spin_some(parent->get_node_base_interface());
          if (!flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
      };

    // (3a) Infeasibility in the overlap ring (d_goal = 0.513 m > 0.35 m cutoff):
    //      requests one replacement plan while stock critics keep guiding the robot.
    controller.setPlan(late_scenario.message());
    auto overlap_bad_pose = pose({-0.48, -0.18, 0.0});
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    geometry_msgs::msg::TwistStamped cmd;
    EXPECT_NO_THROW(cmd = controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_GT(std::fabs(cmd.twist.linear.x), 0.01);
    EXPECT_TRUE(tc->replanSignalSent());
    EXPECT_TRUE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);
    spinForFlag(received_replan_true);
    EXPECT_TRUE(received_replan_true);
    EXPECT_FALSE(received_replan_false);
    EXPECT_EQ(true_msg_count, 1);

    // (3b) Continuing to drive in the overlap ring (d_goal = 0.439 m > 0.35 m) while waiting
    //      for the replacement plan: no planned stop, no duplicate signal, no throw.
    auto overlap_drive_pose = pose({-0.40, -0.18, 0.0});
    EXPECT_NO_THROW(cmd = controller.computeVelocityCommands(overlap_drive_pose, speed, nullptr));
    EXPECT_GT(std::fabs(cmd.twist.linear.x), 0.01);
    EXPECT_TRUE(tc->replanSignalSent());
    EXPECT_TRUE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);
    rclcpp::spin_some(parent->get_node_base_interface());
    EXPECT_EQ(true_msg_count, 1);
    EXPECT_FALSE(received_replan_false);

    // (3c) Replacement not ready in time: crossing cutoff_radius (d_goal = 0.320 m <= 0.35 m)
    //      while replan_pending == true immediately fails the attempt (REPLAN_NOT_READY_BY_CUTOFF).
    auto inside_cutoff_pose = pose({-0.28, -0.15, 0.0});
    try {
      controller.computeVelocityCommands(inside_cutoff_pose, speed, nullptr);
      FAIL() << "Expected InvalidPath when replacement is not ready by cutoff_radius";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=REPLAN_NOT_READY_BY_CUTOFF"), std::string::npos)
        << ex.what();
    }
    EXPECT_FALSE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    // Within the same attempt (without controller.reset()), setPlan() to the same goal cannot
    // replenish the consumed allowance or recover the failed attempt.
    controller.setPlan(early_scenario.message());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_THROW(
      controller.computeVelocityCommands(pose({-0.48, -0.06, 0.0}), speed, nullptr),
      nav2_core::InvalidPath);
    EXPECT_EQ(tc->replanCount(), 1);
    EXPECT_EQ(true_msg_count, 1);

    // (3d) Next navigation task via Nav2's actual task-end lifecycle (controller.reset() +
    //      controller.setPlan(), WITHOUT controller.deactivate() / activate()):
    //      reset() clears the prior task's consumed replan allowance and retracts Bool(false).
    //      When the replacement arrives in time in the overlap ring, it retracts Bool(false),
    //      and if the replacement approach is infeasible, the attempt fails immediately with
    //      no second replan signal.
    received_replan_true = false;
    received_replan_false = false;
    controller.reset();
    EXPECT_FALSE(tc->isReplanAllowanceConsumed());
    EXPECT_FALSE(tc->isReplanPending());
    EXPECT_EQ(tc->replanCount(), 0);
    spinForFlag(received_replan_false);
    EXPECT_TRUE(received_replan_false);

    grid(late_scenario);
    controller.setPlan(late_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_TRUE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);
    spinForFlag(received_replan_true);
    EXPECT_TRUE(received_replan_true);
    EXPECT_EQ(true_msg_count, 2);

    // Deliver an infeasible replacement plan via setPlan(): retracts Bool(false), clears
    // replan_pending, preserves replan_allowance_consumed == true, and throws InvalidPath
    // on evaluation without emitting a second replan signal.
    received_replan_false = false;
    controller.setPlan(late_scenario.message());
    EXPECT_FALSE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    spinForFlag(received_replan_false);
    EXPECT_TRUE(received_replan_false);

    EXPECT_THROW(
      controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr),
      nav2_core::InvalidPath);
    EXPECT_EQ(tc->replanCount(), 1);
    rclcpp::spin_some(parent->get_node_base_interface());
    EXPECT_EQ(true_msg_count, 2);

    // (3e) Consecutive navigation task via controller.reset() + controller.setPlan() (NO
    //      deactivate/activate): Task 1 uses its 1 replan in the overlap ring, receives a
    //      feasible replacement plan, latches, and finishes via controller.reset(); then
    //      Task 2 starts via controller.setPlan() and also has its full 1-replan allowance!
    controller.reset();
    EXPECT_FALSE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 0);
    grid(late_scenario);
    controller.setPlan(late_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_EQ(tc->replanCount(), 1);
    EXPECT_TRUE(tc->isReplanPending());

    grid(early_scenario);
    controller.setPlan(early_scenario.message());
    auto feasible_replacement_pose = pose({-0.48, -0.06, 0.0});
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    // Periodic setPlan() to the exact same goal and gear after replacement latching must
    // preserve the active latch (no false unlatch on periodic updates).
    controller.setPlan(early_scenario.message());
    EXPECT_TRUE(tc->isLatched());
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());

    // Task 1 finishes normally: Nav2's ControllerServer::endTask() calls controller.reset()
    // (NOT deactivate/activate). Task 2 then starts with controller.setPlan() to a second goal
    // (and also works for the same goal), must NOT throw InvalidPath on entry, and can use its
    // own 1-replan allowance!
    controller.reset();
    EXPECT_FALSE(tc->isLatched());
    EXPECT_FALSE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 0);

    // Task 2: first verify a nominal/feasible task after Task 1's replan+reset succeeds cleanly
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_FALSE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 0);

    // Task 3 (also via controller.reset() + setPlan() without deactivate/activate):
    // requests its own replacement plan, latches on the replacement, and if post-latch drift
    // later exceeds max_tracking_drift within Task 3, Task 3 fails immediately with no
    // second replan!
    controller.reset();
    grid(late_scenario);
    controller.setPlan(late_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_EQ(tc->replanCount(), 1);
    EXPECT_TRUE(tc->isReplanPending());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());

    grid(early_scenario);
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    feasible_replacement_pose.pose.position.y -= 0.065;
    EXPECT_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr),
      nav2_core::InvalidPath);
    EXPECT_EQ(tc->replanCount(), 1);

    // Within the failed Task 3 (before controller.reset()), another setPlan() to the same goal
    // is rejected as a second replacement plan
    feasible_replacement_pose = pose({-0.48, -0.06, 0.0});
    controller.setPlan(early_scenario.message());
    try {
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr);
      FAIL() << "Expected SECOND_REPLACEMENT_PLAN_REJECTED within same failed attempt";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=SECOND_REPLACEMENT_PLAN_REJECTED"), std::string::npos)
        << ex.what();
    }

    // (3f) Review #2 & #3 Round 5 (Hole B): A large (> 0.25 m) local odom-frame localization shift
    //      (0.30 m = hypot(0.24, -0.18)) WITHOUT a global goal change in setPlan() must NEVER
    //      replenish replan_allowance_consumed_ or replan_count_ inside score()!
    buffer->setUsingDedicatedThread(true);
    auto setGlobalToMapTf = [&](double dx, double dy) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = rclcpp::Time(0);
        tf_msg.header.frame_id = "global_map";
        tf_msg.child_frame_id = "map";
        tf_msg.transform.translation.x = -dx;
        tf_msg.transform.translation.y = -dy;
        tf_msg.transform.rotation.w = 1.0;
        buffer->setTransform(tf_msg, "test_authority", true);
      };
    setGlobalToMapTf(0.0, 0.0);
    nav_msgs::msg::Path global_late = late_scenario.message();
    global_late.header.frame_id = "global_map";
    for (auto & ps : global_late.poses) {
      ps.header.frame_id = "global_map";
    }
    nav_msgs::msg::Path global_early = early_scenario.message();
    global_early.header.frame_id = "global_map";
    for (auto & ps : global_early.poses) {
      ps.header.frame_id = "global_map";
    }

    controller.reset();
    grid(late_scenario);
    controller.setPlan(global_late);
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    grid(early_scenario);
    controller.setPlan(global_early);
    feasible_replacement_pose = pose({-0.48, -0.06, 0.0});
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(feasible_replacement_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    // Inject a 0.30 m (+0.24 m, -0.18 m) local-frame TF shift (> 0.25 m goal_reset_pos_tol_)
    // WITHOUT calling setPlan() to a new global goal: score() must NOT clear
    // replan_allowance_consumed_ or replan_count_, and must abort under max_redraws = 0!
    setGlobalToMapTf(0.24, -0.18);
    auto tf_shifted_pose = feasible_replacement_pose;
    tf_shifted_pose.pose.position.x += 0.24;
    tf_shifted_pose.pose.position.y -= 0.18;
    try {
      controller.computeVelocityCommands(tf_shifted_pose, speed, nullptr);
      FAIL() << "Expected LOCAL_GOAL_SHIFT_AFTER_REPLAN_CONSUMED on 0.30 m local TF shift";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LOCAL_GOAL_SHIFT_AFTER_REPLAN_CONSUMED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    // Even if periodic setPlan(global_early) is called to the same global goal after the 0.30 m
    // local TF shift, onNewPlan() still recognizes same_goal in global_map and rejects it!
    controller.setPlan(global_early);
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);
    try {
      controller.computeVelocityCommands(tf_shifted_pose, speed, nullptr);
      FAIL() << "Expected SECOND_REPLACEMENT_PLAN_REJECTED after 0.30 m local TF shift";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=SECOND_REPLACEMENT_PLAN_REJECTED"),
        std::string::npos) << ex.what();
    }
    setGlobalToMapTf(0.0, 0.0);

    // Sub-millimetre global goal jitter (0.5 mm <= kGlobalGoalPosTol = 1 mm) in the same
    // global frame is still treated as the SAME global goal and does NOT replenish
    // replan_allowance_consumed_:
    nav_msgs::msg::Path sub_mm_goal_plan = global_early;
    for (auto & ps : sub_mm_goal_plan.poses) {
      ps.pose.position.x += 5e-4;
    }
    controller.setPlan(sub_mm_goal_plan);
    EXPECT_TRUE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 1);

    // Whereas sending a setPlan() to a genuinely NEW global goal (0.15 m shift:
    // > kGlobalGoalPosTol = 1 mm, even though < goal_reset_pos_tol_ = 0.25 m)
    // resets the allowance for the new goal even without an explicit controller.reset()!
    nav_msgs::msg::Path shifted_goal_plan = early_scenario.message();
    for (auto & ps : shifted_goal_plan.poses) {
      ps.pose.position.x += 0.15;
    }
    controller.setPlan(shifted_goal_plan);
    EXPECT_FALSE(tc->isReplanAllowanceConsumed());
    EXPECT_EQ(tc->replanCount(), 0);
    auto shifted_pose = pose({-0.48 + 0.15, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(shifted_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
  }

  // 4. Post-latch "fallback" policy stays unlatched across inner MPPI iterations & next tick
  {
    struct IterationLatchProbeCritic : public mppi::critics::CriticFunction
    {
      void initialize() override {}
      void score(mppi::CriticData & /*data*/) override
      {
        if (tc != nullptr) {
          observed_latched.push_back(tc->isLatched());
        }
      }
      mppi::critics::TerminalApproachCritic * tc{nullptr};
      std::vector<bool> observed_latched;
    };

    const auto & scenario = atlas->at("curved_capture");
    grid(scenario);
    auto parent = node(true);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("fallback")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);

    auto probe_uptr = std::make_unique<IterationLatchProbeCritic>();
    auto * probe = probe_uptr.get();
    probe->tc = optimizer.terminalCritic();
    probe_uptr->on_configure(
      parent, "FollowPath", "FollowPath.IterationLatchProbeCritic", cm, &params);
    optimizer.addCritic(std::move(probe_uptr), "IterationLatchProbeCritic");

    const auto goal = scenario.message().poses.back().pose;
    auto actual = pose(scenario.starts.at("inside"));
    auto local_path = handler.transformPath(actual);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    ASSERT_NE(optimizer.terminalCritic(), nullptr);
    EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(probe->observed_latched, (std::vector<bool>{true, true, true}));

    // Inject 65 mm lateral perturbation; "fallback" must unlatch on iteration 0 and remain
    // unlatched across all inner iterations (i = 0, 1, 2) and on the subsequent tick
    probe->observed_latched.clear();
    actual.pose.position.y -= 0.065;
    local_path = handler.transformPath(actual);
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(probe->observed_latched, (std::vector<bool>{false, false, false}));

    probe->observed_latched.clear();
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(probe->observed_latched, (std::vector<bool>{false, false, false}));
  }

  // 5. CriticManager preserves declared critic order without cross-critic coupling
  {
    struct StaticCriticManager : mppi::CriticManager
    {
      void loadCritics() override {}
    };
    auto parent = std::make_shared<rclcpp_lifecycle::LifecycleNode>("critic_order_node");
    parent->declare_parameter(
      "FollowPath.critics",
      rclcpp::ParameterValue(
        std::vector<std::string>{
        "PathFollowCritic", "PathAlignCritic", "TerminalApproachCritic"}));
    mppi::ParametersHandler params(parent);
    params.start();
    StaticCriticManager mgr;
    mgr.on_configure(parent, "FollowPath", cm, &params);
    const auto & loaded_names = mgr.*CriticAccess::names();
    ASSERT_EQ(loaded_names.size(), 3u);
    EXPECT_EQ(loaded_names[0], "PathFollowCritic");
    EXPECT_EQ(loaded_names[1], "PathAlignCritic");
    EXPECT_EQ(loaded_names[2], "TerminalApproachCritic");
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11PullthroughTaperFootprintAndCacheInvalidation)
{
  const auto & scenario = atlas->at("curved_capture");
  grid(scenario);
  auto parent = node(true);
  parent->declare_parameter(
    "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
  parent->declare_parameter(
    "FollowPath.TerminalApproachCritic.infeasible_policy",
    rclcpp::ParameterValue(std::string("abort")));
  mppi::ParametersHandler params(parent);
  params.start();
  auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
  ScenarioPathHandler handler;
  handler.initialize(parent, "FollowPath", cm, buffer, &params);
  handler.setPath(scenario.message());
  ScenarioOptimizer optimizer;
  optimizer.configure(parent, cm, &params, true);

  auto * tc = optimizer.terminalCritic();
  ASSERT_NE(tc, nullptr);

  // 1. Smooth pull-through taper formula & release at <= 0.015 m
  EXPECT_FLOAT_EQ(tc->computePullthroughLength(0.30f), 0.0f);
  EXPECT_FLOAT_EQ(tc->computePullthroughLength(0.10f), 0.10f);
  EXPECT_NEAR(tc->computePullthroughLength(0.05f), 0.0525f, 1e-5f);
  EXPECT_NEAR(tc->computePullthroughLength(0.03f), 0.0225f, 1e-5f);
  EXPECT_FLOAT_EQ(tc->computePullthroughLength(0.015f), 0.0f);
  EXPECT_FLOAT_EQ(tc->computePullthroughLength(0.005f), 0.0f);
  EXPECT_FLOAT_EQ(tc->computePullthroughLength(-0.01f), 0.0f);

  // 2. Pruned local path endpoint gate & one-plan post-latch flicker regression
  //    (Review #1 Round 3 Finding 1 & Review #2 Round 3 Finding 2):
  //    (a) If prune_distance truncates local_path before latching, do NOT latch.
  //    (b) If local_path transiently omits the goal AFTER latching under max_redraws = 0,
  //        score() clears the active curve and marks the approach consumed so when the
  //        terminal goal reappears on local_path, it does NOT solve a second capture curve.
  {
    auto actual = pose(scenario.starts.at("inside"));
    auto full_path = handler.transformPath(actual);
    auto truncated_path = full_path;
    ASSERT_GT(truncated_path.poses.size(), 4u);
    truncated_path.poses.resize(truncated_path.poses.size() - 3u);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    const auto goal = scenario.message().poses.back().pose;

    // (a) Truncated before latching -> remains unlatched
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, truncated_path, goal, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // (b) Latch on full_path -> truncate local_path (!has_terminal_goal) -> restore full_path
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, full_path, goal, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, truncated_path, goal, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Restoring the terminal goal (even with a periodic onNewPlan to the same goal & gear)
    // must NOT permit a second capture curve solve on the same approach under max_redraws = 0!
    optimizer.onNewPlan(scenario.message());
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, full_path, goal, nullptr));
    EXPECT_FALSE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // Reaching cutoff_radius (d_goal = 0.200 m <= 0.35 m) while unlatched/escalated must
    // fail the attempt immediately rather than leaving zero active critics inside the cutoff!
    auto inside_cutoff_actual = pose({0.443280318, -0.066995269, -0.300000006});
    auto inside_cutoff_path = handler.transformPath(inside_cutoff_actual);
    try {
      optimizer.evalControl(inside_cutoff_actual, speed, inside_cutoff_path, goal, nullptr);
      FAIL() << "Expected InvalidPath when unlatched at cutoff_radius";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=UNLATCHED_AT_CUTOFF"), std::string::npos)
        << ex.what();
    }

    // Explicit task reset clears the consumed state for the next test block
    optimizer.resetCritics();
  }

  // 3. Latch on full terminal path, then verify dynamic parameter update invalidates latch
  {
    auto actual = pose(scenario.starts.at("inside"));
    auto local_path = handler.transformPath(actual);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());
    ASSERT_GT(tc->latchedCurve().size(), 4u);

    // Verify that when PathHandler prunes local_path down to a single final vertex (n_path == 1)
    // in the final half-segment (< 2.5 cm from goal), the active latch is NOT dropped
    nav_msgs::msg::Path single_vertex_path;
    single_vertex_path.header = local_path.header;
    single_vertex_path.poses.push_back(scenario.message().poses.back());
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, single_vertex_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());

    const auto mid_pose = tc->latchedCurve()[tc->latchedCurve().size() / 2u];

    // 4. Same-costmap in-place obstacle mutation WHILE LATCHED (Review #2 Round 2 Finding 2):
    // Do NOT touch parameters first; verify that an obstacle appearing on the existing
    // latched curve is detected by the latched-curve recheck and throws LATCHED_CURVE_BLOCKED.
    const double obs_x = mid_pose.x - 0.13 * std::sin(mid_pose.theta);
    const double obs_y = mid_pose.y + 0.13 * std::cos(mid_pose.theta);
    unsigned int ox = 0u, oy = 0u, cx = 0u, cy = 0u;
    ASSERT_TRUE(cm->getCostmap()->worldToMap(obs_x, obs_y, ox, oy));
    ASSERT_TRUE(cm->getCostmap()->worldToMap(mid_pose.x, mid_pose.y, cx, cy));
    ASSERT_TRUE(ox != cx || oy != cy);
    cm->getCostmap()->setCost(ox, oy, nav2_costmap_2d::LETHAL_OBSTACLE);
    ASSERT_EQ(cm->getCostmap()->getCost(cx, cy), nav2_costmap_2d::FREE_SPACE);
    EXPECT_TRUE(tc->isLatched());

    try {
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr);
      FAIL() << "Expected nav2_core::InvalidPath with LATCHED_CURVE_BLOCKED";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(std::string(ex.what()).find("reason=LATCHED_CURVE_BLOCKED"), std::string::npos)
        << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // 5. Clear obstacle, resetCritics(), re-latch, and verify:
    //    (a) Leaving the 0.70 m hysteresis circle while latched and re-entering preserves the
    //        original latched curve with 0 redraws and does NOT throw UNLATCHED_AT_CUTOFF.
    //    (b) Dynamic parameter invalidation while latched under max_redraws = 0 clears the
    //        active curve and marks the approach consumed so it does NOT solve a second curve
    //        on the same approach (Review #2 & #3 Round 4 Finding 3).
    cm->getCostmap()->setCost(ox, oy, nav2_costmap_2d::FREE_SPACE);
    optimizer.resetCritics();
    optimizer.onNewPlan(scenario.message());
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());
    const auto initial_curve = tc->latchedCurve();
    ASSERT_FALSE(initial_curve.empty());

    // (a) Excursion outside 0.70 m hysteresis circle while latched preserves the original curve,
    //     including when stepping inside cutoff_radius_ (<= 0.35 m)
    auto excursion_pose = pose({-0.25, 0.0, 0.0});
    EXPECT_NO_THROW(
      optimizer.evalControl(
        excursion_pose, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());
    const auto & inner_pt = initial_curve[(initial_curve.size() * 2u) / 3u];
    auto inner_pose = pose({inner_pt.x, inner_pt.y, inner_pt.theta});
    auto inner_local_path = handler.transformPath(inner_pose);
    EXPECT_NO_THROW(
      optimizer.evalControl(
        inner_pose, speed, inner_local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);
    ASSERT_EQ(tc->latchedCurve().size(), initial_curve.size());
    EXPECT_FLOAT_EQ(tc->latchedCurve().front().x, initial_curve.front().x);

    // (b) Dynamic parameter update while latched under max_redraws = 0 unlatches and prevents
    //     solving a second curve on the same approach (including if enabled is toggled off/on)
    const auto set_res = parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.min_turning_radius", 0.60)});
    ASSERT_EQ(set_res.size(), 1u);
    EXPECT_TRUE(set_res[0].successful);
    EXPECT_FALSE(tc->isLatched());

    // Evaluating again on the same approach must remain unlatched (no second curve solve)
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Toggling enabled false -> true on the same approach must also NOT solve a second curve
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.enabled", false)})[0].successful);
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.enabled", true)})[0].successful);
    EXPECT_NO_THROW(
      optimizer.evalControl(
        actual, speed, local_path, scenario.message().poses.back().pose, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Stepping inside cutoff_radius_ (<= 0.35 m) after the dynamic parameter change while
    // latched escalates with PARAM_CHANGE_WHILE_LATCHED rather than UNLATCHED_AT_CUTOFF:
    try {
      optimizer.evalControl(
        inner_pose, speed, inner_local_path, scenario.message().poses.back().pose, nullptr);
      FAIL() << "Expected PARAM_CHANGE_WHILE_LATCHED inside cutoff_radius_ after param update";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=PARAM_CHANGE_WHILE_LATCHED"),
        std::string::npos) << ex.what();
    }

    // Restore default min_turning_radius (0.5605 m) and reset for next block
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.min_turning_radius", 0.5605)})[0]
      .successful);
    optimizer.resetCritics();
  }

  // 6. Strict live Goal-Checker endpoint tolerance validation (Review #2 Round 2 Finding 1):
  //    Verify that a bounded curve ending ~2.2 cm from the goal is accepted under a 10 cm
  //    StoppedGoalChecker (max_end_xy = 3.5 cm), rejected at entry under a 2 cm StoppedGoalChecker
  //    (max_end_xy = 0.7 cm), and invalidated with LATCHED_TOLERANCE_VIOLATED if tolerances
  //    tighten to 2 cm while already latched.
  {
    const auto & early_scenario = atlas->at("early_lateral_capture");
    grid(early_scenario);
    handler.setPath(early_scenario.message());

    parent->declare_parameter(
      "FollowPath.std_gc.xy_goal_tolerance", rclcpp::ParameterValue(0.10));
    parent->declare_parameter(
      "FollowPath.std_gc.yaw_goal_tolerance", rclcpp::ParameterValue(0.25));
    parent->declare_parameter(
      "FollowPath.std_gc.trans_stopped_velocity", rclcpp::ParameterValue(0.04));
    parent->declare_parameter(
      "FollowPath.std_gc.rot_stopped_velocity", rclcpp::ParameterValue(0.12));
    parent->declare_parameter(
      "FollowPath.std_gc.stateful", rclcpp::ParameterValue(false));
    nav2_controller::StoppedGoalChecker std_gc;
    std_gc.initialize(parent, "FollowPath.std_gc", cm);

    parent->declare_parameter(
      "FollowPath.tight_gc.xy_goal_tolerance", rclcpp::ParameterValue(0.02));
    parent->declare_parameter(
      "FollowPath.tight_gc.yaw_goal_tolerance", rclcpp::ParameterValue(0.05));
    parent->declare_parameter(
      "FollowPath.tight_gc.trans_stopped_velocity", rclcpp::ParameterValue(0.04));
    parent->declare_parameter(
      "FollowPath.tight_gc.rot_stopped_velocity", rclcpp::ParameterValue(0.12));
    parent->declare_parameter(
      "FollowPath.tight_gc.stateful", rclcpp::ParameterValue(false));
    nav2_controller::StoppedGoalChecker tight_gc;
    tight_gc.initialize(parent, "FollowPath.tight_gc", cm);

    // Case A (Review #2's tight-tolerance endpoint validation): at [-0.25, -0.050, 0.0],
    // R_exact < 0.5605 m and Family 2a's bounded bi-arc ends ~2.13 cm from the goal (> 2.0 cm
    // tight_gc tolerance, <= 2.5 cm max_residual_y). Must be rejected at entry under tight_gc
    // (ENTRY_INFEASIBLE), while accepted under 10 cm std_gc.
    auto entry_2cm_pose = pose({-0.25, -0.050, 0.0});
    auto entry_2cm_path = handler.transformPath(entry_2cm_pose);
    const auto goal = early_scenario.message().poses.back().pose;
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    optimizer.resetCritics();
    try {
      optimizer.evalControl(entry_2cm_pose, speed, entry_2cm_path, goal, &tight_gc);
      FAIL() << "Expected ENTRY_INFEASIBLE under 2 cm goal_checker at [-0.25, -0.050, 0.0]";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos)
        << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // Case B (Live tolerance tightening while latched): at [-0.34, -0.055, 0.0] under the 10 cm
    // StoppedGoalChecker, Family 2a latches a bounded curve ending ~1.52 cm from the goal
    // (<= 2.5 cm limit). When the live goal_checker then tightens to 2 cm (limit 0.5 cm),
    // the next latched tick must detect LATCHED_TOLERANCE_VIOLATED.
    auto latch_pose = pose({-0.34, -0.055, 0.0});
    auto latch_path = handler.transformPath(latch_pose);
    optimizer.onNewPlan(early_scenario.message());
    EXPECT_NO_THROW(optimizer.evalControl(latch_pose, speed, latch_path, goal, &std_gc));
    EXPECT_TRUE(tc->isLatched());
    ASSERT_FALSE(tc->latchedCurve().empty());
    const double latched_end_err = std::hypot(
      tc->latchedCurve().back().x - goal.position.x,
      tc->latchedCurve().back().y - goal.position.y);
    EXPECT_GT(latched_end_err, 0.005);
    EXPECT_LE(latched_end_err, 0.025 + 1e-5);

    try {
      optimizer.evalControl(latch_pose, speed, latch_path, goal, &tight_gc);
      FAIL() << "Expected LATCHED_TOLERANCE_VIOLATED when live goal_checker tightens to 2 cm";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LATCHED_TOLERANCE_VIOLATED"), std::string::npos)
        << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11M5M8RetryAndLifecycleRegressions)
{
  auto makeGoalChecker = [&](rclcpp_lifecycle::LifecycleNode::SharedPtr parent) {
      parent->declare_parameter(
        "FollowPath.goal_checker.xy_goal_tolerance", rclcpp::ParameterValue(0.10));
      parent->declare_parameter(
        "FollowPath.goal_checker.yaw_goal_tolerance", rclcpp::ParameterValue(0.25));
      parent->declare_parameter(
        "FollowPath.goal_checker.trans_stopped_velocity", rclcpp::ParameterValue(0.04));
      parent->declare_parameter(
        "FollowPath.goal_checker.rot_stopped_velocity", rclcpp::ParameterValue(0.12));
      parent->declare_parameter(
        "FollowPath.goal_checker.stateful", rclcpp::ParameterValue(false));
      auto gc = std::make_unique<nav2_controller::StoppedGoalChecker>();
      gc->initialize(parent, "FollowPath.goal_checker", cm);
      return gc;
    };

  // 1. M5 (reverse_left_90) candidate rejection trace at d=0.59m, 60 deg from final goal heading,
  //    plus closed-loop completion without false abort
  {
    Scenario m5_reverse;
    m5_reverse.id = "m5_reverse_left_90";
    // 90-deg reverse curve of radius R_path = 0.42 m in Goal Travel Frame (gear = -1,
    // goal at [0, 0, 0] with nose yaw 0.0 -> travel yaw pi).
    // In world frame, robot backs from x > 0 toward [0, 0, 0]:
    // X_travel = -dx, Y_travel = -dy, theta_travel = wrap(nose_yaw)
    const double r_path = 0.42;
    const size_t n_pts = 20u;
    const double th_entry = 1.25;
    const double x_arc0 = r_path * std::sin(th_entry);
    const double y_arc0 = r_path * (1.0 - std::cos(th_entry));
    m5_reverse.path.push_back(
      {x_arc0 + 0.14 * std::cos(th_entry), y_arc0 + 0.14 * std::sin(th_entry), th_entry});
    m5_reverse.gears.push_back(-1);
    for (size_t i = 0; i <= n_pts; ++i) {
      const double frac = static_cast<double>(n_pts - i) / static_cast<double>(n_pts);
      const double th = th_entry * frac;  // 71.6 deg down to 0 deg
      const double x_tr = -r_path * std::sin(th);
      const double y_tr = -r_path * (1.0 - std::cos(th));
      m5_reverse.path.push_back({-x_tr, -y_tr, th});
      if (i < n_pts) {
        m5_reverse.gears.push_back(-1);
      }
    }
    // Exact Gazebo M5 circle-entry state in Goal Travel Frame:
    // X0 = -0.43 m, Y0 = -0.41 m, theta0 = +1.05 rad (~60 deg), d = 0.594 m,
    // ~3 cm cross-track from the R=0.42m curved reference path
    m5_reverse.starts["entry_60deg"] = {0.43, 0.41, 1.05};
    grid(m5_reverse);

    auto parent = node(true, 500, 0.325, 7, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto goal_checker = makeGoalChecker(parent);
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(m5_reverse.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    optimizer.setSeededGaussianBank(42u);

    auto * tc = optimizer.terminalCritic();
    ASSERT_NE(tc, nullptr);
    const auto goal_msg = m5_reverse.message().poses.back().pose;
    auto actual = pose(m5_reverse.starts.at("entry_60deg"));

    // Verify exact mathematical rejection reasons across all candidate families at M5 entry
    const auto traces = tc->traceCaptureCandidates(actual.pose, goal_msg, -1.0, 0.10, 0.25);
    ASSERT_GE(traces.size(), 5u);
    auto findTrace = [&](const std::string & fam) -> const mppi::critics::CaptureFamilyTrace * {
        for (const auto & tr : traces) {
          if (tr.family == fam) {
            return &tr;
          }
        }
        return nullptr;
      };
    const auto * tr_biarc_pos = findTrace("1a_biarc_s=+1");
    const auto * tr_sc = findTrace("1b_sc_straight_arc");
    const auto * tr_cs = findTrace("1c_cs_arc_straight");
    const auto * tr_b_biarc = findTrace("2a_bounded_biarc_s=+1");
    ASSERT_NE(tr_biarc_pos, nullptr);
    ASSERT_NE(tr_sc, nullptr);
    ASSERT_NE(tr_cs, nullptr);
    ASSERT_NE(tr_b_biarc, nullptr);
    EXPECT_FALSE(tr_biarc_pos->accepted);
    EXPECT_EQ(tr_biarc_pos->rejection_reason, "R < min_turning_r");
    EXPECT_NEAR(tr_biarc_pos->R, 0.400, 0.015);
    EXPECT_FALSE(tr_sc->accepted);
    EXPECT_EQ(tr_sc->rejection_reason, "R < min_turning_r");
    EXPECT_NEAR(tr_sc->R, 0.334, 0.015);
    EXPECT_FALSE(tr_cs->accepted);
    EXPECT_NE(tr_cs->rejection_reason.find("Ls < 0"), std::string::npos);
    EXPECT_NEAR(tr_cs->R, 0.822, 0.015);
    EXPECT_LT(tr_cs->Ls, -0.25);
    EXPECT_FALSE(tr_b_biarc->accepted);
    EXPECT_FALSE(tr_b_biarc->rejection_reason.empty());

    // Closed-loop execution must NOT false-abort on M5, must latch the reference path in the
    // overlap ring, and must guide the robot across the 0.35 m cutoff to StoppedGoalChecker.
    geometry_msgs::msg::Twist speed;
    speed.linear.x = -0.15;
    AckermannPlantState plant{speed.linear.x, 0.0};
    bool reached = false;
    for (size_t tick = 0; tick < 180u; ++tick) {
      if (goal_checker->isGoalReached(actual.pose, goal_msg, speed)) {
        reached = true;
        break;
      }
      auto local_path = handler.transformPath(actual);
      geometry_msgs::msg::TwistStamped cmd;
      ASSERT_NO_THROW(
        cmd = optimizer.evalControl(actual, speed, local_path, goal_msg, goal_checker.get()))
        << "False abort on M5 at tick " << tick;
      stepAckermannPose(actual, speed, cmd.twist, plant, 0.05, 0.5605);
    }
    if (!reached && goal_checker->isGoalReached(actual.pose, goal_msg, speed)) {
      reached = true;
    }
    EXPECT_TRUE(reached)
      << "M5 end=(" << actual.pose.position.x << "," << actual.pose.position.y
      << ") vx=" << speed.linear.x;
    EXPECT_TRUE(tc->isLatched());
    EXPECT_TRUE(tc->isReferencePathLatched());

    // 1b. Recorded Gazebo M5 Smac plan (m5_recorded, R=0.560m reverse-left 90-deg turn)
    //     Falsifiability proof: with admission guards disabled, all 3 recorded Gazebo circle-entry
    //     states (entry_c1, entry_c2, entry_c3) FAIL with ENTRY_INFEASIBLE; with default admission
    //     guards enabled, all 3 circle-entry states AND full lagged rollout from nominal latch
    //     the reference path and PASS across the 0.35 m cutoff.
    const auto & m5_rec = atlas->at("m5_recorded");
    for (const std::string & m5_start : {
        std::string("entry_c1"),
        std::string("entry_c2"),
        std::string("entry_c3"),
        std::string("nominal")})
    {
      for (const bool guards_enabled : {false, true}) {
        if (m5_start == "nominal" && !guards_enabled) {
          continue;
        }
        SCOPED_TRACE(
          "m5_recorded/" + m5_start + (guards_enabled ? "/guards_on" : "/guards_off"));
        grid(m5_rec);
        auto m5_node = node(true, 500, 0.325, 7, 0.35);
        m5_node->declare_parameter(
          "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
        m5_node->declare_parameter(
          "FollowPath.TerminalApproachCritic.infeasible_policy",
          rclcpp::ParameterValue(std::string("abort")));
        m5_node->declare_parameter(
          "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
        if (!guards_enabled) {
          m5_node->declare_parameter(
            "FollowPath.TerminalApproachCritic.max_entry_path_turn_angle",
            rclcpp::ParameterValue(3.14));
          m5_node->declare_parameter(
            "FollowPath.TerminalApproachCritic.short_approach_distance",
            rclcpp::ParameterValue(0.0));
        }
        auto m5_gc = makeGoalChecker(m5_node);
        mppi::ParametersHandler m5_params(m5_node);
        m5_params.start();
        auto m5_buf = std::make_shared<tf2_ros::Buffer>(m5_node->get_clock());
        ScenarioPathHandler m5_handler;
        m5_handler.initialize(m5_node, "FollowPath", cm, m5_buf, &m5_params);
        m5_handler.setPath(m5_rec.message());
        ScenarioOptimizer m5_opt;
        m5_opt.configure(m5_node, cm, &m5_params, true);
        m5_opt.setSeededGaussianBank(42u);

        const auto m5_goal = m5_rec.message().poses.back().pose;
        auto m5_pose = pose(m5_rec.starts.at(m5_start));
        geometry_msgs::msg::Twist m5_speed;
        m5_speed.linear.x = -0.15;
        AckermannPlantState m5_plant{m5_speed.linear.x, 0.0};
        const double tau_v = (m5_start == "nominal") ? 0.176 : 0.0;
        const double tau_k = (m5_start == "nominal") ? 0.10 : 0.0;

        bool m5_aborted = false;
        bool m5_reached = false;
        for (size_t tick = 0; tick < 220u; ++tick) {
          if (m5_gc->isGoalReached(m5_pose.pose, m5_goal, m5_speed)) {
            m5_reached = true;
            break;
          }
          auto local_path = m5_handler.transformPath(m5_pose);
          try {
            const auto cmd = m5_opt.evalControl(
              m5_pose, m5_speed, local_path, m5_goal, m5_gc.get());
            stepAckermannPose(m5_pose, m5_speed, cmd.twist, m5_plant, 0.05, 0.5605, tau_v, tau_k);
          } catch (const nav2_core::InvalidPath & ex) {
            m5_aborted = true;
            EXPECT_NE(std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos)
              << ex.what();
            break;
          }
        }
        if (!m5_aborted && !m5_reached && m5_gc->isGoalReached(m5_pose.pose, m5_goal, m5_speed)) {
          m5_reached = true;
        }
        if (!guards_enabled) {
          EXPECT_TRUE(m5_aborted) << "Expected ENTRY_INFEASIBLE abort on " << m5_start
                                  << " when admission guards are disabled";
        } else {
          EXPECT_FALSE(m5_aborted) << "Unexpected abort on " << m5_start;
          EXPECT_TRUE(m5_reached) << "Failed to reach goal on " << m5_start;
          EXPECT_TRUE(m5_opt.terminalCritic()->isLatched()) << m5_start;
          EXPECT_TRUE(m5_opt.terminalCritic()->isReferencePathLatched()) << m5_start;
        }
      }
    }
  }

  // 2. M8 post-cusp handoff poses (all 3 Gazebo attempts), short_cusp_exit: stress,
  //    Review #1's worked example [-0.20, 0.05, 0.0], and a true 2-segment sub-min_approach
  //    cusp scenario (reverse -> cusp at x=-0.10 -> 10 cm forward stub to [0, 0, 0]) latch the
  //    reference path and complete in closed loop inside the 0.35 m stock-critic cutoff with
  //    0 aborts under Ackermann plant, falsifiably abort when short-approach guards are disabled,
  //    and reject kinematically unreachable heading errors (> viable_yaw + goal_dist / r_min).
  {
    Scenario short_exit = atlas->at("short_cusp_exit");
    short_exit.starts["m8_attempt_1"] = {-0.248, 0.430, -0.5624};
    short_exit.starts["m8_attempt_2"] = {-0.197, 0.404, -0.4995};
    short_exit.starts["m8_attempt_3"] = {-0.200, 0.398, -0.5112};
    Scenario straight_short = atlas->at("early_lateral_capture");
    straight_short.starts["review1_worked_example"] = {-0.200, 0.050, 0.0};

    // True 2-segment cusp path (reverse into cusp at (-0.12, 0.0, 0.0), then 12 cm forward stub
    // to (0.0, 0.0, 0.0), where the post-cusp stub length 0.12 m < min_approach_distance_ 0.16 m
    // while remaining outside the 0.10 m xy_goal_tolerance at cusp handoff)
    Scenario sub_min_cusp;
    sub_min_cusp.id = "sub_min_dist_cusp";
    sub_min_cusp.path = {
      {0.10, -0.04, -0.20},
      {0.03, -0.02, -0.15},
      {-0.03, -0.01, -0.10},
      {-0.08, -0.005, -0.05},
      {-0.12, 0.0, 0.0},      // Cusp vertex (inversion point at index 5)
      {-0.08, 0.0, 0.0},      // Post-cusp forward stub (< 0.16 m from goal)
      {-0.04, 0.0, 0.0},
      {0.0, 0.0, 0.0}};
    sub_min_cusp.gears = {-1, -1, -1, -1, 1, 1, 1};
    sub_min_cusp.starts = {
      {"pre_cusp_reverse", {-0.065, 0.025, -0.340}},
      {"post_cusp_reachable", {-0.115, 0.025, -0.340}},
      {"post_cusp_unreachable_yaw", {-0.115, 0.025, -0.500}},
      {"post_cusp_pointing_away", {-0.115, 0.080, 0.300}},
      {"post_cusp_abeam", {0.000, 0.085, -0.200}}};
    sub_min_cusp.obstacles = {};
    auto sub_min_cusp_msg = sub_min_cusp.message();
    ASSERT_EQ(mppi::utils::findFirstPathInversion(sub_min_cusp_msg), 5u);

    // Falsifiability proof: without the short-approach / viable-corridor admission guards,
    // M8 post-cusp handoff (including the < 16 cm 2-segment post-cusp stub with -0.34 rad yaw
    // error) and Review #1's worked example abort with ENTRY_INFEASIBLE
    for (const std::string & fail_key : {
        std::string("m8_attempt_1"),
        std::string("m8_attempt_2"),
        std::string("m8_attempt_3"),
        std::string("review1_worked_example"),
        std::string("post_cusp_reachable")})
    {
      const Scenario & sc =
        (fail_key == "review1_worked_example") ? straight_short :
        ((fail_key == "post_cusp_reachable") ? sub_min_cusp : short_exit);
      grid(sc);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.short_approach_distance",
        rclcpp::ParameterValue(0.0));
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(sc.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      const auto goal_msg = sc.message().poses.back().pose;
      auto actual = pose(sc.starts.at(fail_key));
      auto local_path = handler.transformPath(actual);
      if (fail_key == "post_cusp_reachable") {
        // First transformPath at (-0.115, 0.025) is within 0.15 m of cusp (-0.12, 0.0), so
        // ScenarioPathHandler prunes past the cusp onto the 12 cm forward stub (inversion == 0)
        if (handler.inversion() != 0u) {
          local_path = handler.transformPath(actual);
        }
        ASSERT_EQ(handler.inversion(), 0u);
        ASSERT_FALSE(handler.reverse());
      }
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.05;
      EXPECT_THROW(
        optimizer.evalControl(actual, speed, local_path, goal_msg, goal_checker.get()),
        nav2_core::InvalidPath)
        << "Expected ENTRY_INFEASIBLE on " << fail_key << " when guards are disabled";
    }

    // Kinematic reachability bound proof on the 2-segment post-cusp stub (< 0.16 m):
    // 1) post_cusp_unreachable_yaw (-0.115, 0.025, -0.500): |yaw_err| = 0.500 rad >
    //    0.25 + 0.115 / 0.5605 = 0.4552 rad -> ENTRY_INFEASIBLE.
    // 2) post_cusp_pointing_away (-0.115, 0.080, +0.300): y0 = +0.080 m and theta0 = +0.300 rad
    //    have the same sign (pointing away from y=0), so |y_end_best| = 0.1027 m > 0.095 m
    //    -> ENTRY_INFEASIBLE.
    // 3) post_cusp_abeam (0.000, 0.085, -0.200): d_goal = 0.085 m < 0.16 m, |y0| <= 0.095 m,
    //    |theta0| <= 0.25 rad, but lon_remaining = 0.000 m < 0.02 m (abeam of goal)
    //    -> ENTRY_INFEASIBLE.
    for (const std::string & unreach_key : {
        std::string("post_cusp_unreachable_yaw"),
        std::string("post_cusp_pointing_away"),
        std::string("post_cusp_abeam")})
    {
      SCOPED_TRACE("sub_min_cusp_unreach/" + unreach_key);
      grid(sub_min_cusp);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(sub_min_cusp.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      const auto goal_msg = sub_min_cusp.message().poses.back().pose;
      auto actual = pose(sub_min_cusp.starts.at(unreach_key));
      auto local_path = handler.transformPath(actual);
      while (handler.inversion() != 0u) {
        auto cusp_p = pose({-0.12, 0.0, 0.0});
        (void)handler.transformPath(cusp_p);
        local_path = handler.transformPath(actual);
      }
      ASSERT_EQ(handler.inversion(), 0u);
      ASSERT_FALSE(handler.reverse());
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.05;
      try {
        optimizer.evalControl(actual, speed, local_path, goal_msg, goal_checker.get());
        FAIL() << "Expected ENTRY_INFEASIBLE on kinematically unreachable " << unreach_key;
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos) << ex.what();
      }
      EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
    }

    // Closed-loop 2-segment cusp traversal (pre_cusp_reverse -> cusp at -0.12 m -> forward stub):
    // Falsifiability: with short_approach_distance = 0.0, aborts with ENTRY_INFEASIBLE upon
    // switching gear across the cusp at d_goal < 0.16 m with |yaw_err| > 0.25 rad; with
    // short_approach_distance = 0.35, latches reference_path and reaches final_xy <= 0.03 m.
    for (const bool guards_on : {false, true}) {
      SCOPED_TRACE(guards_on ? "pre_cusp_reverse/guards_on" : "pre_cusp_reverse/guards_off");
      grid(sub_min_cusp);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
      if (!guards_on) {
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.short_approach_distance",
          rclcpp::ParameterValue(0.0));
      }
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(sub_min_cusp.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_msg = sub_min_cusp.message().poses.back().pose;
      auto actual = pose(sub_min_cusp.starts.at("pre_cusp_reverse"));
      geometry_msgs::msg::Twist speed;
      speed.linear.x = -0.05;
      AckermannPlantState plant{speed.linear.x, 0.0};
      bool reached = false;
      bool aborted = false;
      size_t cusp_switches = 0u;
      bool prev_rev = true;
      for (size_t tick = 0; tick < 160u; ++tick) {
        if (handler.inversion() == 0u &&
          goal_checker->isGoalReached(actual.pose, goal_msg, speed))
        {
          reached = true;
          break;
        }
        auto local_path = handler.transformPath(actual);
        if (tick == 0u) {
          ASSERT_TRUE(handler.reverse());
        } else if (handler.reverse() != prev_rev) {
          ++cusp_switches;
          prev_rev = handler.reverse();
        }
        try {
          const auto cmd = optimizer.evalControl(
            actual, speed, local_path, goal_msg, goal_checker.get());
          stepAckermannPose(actual, speed, cmd.twist, plant, 0.05, 0.5605);
        } catch (const nav2_core::InvalidPath & ex) {
          aborted = true;
          EXPECT_NE(
            std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos) << ex.what();
          break;
        }
      }
      if (!aborted && !reached && handler.inversion() == 0u &&
        goal_checker->isGoalReached(actual.pose, goal_msg, speed))
      {
        reached = true;
      }
      EXPECT_EQ(cusp_switches, 1u);
      if (!guards_on) {
        EXPECT_TRUE(aborted) << "Expected ENTRY_INFEASIBLE after cusp switch when guards_on=false";
      } else {
        const double final_xy = std::hypot(
          actual.pose.position.x - goal_msg.position.x,
          actual.pose.position.y - goal_msg.position.y);
        EXPECT_FALSE(aborted);
        EXPECT_TRUE(reached) << "final_xy=" << final_xy;
        EXPECT_LE(final_xy, 0.030);
        EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
        EXPECT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());
      }
    }

    for (const std::string & start_key : {
        std::string("stress"),
        std::string("m8_attempt_1"),
        std::string("m8_attempt_2"),
        std::string("m8_attempt_3"),
        std::string("review1_worked_example"),
        std::string("post_cusp_reachable")})
    {
      SCOPED_TRACE("short_exit/" + start_key);
      const Scenario & sc =
        (start_key == "review1_worked_example") ? straight_short :
        ((start_key == "post_cusp_reachable") ? sub_min_cusp : short_exit);
      grid(sc);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(sc.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_msg = sc.message().poses.back().pose;
      auto actual = pose(sc.starts.at(start_key));
      if (start_key == "post_cusp_reachable") {
        // Advance ScenarioPathHandler onto the post-cusp forward stub; verify that at tick 0
        // actual is outside xy_goal_tolerance (0.1177 m > 0.10 m) so evalControl MUST run.
        auto lp = handler.transformPath(actual);
        (void)lp;
        ASSERT_EQ(handler.inversion(), 0u);
      }
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.05;
      ASSERT_FALSE(goal_checker->isGoalReached(actual.pose, goal_msg, speed)) << start_key;
      AckermannPlantState plant{speed.linear.x, 0.0};
      bool reached = false;
      for (size_t tick = 0; tick < 140u; ++tick) {
        if (goal_checker->isGoalReached(actual.pose, goal_msg, speed)) {
          reached = true;
          break;
        }
        auto local_path = handler.transformPath(actual);
        geometry_msgs::msg::TwistStamped cmd;
        ASSERT_NO_THROW(
          cmd = optimizer.evalControl(actual, speed, local_path, goal_msg, goal_checker.get()))
          << "Unexpected abort on " << start_key << " at tick " << tick;
        stepAckermannPose(actual, speed, cmd.twist, plant, 0.05, 0.5605);
      }
      if (!reached && goal_checker->isGoalReached(actual.pose, goal_msg, speed)) {
        reached = true;
      }
      const double final_xy = std::hypot(
        actual.pose.position.x - goal_msg.position.x,
        actual.pose.position.y - goal_msg.position.y);
      EXPECT_TRUE(reached) << start_key << " final_xy=" << final_xy << " vx=" << speed.linear.x;
      EXPECT_LE(final_xy, (start_key == "post_cusp_reachable") ? 0.030 : 0.060) << start_key;
      EXPECT_TRUE(optimizer.terminalCritic()->isLatched()) << start_key;
      EXPECT_TRUE(optimizer.terminalCritic()->isReferencePathLatched()) << start_key;
    }

    // Post-latch drift envelope consistency & ratcheting (m3 post-cusp 0.062 -> 0.100 m):
    // 1) When reference_path latches at initial_cross_track = 0.062 m (<= 0.095 m viable limit),
    //    the initial post-latch drift limit is latch_initial_cross_track_ + max_tracking_drift_
    //    (0.062 + 0.045 = 0.107 m):
    //     - A +0.038 m post-latch excursion (to 0.100 m <= 0.107 m) does NOT abort;
    //     - A +0.055 m post-latch excursion (to 0.117 m > 0.107 m) DOES abort with DRIFT_EXCEEDED.
    // 2) When reference_path latches at initial_cross_track = 0.000 m ((-0.30, 0.0, 0.0)), a
    //    0.060 m excursion (> 0.045 m, < 0.095 m) aborts with DRIFT_EXCEEDED (proving the limit
    //    is not a blanket 0.095 m waiver).
    // 3) As the robot converges from 0.062 m to 0.010 m, latch_initial_cross_track_ ratchets down
    //    to 0.010 m so a subsequent excursion to 0.070 m (> 0.010 + 0.045 = 0.055 m) aborts.
    {
      grid(straight_short);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(straight_short.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);

      const auto goal_msg = straight_short.message().poses.back().pose;
      auto latch_p = pose({-0.12, 0.062, 0.20});
      auto local_p = handler.transformPath(latch_p);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.05;
      EXPECT_NO_THROW(optimizer.evalControl(latch_p, speed, local_p, goal_msg, goal_checker.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      ASSERT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());

      // +38 mm post-latch excursion (total cross_track = 0.100 m <= 0.062 + 0.045 = 0.107 m):
      auto coast_p = pose({-0.13, 0.100, 0.18});
      local_p = handler.transformPath(coast_p);
      EXPECT_NO_THROW(optimizer.evalControl(coast_p, speed, local_p, goal_msg, goal_checker.get()));
      EXPECT_TRUE(optimizer.terminalCritic()->isLatched());

      // +55 mm post-latch excursion (total cross_track = 0.117 m > 0.107 m) -> DRIFT_EXCEEDED:
      auto over_p = pose({-0.13, 0.117, 0.18});
      local_p = handler.transformPath(over_p);
      try {
        optimizer.evalControl(over_p, speed, local_p, goal_msg, goal_checker.get());
        FAIL() << "Expected DRIFT_EXCEEDED when post-latch drift (55 mm) exceeds 45 mm limit";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
      }
      EXPECT_FALSE(optimizer.terminalCritic()->isLatched());

      // Zero-initial-offset latch at (-0.30, 0.0, 0.0): 60 mm (< 95 mm) excursion -> DRIFT_EXCEEDED
      handler.setPath(straight_short.message());
      optimizer.terminalCritic()->onNewPlan(straight_short.message());
      auto zero_latch_p = pose({-0.30, 0.0, 0.0});
      local_p = handler.transformPath(zero_latch_p);
      EXPECT_NO_THROW(
        optimizer.evalControl(zero_latch_p, speed, local_p, goal_msg, goal_checker.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      auto sixty_mm_p = pose({-0.28, 0.060, 0.0});
      local_p = handler.transformPath(sixty_mm_p);
      EXPECT_THROW(
        optimizer.evalControl(sixty_mm_p, speed, local_p, goal_msg, goal_checker.get()),
        nav2_core::InvalidPath);

      // Convergence ratcheting: latch at 0.062 m, converge to 0.010 m, then drift to 0.070 m
      handler.setPath(straight_short.message());
      optimizer.terminalCritic()->onNewPlan(straight_short.message());
      local_p = handler.transformPath(latch_p);
      EXPECT_NO_THROW(optimizer.evalControl(latch_p, speed, local_p, goal_msg, goal_checker.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      auto converged_p = pose({-0.11, 0.010, 0.05});
      local_p = handler.transformPath(converged_p);
      EXPECT_NO_THROW(
        optimizer.evalControl(converged_p, speed, local_p, goal_msg, goal_checker.get()));
      auto rebound_p = pose({-0.11, 0.070, 0.05});
      local_p = handler.transformPath(rebound_p);
      EXPECT_THROW(
        optimizer.evalControl(rebound_p, speed, local_p, goal_msg, goal_checker.get()),
        nav2_core::InvalidPath);
    }

    // Curved-turn overlap-ring corner-cutting regression (m7_forward_180 r3 at d_goal ~ 0.50 m,
    // cross_track ~ 0.097 m > 0.095 m while stock PathFollowCritic steers in 0.35..0.65 m):
    // Must NOT self-abort in the overlap ring while stock critics are steering, and must latch
    // reference_path freshly inside cutoff_radius_ (<= 0.35 m).
    {
      const auto & m5_rec = atlas->at("m5_recorded");
      grid(m5_rec);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
      auto goal_checker = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(m5_rec.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);

      const auto goal_msg = m5_rec.message().poses.back().pose;
      geometry_msgs::msg::Twist speed;
      speed.linear.x = -0.15;
      // Entry at d_goal ~ 0.62 m (entry_c1): defers reference_path latch in overlap ring
      auto entry_p = pose(m5_rec.starts.at("entry_c1"));
      auto local_p = handler.transformPath(entry_p);
      EXPECT_NO_THROW(optimizer.evalControl(entry_p, speed, local_p, goal_msg, goal_checker.get()));
      EXPECT_FALSE(optimizer.terminalCritic()->isLatched());

      // Mid-overlap corner-cut at d_goal ~ 0.50 m with cross_track ~ 0.100 m (> 0.095 m,
      // <= 0.095 + turn_sagitta_allowance = 0.122 m): must NOT abort with DRIFT_EXCEEDED or
      // ENTRY_INFEASIBLE while stock critics steer!
      auto corner_cut_p = pose({-0.840, 0.255, -0.750});
      local_p = handler.transformPath(corner_cut_p);
      EXPECT_NO_THROW(
        optimizer.evalControl(corner_cut_p, speed, local_p, goal_msg, goal_checker.get()));
      EXPECT_FALSE(optimizer.terminalCritic()->isLatched());

      // Crossing inside cutoff_radius_ (d_goal ~ 0.28 m <= 0.35 m) latches reference_path freshly:
      auto inside_cutoff_p = pose({-1.030, 0.340, -1.100});
      local_p = handler.transformPath(inside_cutoff_p);
      EXPECT_NO_THROW(
        optimizer.evalControl(inside_cutoff_p, speed, local_p, goal_msg, goal_checker.get()));
      EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
      EXPECT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());
    }
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11ClosedLoopM8AndRetryRegressions)
{
  auto makeGoalChecker = [&](rclcpp_lifecycle::LifecycleNode::SharedPtr parent) {
      parent->declare_parameter(
        "FollowPath.goal_checker.xy_goal_tolerance", rclcpp::ParameterValue(0.10));
      parent->declare_parameter(
        "FollowPath.goal_checker.yaw_goal_tolerance", rclcpp::ParameterValue(0.25));
      parent->declare_parameter(
        "FollowPath.goal_checker.trans_stopped_velocity", rclcpp::ParameterValue(0.04));
      parent->declare_parameter(
        "FollowPath.goal_checker.rot_stopped_velocity", rclcpp::ParameterValue(0.12));
      parent->declare_parameter(
        "FollowPath.goal_checker.stateful", rclcpp::ParameterValue(false));
      auto gc = std::make_unique<nav2_controller::StoppedGoalChecker>();
      gc->initialize(parent, "FollowPath.goal_checker", cm);
      return gc;
    };

  // 3. Full closed-loop M8 (m8_full: forward -> reverse -> forward across 2 cusps)
  //    completes with TerminalApproachCritic enabled and 0.35 m stock-critic cutoff
  //    with 0 aborts under both ideal Ackermann plant and first-order lagged Ackermann
  //    approximation (tau_v=0.176s, tau_steer=0.05s, wheel_gain=0.945, steer_gain=1.21)
  for (const bool calibrated_sim_plant : {false, true}) {
    SCOPED_TRACE(
      calibrated_sim_plant ? "m8_full/lagged_ackermann" : "m8_full/ideal_ackermann");
    const auto & scenario = atlas->at("m8_full");
    grid(scenario);
    auto parent = node(true, 500, 0.325, 7, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto goal_checker = makeGoalChecker(parent);
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    optimizer.setSeededGaussianBank(42u);

    const auto goal_msg = scenario.message().poses.back().pose;
    auto actual = pose(scenario.starts.at("nominal"));
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    AckermannPlantState plant{speed.linear.x, 0.0};
    const double tau_v = calibrated_sim_plant ? 0.176 : 0.0;
    const double tau_steer = calibrated_sim_plant ? 0.05 : 0.0;
    const double wheel_gain = calibrated_sim_plant ? 0.945 : 1.0;
    const double steer_gain = calibrated_sim_plant ? 1.21 : 1.0;

    bool reached = false;
    size_t cusp_switches = 0u;
    bool prev_rev = handler.reverse();
    for (size_t tick = 0; tick < 360u; ++tick) {
      if (handler.inversion() == 0u &&
        goal_checker->isGoalReached(actual.pose, goal_msg, speed))
      {
        reached = true;
        break;
      }
      auto local_path = handler.transformPath(actual);
      if (handler.reverse() != prev_rev) {
        ++cusp_switches;
        prev_rev = handler.reverse();
      }
      geometry_msgs::msg::TwistStamped cmd;
      ASSERT_NO_THROW(
        cmd = optimizer.evalControl(actual, speed, local_path, goal_msg, goal_checker.get()))
        << "Abort on closed-loop m8_full at tick " << tick;
      stepAckermannPose(
        actual, speed, cmd.twist, plant, 0.05, 0.5605,
        tau_v, tau_steer, wheel_gain, steer_gain);
    }
    EXPECT_EQ(cusp_switches, 2u);
    EXPECT_TRUE(reached)
      << "m8_full end=(" << actual.pose.position.x << "," << actual.pose.position.y << ")";
    EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
    EXPECT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());
  }

  // 4. Optimizer fallback retry (retry_attempt_limit = 1) with FailOnceProbeCritic at index 0
  //    (before TerminalApproachCritic): verifies both (a) that Optimizer::reset() preserves
  //    latched curve & redraw_count across retries, and (b) that Optimizer::prepare() ->
  //    CriticManager::onTickStart() resets tick_evaluated_ so when attempt 0 fails BEFORE
  //    TerminalApproachCritic runs, attempt 1 (reuse_obstacles_cache == true) still executes
  //    TerminalApproachCritic's drift/obstacle evaluation on that tick (Review #2 Finding 1).
  {
    struct FailOnceProbeCritic : public mppi::critics::CriticFunction
    {
      void initialize() override {}
      void score(mppi::CriticData & data) override
      {
        ++calls;
        if (trigger_fail_once) {
          trigger_fail_once = false;
          data.fail_flag = true;
        }
      }
      bool trigger_fail_once{false};
      int calls{0};
    };

    const auto & scenario = atlas->at("curved_capture");
    grid(scenario);
    auto parent = node(true);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(scenario.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    ASSERT_TRUE(
      parent->set_parameters({rclcpp::Parameter("FollowPath.retry_attempt_limit", 1)})[0]
      .successful);

    // Prepend FailOnceProbeCritic at index 0 BEFORE TerminalApproachCritic so when it sets
    // data.fail_flag = true on attempt 0 (reuse_obstacles_cache == false), CriticManager breaks
    // out of evalTrajectoriesScores() before TerminalApproachCritic::score() is called.
    auto probe_uptr = std::make_unique<FailOnceProbeCritic>();
    auto * probe = probe_uptr.get();
    probe_uptr->on_configure(
      parent, "FollowPath", "FollowPath.FailOnceProbeCritic", cm, &params);
    optimizer.prependCritic(std::move(probe_uptr), "FailOnceProbeCritic");

    const auto goal = scenario.message().poses.back().pose;
    auto actual = pose(scenario.starts.at("inside"));
    auto local_path = handler.transformPath(actual);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    // Tick 0: latches capture curve (redrawCount == 0)
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    auto * tc = optimizer.terminalCritic();
    ASSERT_NE(tc, nullptr);
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // Tick 1: perturb by 55 mm (> 45 mm) so the single permitted redraw is consumed
    actual.pose.position.y -= 0.055;
    local_path = handler.transformPath(actual);
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // Tick 2: force attempt 0 to fail at index 0 so fallback() runs and retries on attempt 1.
    // Preserves isLatched() and redrawCount() == 1 across Optimizer::reset().
    probe->trigger_fail_once = true;
    probe->calls = 0;
    EXPECT_NO_THROW(optimizer.evalControl(actual, speed, local_path, goal, nullptr));
    EXPECT_GT(probe->calls, 3);  // Attempt 0 (1 call) + Attempt 1 (3 iterations)
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // Tick 3: combine attempt-0 failure at index 0 WITH a second > 45 mm drift perturbation on
    // the SAME tick. Without onTickStart() in Optimizer::prepare(), TerminalApproachCritic would
    // never see reuse_obstacles_cache == false on Tick 3 and would skip the drift check on
    // attempt 1! With onTickStart(), attempt 1 evaluates drift and throws InvalidPath.
    actual.pose.position.y -= 0.055;
    local_path = handler.transformPath(actual);
    probe->trigger_fail_once = true;
    probe->calls = 0;
    EXPECT_THROW(
      optimizer.evalControl(actual, speed, local_path, goal, nullptr),
      nav2_core::InvalidPath);
    EXPECT_EQ(probe->calls, 2);  // Attempt 0 failed at idx 0, Attempt 1 threw on iteration 0
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11LifecycleAndReplanRegressions)
{
  // 5. MPPIController::setPlan() and MPPIController::reset() clear prior abort/latch state,
  //    and an 80 mm (> max_tracking_drift) odom shift rigidly transforms latched_curve_
  {
    const auto & late_scenario = atlas->at("late_lateral_capture");
    grid(late_scenario);
    auto parent = node(true);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_tracking_drift", rclcpp::ParameterValue(0.045));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    auto * tc = controller.terminalCritic();
    ASSERT_NE(tc, nullptr);

    controller.setPlan(late_scenario.message());
    const auto goal = late_scenario.message().poses.back().pose;
    const auto bad_pose = pose(late_scenario.starts.at("offset"));
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    // Trigger abort for goal [0, 0, 0] through MPPIController::computeVelocityCommands()
    EXPECT_THROW(
      controller.computeVelocityCommands(bad_pose, speed, nullptr),
      nav2_core::InvalidPath);
    // Without a new plan, repeating the same goal pose immediately aborts again
    EXPECT_THROW(
      controller.computeVelocityCommands(bad_pose, speed, nullptr),
      nav2_core::InvalidPath);

    // Calling MPPIController::setPlan() with a new plan to the SAME goal pose [0, 0, 0]
    // must clear the prior abort and allow latching on the new plan
    const auto & early_scenario = atlas->at("early_lateral_capture");
    grid(early_scenario);
    controller.setPlan(early_scenario.message());

    auto good_pose = pose({-0.55, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // Calling MPPIController::reset() must also clear the critic latch
    controller.reset();
    EXPECT_FALSE(tc->isLatched());

    // Re-latch on early_scenario and consume the 1 permitted redraw (redrawCount() == 1)
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    good_pose.pose.position.y -= 0.055;
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);
    const auto preserved_curve = tc->latchedCurve();
    ASSERT_FALSE(preserved_curve.empty());

    // Periodic ~1 Hz setPlan() to the SAME goal and SAME forward gear (Review #1 Round 2 Finding 1)
    // must preserve the active valid latch, latchedCurve(), and redrawCount() == 1!
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);
    ASSERT_EQ(tc->latchedCurve().size(), preserved_curve.size());
    EXPECT_FLOAT_EQ(tc->latchedCurve().front().x, preserved_curve.front().x);
    EXPECT_FLOAT_EQ(tc->latchedCurve().front().y, preserved_curve.front().y);

    // Unrelated dynamic parameter update must NOT reset latch or redrawCount()
    const auto set_res = parent->set_parameters({
        rclcpp::Parameter("FollowPath.temperature", 0.35)});
    ASSERT_EQ(set_res.size(), 1u);
    EXPECT_TRUE(set_res[0].successful);
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // (5a) Review #2 Round 3 Finding 1: A periodic setPlan() in global_map frame coinciding
    //      with an 80 mm (+64 mm, -48 mm) localization shift in the local costmap frame must
    //      NOT reset the latch or redraw_count_ (which would happen if pending_new_plan_
    //      checked goal identity in the shifted local costmap frame with a 5 cm threshold).
    buffer->setUsingDedicatedThread(true);
    auto setGlobalToMapTransform = [&](double dx, double dy) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = rclcpp::Time(0);
        tf_msg.header.frame_id = "global_map";
        tf_msg.child_frame_id = "map";
        tf_msg.transform.translation.x = -dx;
        tf_msg.transform.translation.y = -dy;
        tf_msg.transform.rotation.w = 1.0;
        buffer->setTransform(tf_msg, "test_authority", true);
      };
    setGlobalToMapTransform(0.0, 0.0);

    nav_msgs::msg::Path global_plan = early_scenario.message();
    global_plan.header.frame_id = "global_map";
    for (auto & ps : global_plan.poses) {
      ps.header.frame_id = "global_map";
    }

    controller.reset();
    good_pose = pose({-0.55, -0.06, 0.0});
    controller.setPlan(global_plan);
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // Consume the 1 permitted redraw (redrawCount() == 1)
    good_pose.pose.position.y -= 0.055;
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // Coinciding event: periodic setPlan(global_plan) + 80 mm (+64 mm, -48 mm) TF shift
    setGlobalToMapTransform(0.064, -0.048);
    good_pose.pose.position.x += 0.064;
    good_pose.pose.position.y -= 0.048;
    controller.setPlan(global_plan);
    EXPECT_NO_THROW(controller.computeVelocityCommands(good_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);
    ASSERT_FALSE(tc->latchedCurve().empty());
    EXPECT_NEAR(tc->latchedCurve().back().x, goal.position.x + 0.064, 1e-4);
    EXPECT_NEAR(tc->latchedCurve().back().y, goal.position.y - 0.048, 1e-4);

    // Because redrawCount() == 1 was preserved across the coinciding setPlan() + 80 mm TF shift,
    // an additional > 45 mm tracking drift must throw InvalidPath (DRIFT_EXCEEDED).
    auto over_drift_pose = good_pose;
    over_drift_pose.pose.position.y -= 0.055;
    EXPECT_THROW(
      controller.computeVelocityCommands(over_drift_pose, speed, nullptr),
      nav2_core::InvalidPath);

    // (5b) Review #1 Round 3 Finding 3 & Review #2 Round 2: Under infeasible_policy = "fallback"
    //      and max_redraws = 0, once a latched curve is abandoned due to DRIFT_EXCEEDED,
    //      periodic setPlan() OR leaving and re-entering the approach circle on the same
    //      approach must NOT re-latch a second curve, whereas a terminal gear change or
    //      reset() clears state.
    setGlobalToMapTransform(0.0, 0.0);
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.infeasible_policy", "fallback"),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 0)})[0].successful);
    controller.reset();
    controller.setPlan(early_scenario.message());
    auto fb_pose = pose({-0.55, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(fb_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());

    // Perturb > 45 mm to trigger DRIFT_EXCEEDED fallback
    fb_pose.pose.position.y -= 0.065;
    EXPECT_NO_THROW(controller.computeVelocityCommands(fb_pose, speed, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Step outside approach_radius + hysteresis (goal_dist = 0.782 m > 0.70 m) and re-enter:
    // approach_escalated_ must remain sticky (no second capture solve on re-entry)
    auto outside_circle_pose = pose({-0.78, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(outside_circle_pose, speed, nullptr));
    EXPECT_FALSE(tc->isLatched());

    auto relatch_candidate_pose = pose({-0.55, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(relatch_candidate_pose, speed, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Periodic 1 Hz setPlan() to the same goal and same forward gear must still NOT re-latch
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(relatch_candidate_pose, speed, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Review #3 Round 5 (Hole A): Changing terminal gear on setPlan() (forward -> reverse to the
    // EXACT same goal [0, 0, 0]) WITHOUT controller.reset() under max_redraws = 0 must preserve
    // latch_consumed_ and NOT solve a second capture curve on the same goal!
    nav_msgs::msg::Path same_goal_rev_path = early_scenario.message();
    for (auto & ps : same_goal_rev_path.poses) {
      ps.pose.position.x = -ps.pose.position.x;
    }
    controller.setPlan(same_goal_rev_path);
    auto rev_pose = pose({0.55, 0.06, 0.0});
    geometry_msgs::msg::Twist rev_speed;
    rev_speed.linear.x = -0.10;
    EXPECT_NO_THROW(controller.computeVelocityCommands(rev_pose, rev_speed, nullptr));
    EXPECT_FALSE(tc->isLatched());

    // Under infeasible_policy = "abort" and max_redraws = 0, opposite-gear setPlan() to the
    // same goal while latched must escalate with GEAR_FLIP_ON_SAME_GOAL rather than re-latching!
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.infeasible_policy", "abort"),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 0)})[0].successful);
    controller.reset();
    controller.setPlan(early_scenario.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(fb_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());

    controller.setPlan(same_goal_rev_path);
    EXPECT_FALSE(tc->isLatched());
    try {
      controller.computeVelocityCommands(rev_pose, rev_speed, nullptr);
      FAIL() << "Expected GEAR_FLIP_ON_SAME_GOAL when opposite-gear plan arrives for same goal";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=GEAR_FLIP_ON_SAME_GOAL"), std::string::npos)
        << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // Review #3 Round 6 (minDistanceToCurve first-segment clamp & abort on local odom shifts):
    // 1) Backing away behind latched_curve_[0] by 7 cm (> 4.5 cm max_tracking_drift_) must
    //    measure true distance from latched_curve_[0] (t clamped >= 0.0) and abort with
    //    DRIFT_EXCEEDED rather than projecting onto a backward ray with 0 drift:
    controller.reset();
    controller.setPlan(early_scenario.message());
    auto latch_pose = pose({-0.55, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(latch_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    auto backed_pose = pose({-0.62, -0.06, 0.0});
    try {
      controller.computeVelocityCommands(backed_pose, speed, nullptr);
      FAIL() << "Expected DRIFT_EXCEEDED when backing 7 cm behind latched_curve_[0]";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // 2) An 8 cm map -> odom localization jump (> 4.5 cm max_tracking_drift_, <= 0.25 m
    //    goal_reset_pos_tol_) that shifts the transformed goal away from the odom robot pose
    //    under infeasible_policy = "abort" and max_redraws = 0 must abort with
    //    reason=LOCALIZATION_JUMP rather than masking the jump as DRIFT_EXCEEDED:
    setGlobalToMapTransform(0.0, 0.0);
    controller.reset();
    controller.setPlan(global_plan);
    EXPECT_NO_THROW(controller.computeVelocityCommands(latch_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    setGlobalToMapTransform(0.0, -0.08);
    try {
      controller.computeVelocityCommands(latch_pose, speed, nullptr);
      FAIL() << "Expected LOCALIZATION_JUMP on 8 cm map->odom goal shift while latched";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LOCALIZATION_JUMP"), std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // 3) A 0.30 m (+0.24 m, -0.18 m) map -> odom shift (> 0.25 m goal_reset_pos_tol_) while
    //    latched under infeasible_policy = "abort" and max_redraws = 0 must abort with
    //    reason=LOCAL_GOAL_SHIFT_EXCEEDED and consume the latch:
    setGlobalToMapTransform(0.0, 0.0);
    controller.reset();
    controller.setPlan(global_plan);
    EXPECT_NO_THROW(controller.computeVelocityCommands(latch_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    setGlobalToMapTransform(0.24, -0.18);
    auto shifted_latch_pose = latch_pose;
    shifted_latch_pose.pose.position.x += 0.24;
    shifted_latch_pose.pose.position.y -= 0.18;
    try {
      controller.computeVelocityCommands(shifted_latch_pose, speed, nullptr);
      FAIL() << "Expected LOCAL_GOAL_SHIFT_EXCEEDED on 0.30 m map->odom shift under abort";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LOCAL_GOAL_SHIFT_EXCEEDED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    setGlobalToMapTransform(0.0, 0.0);

    // After a task-boundary controller.reset(), setPlan(same_goal_rev_path) latches in reverse:
    controller.reset();
    controller.setPlan(same_goal_rev_path);
    EXPECT_NO_THROW(controller.computeVelocityCommands(rev_pose, rev_speed, nullptr));
    EXPECT_TRUE(tc->isLatched());

    // (5c) Review #2 Round 3 Finding 2: On a reverse capture approach, once the robot is 1.5 cm
    //      short of the goal (d_nose = -0.015 m > -0.02 m) and path pruning leaves a 1-point
    //      path (no segment from which to infer gear), repeated setPlan() calls (both full
    //      reverse plan and 1-point plan) + score() must preserve the reverse latch rather than
    //      misclassifying the approach as forward and discarding the capture curve.
    auto near_arrival_rev_pose = pose({0.015, 0.005, 0.0});
    controller.setPlan(same_goal_rev_path);
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(near_arrival_rev_pose, rev_speed, nullptr));
    EXPECT_TRUE(tc->isLatched());

    nav_msgs::msg::Path one_point_rev_plan;
    one_point_rev_plan.header = same_goal_rev_path.header;
    one_point_rev_plan.poses.push_back(same_goal_rev_path.poses.back());
    tc->onNewPlan(one_point_rev_plan);

    mppi::models::State rev_st;
    rev_st.reset(1, 20);
    rev_st.pose = near_arrival_rev_pose;
    mppi::models::Trajectories rev_tr;
    rev_tr.reset(1, 20);
    mppi::models::Path one_pt_mp = mppi::utils::toTensor(one_point_rev_plan);
    ASSERT_EQ(one_pt_mp.x.shape(0), 1u);
    xt::xtensor<float, 1> rev_costs = xt::zeros<float>({1});
    float dt = 0.05f;
    mppi::CriticData rev_data = {
      rev_st, rev_tr, one_pt_mp, goal, rev_costs, dt, false, nullptr,
      nullptr, std::nullopt, std::nullopt};
    EXPECT_NO_THROW(tc->score(rev_data));
    EXPECT_TRUE(tc->isLatched());

    // (5d) Review #1 Round 3 Finding 2 & Review #2 Round 3 Finding 4:
    //      When replan_signal publishes Bool(true) and the controller is then deactivated,
    //      a parameter update while deactivated must NOT reactivate replan_pub_; instead,
    //      retraction (Bool(false)) is deferred until controller.activate().
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.infeasible_policy", "replan_signal"),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 0)})[0].successful);
    grid(late_scenario);
    controller.reset();
    controller.setPlan(late_scenario.message());

    bool sub_got_true = false;
    bool sub_got_false = false;
    auto replan_sub = parent->create_subscription<std_msgs::msg::Bool>(
      "FollowPath/terminal_replan_requested",
      rclcpp::QoS(1).transient_local(),
      [&sub_got_true, &sub_got_false](const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
          sub_got_true = true;
        } else {
          sub_got_false = true;
        }
      });
    auto spinFor = [&](const bool & flag) {
        for (int i = 0; i < 20 && !flag; ++i) {
          rclcpp::spin_some(parent->get_node_base_interface());
          if (!flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
      };

    const auto overlap_bad_pose = pose({-0.48, -0.18, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_bad_pose, speed, nullptr));
    EXPECT_TRUE(tc->replanSignalSent());
    EXPECT_TRUE(tc->isReplanPublisherActivated());
    spinFor(sub_got_true);
    EXPECT_TRUE(sub_got_true);
    EXPECT_FALSE(sub_got_false);

    controller.deactivate();
    EXPECT_FALSE(tc->isReplanPublisherActivated());

    // Update a TerminalApproachCritic parameter while deactivated -> triggers resetLatch()
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_tracking_drift", 0.050)})[0]
      .successful);
    EXPECT_FALSE(tc->replanSignalSent());
    EXPECT_FALSE(tc->isReplanPublisherActivated());
    rclcpp::spin_some(parent->get_node_base_interface());
    EXPECT_FALSE(sub_got_false);

    // Reactivating the controller activates replan_pub_ and flushes the deferred Bool(false)
    controller.activate();
    EXPECT_TRUE(tc->isReplanPublisherActivated());
    spinFor(sub_got_false);
    EXPECT_TRUE(sub_got_false);

    controller.deactivate();
    controller.cleanup();
    // Null-guarded reset() after cleanup() must not dereference null parameters_handler_
    EXPECT_NO_THROW(controller.reset());
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
