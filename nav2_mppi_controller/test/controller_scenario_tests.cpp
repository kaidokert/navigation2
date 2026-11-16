// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "rcutils/logging.h"
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
  const auto & costs() const {return costs_;}
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
  float evaluateTerminalCriticOnlyCost()
  {
    if (terminal_critic_ == nullptr) {
      return 0.0f;
    }
    xt::xtensor<float, 1> tc_costs = xt::zeros<float>({costs_.shape(0)});
    mppi::CriticData tc_data = {
      state_, generated_trajectories_, path_, goal_,
      tc_costs, settings_.model_dt, false, nullptr, nullptr,
      std::nullopt, std::nullopt};
    terminal_critic_->score(tc_data);
    return xt::sum(tc_costs)();
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

    // m6_cusp_recorded, wp1_overlap_recorded, and m8_cusp_recorded are loaded directly from
    // the shared fixture file nav2_mppi_controller/test/fixtures/approach_atlas.json.

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
    const double ox = (scenario.id == "wp1_overlap_recorded") ?
      1.0 : controller_test::number(map["origin"][0]);
    const double oy = controller_test::number(map["origin"][1]);
    cm->getLayeredCostmap()->resizeMap(
      map["size"][0].as<unsigned int>(), map["size"][1].as<unsigned int>(),
      controller_test::number(map["resolution"]), ox, oy);
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
  // First-order speed/curvature lag approximation of selected plant parameters (tau_v, tau_steer,
  // wheel_speed_gain, steering_gain). Not a complete model of Gazebo bridge actuator quantization
  // or contact dynamics; recorded Gazebo state trajectories (m6_cusp_recorded,
  // wp1_overlap_recorded) are used for deterministic reproduction of live Gazebo transitions.
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
  ASSERT_EQ(atlas->cases.size(), 20u);
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
    if (original.id == "m8_full" || original.id == "m5_recorded" ||
      original.id == "m6_cusp_recorded" || original.id == "m8_cusp_recorded" ||
      original.id == "wp1_overlap_recorded")
    {
      continue;
    }
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
          "FollowPath.TerminalApproachCritic.min_turning_radius", rclcpp::ParameterValue(0.56));
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

    // Stateful 2-tick sequence on the same controller instance with stock_cutoff_r = 0.01:
    // tick 0 at (-0.30, 0.0, 0.0) followed by a 55 mm (> 45 mm) corner-cutting step at
    // (-0.25, 0.055, 0.0) must NOT self-abort with DRIFT_EXCEEDED while stock critics steer,
    // remaining in PREPARE ownership with isLatched() == false. Conversely, when stock critics
    // hand off at 0.35 m (TERMINAL_EXCLUSIVE ownership), the same 55 mm step from a latched
    // reference path at (-0.30, 0.0, 0.0) to (-0.25, 0.055, 0.0) aborts with DRIFT_EXCEEDED
    // and 0 redraws even when max_redraws = 1. Also verify STOCK_THRESHOLD_MISMATCH refusal
    // when PathFollowCritic and PathAlignCritic have unequal threshold_to_consider.
    {
      using Ownership = mppi::critics::GuidanceOwnership;
      auto parent_001 = node(true, 64, 0.558, 6, 0.01);
      parent_001->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent_001->declare_parameter(
        "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
      parent_001->declare_parameter(
        "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
      parent_001->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent_001->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
      mppi::ParametersHandler params_001(parent_001);
      params_001.start();
      auto buffer_001 = std::make_shared<tf2_ros::Buffer>(parent_001->get_clock());
      ScenarioPathHandler handler_001;
      handler_001.initialize(parent_001, "FollowPath", cm, buffer_001, &params_001);
      handler_001.setPath(scenario.message());
      ScenarioOptimizer opt_001;
      opt_001.configure(parent_001, cm, &params_001, true);
      opt_001.setSeededGaussianBank(42u);
      geometry_msgs::msg::Twist speed_001;
      speed_001.linear.x = 0.15;
      const auto goal_001 = scenario.message().poses.back().pose;
      auto p0 = pose({-0.30, 0.0, 0.0});
      auto lp0 = handler_001.transformPath(p0);
      EXPECT_NO_THROW(opt_001.evalControl(p0, speed_001, lp0, goal_001, nullptr));
      EXPECT_FALSE(opt_001.terminalCritic()->isLatched());
      EXPECT_FALSE(opt_001.terminalCritic()->isReferencePathLatched());
      EXPECT_EQ(opt_001.terminalCritic()->guidanceOwnership(), Ownership::PREPARE);
      auto p1 = pose({-0.25, 0.055, 0.0});
      auto lp1 = handler_001.transformPath(p1);
      EXPECT_NO_THROW(opt_001.evalControl(p1, speed_001, lp1, goal_001, nullptr));
      EXPECT_FALSE(opt_001.terminalCritic()->isLatched());
      EXPECT_EQ(opt_001.terminalCritic()->guidanceOwnership(), Ownership::PREPARE);
      EXPECT_EQ(opt_001.terminalCritic()->redrawCount(), 0);

      // Companion check when stock_cutoff_r = 0.35 (TERMINAL_EXCLUSIVE at d_goal <= 0.35 m):
      auto parent_035 = node(true, 64, 0.558, 6, 0.35);
      parent_035->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent_035->declare_parameter(
        "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
      parent_035->declare_parameter(
        "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
      parent_035->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent_035->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
      mppi::ParametersHandler params_035(parent_035);
      params_035.start();
      auto buffer_035 = std::make_shared<tf2_ros::Buffer>(parent_035->get_clock());
      ScenarioPathHandler handler_035;
      handler_035.initialize(parent_035, "FollowPath", cm, buffer_035, &params_035);
      handler_035.setPath(scenario.message());
      ScenarioOptimizer opt_035;
      opt_035.configure(parent_035, cm, &params_035, true);
      opt_035.setSeededGaussianBank(42u);
      lp0 = handler_035.transformPath(p0);
      EXPECT_NO_THROW(opt_035.evalControl(p0, speed_001, lp0, goal_001, nullptr));
      EXPECT_TRUE(opt_035.terminalCritic()->isLatched());
      EXPECT_TRUE(opt_035.terminalCritic()->isReferencePathLatched());
      EXPECT_TRUE(opt_035.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(opt_035.terminalCritic()->guidanceOwnership(), Ownership::TERMINAL_EXCLUSIVE);
      lp1 = handler_035.transformPath(p1);
      try {
        opt_035.evalControl(p1, speed_001, lp1, goal_001, nullptr);
        FAIL() << "Expected DRIFT_EXCEEDED on 55 mm step in TERMINAL_EXCLUSIVE ownership";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
      }
      EXPECT_FALSE(opt_035.terminalCritic()->isLatched());
      EXPECT_TRUE(opt_035.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(opt_035.terminalCritic()->redrawCount(), 0);

      // Unequal stock critic thresholds (PathFollowCritic=0.35, PathAlignCritic=0.01) while
      // TerminalApproachCritic is enabled with cutoff_radius=0.35 must immediately refuse
      // before motion with STOCK_THRESHOLD_MISMATCH:
      ASSERT_TRUE(
        parent_035->set_parameters({
          rclcpp::Parameter("FollowPath.PathAlignCritic.threshold_to_consider", 0.01)})[0]
        .successful);
      opt_035.resetCritics();
      handler_035.setPath(scenario.message());
      lp0 = handler_035.transformPath(p0);
      try {
        opt_035.evalControl(p0, speed_001, lp0, goal_001, nullptr);
        FAIL() << "Expected STOCK_THRESHOLD_MISMATCH when stock critic thresholds disagree";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=STOCK_THRESHOLD_MISMATCH"), std::string::npos)
          << ex.what();
      }
    }

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

  // 5. CriticManager preserves declared critic order in both loaded names and actual score()
  //    execution order when evalTrajectoriesScores() is invoked on real compiled critics
  //    (PathFollowCritic, PathAlignCritic, TerminalApproachCritic), producing non-zero real scores
  {
    struct RealOrderRecordingCritic : public mppi::critics::CriticFunction
    {
      RealOrderRecordingCritic(
        std::string critic_id,
        std::unique_ptr<mppi::critics::CriticFunction> wrapped,
        std::vector<std::string> * out_log,
        std::vector<float> * out_deltas)
      : id(std::move(critic_id)), inner(std::move(wrapped)), log(out_log), deltas(out_deltas) {}
      void initialize() override
      {
        inner->on_configure(parent_, parent_name_, name_, costmap_ros_, parameters_handler_);
      }
      void score(mppi::CriticData & data) override
      {
        if (log != nullptr) {
          log->push_back(id);
        }
        const float before = xt::sum(data.costs)();
        inner->score(data);
        const float after = xt::sum(data.costs)();
        if (deltas != nullptr) {
          deltas->push_back(after - before);
        }
      }
      std::string id;
      std::unique_ptr<mppi::critics::CriticFunction> inner;
      std::vector<std::string> * log{nullptr};
      std::vector<float> * deltas{nullptr};
    };
    struct StaticCriticManager : mppi::CriticManager
    {
      std::vector<std::string> executed_order;
      std::vector<float> cost_deltas;
      void loadCritics() override
      {
        auto & critics_vec = this->*CriticAccess::critics();
        critics_vec.clear();
        for (const auto & cname : this->*CriticAccess::names()) {
          std::unique_ptr<mppi::critics::CriticFunction> real_critic;
          if (cname == "PathFollowCritic") {
            real_critic = std::make_unique<mppi::critics::PathFollowCritic>();
          } else if (cname == "PathAlignCritic") {
            real_critic = std::make_unique<mppi::critics::PathAlignCritic>();
          } else {
            real_critic = std::make_unique<mppi::critics::TerminalApproachCritic>();
          }
          auto wrapper = std::make_unique<RealOrderRecordingCritic>(
            cname, std::move(real_critic), &executed_order, &cost_deltas);
          wrapper->on_configure(
            parent_, name_, name_ + "." + cname, costmap_ros_, parameters_handler_);
          critics_vec.push_back(std::move(wrapper));
        }
      }
    };
    rclcpp::NodeOptions plugin_opts;
    plugin_opts.parameter_overrides({
        rclcpp::Parameter(
        "FollowPath.critics",
        std::vector<std::string>{
          "PathFollowCritic", "PathAlignCritic", "TerminalApproachCritic"}),
        rclcpp::Parameter("FollowPath.PathAlignCritic.threshold_to_consider", 0.35),
        rclcpp::Parameter("FollowPath.PathAlignCritic.offset_from_furthest", 1),
        rclcpp::Parameter("FollowPath.PathAlignCritic.cost_weight", 28.0),
        rclcpp::Parameter("FollowPath.PathAlignCritic.use_path_orientations", true),
        rclcpp::Parameter("FollowPath.PathFollowCritic.threshold_to_consider", 0.35),
        rclcpp::Parameter("FollowPath.PathFollowCritic.offset_from_furthest", 6),
        rclcpp::Parameter("FollowPath.PathFollowCritic.cost_weight", 66.05),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.enabled", true),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.approach_radius", 0.65),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.cutoff_radius", 0.35)});
    auto parent = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "scenario_critic_mgr", plugin_opts);
    mppi::ParametersHandler params(parent);
    params.start();
    StaticCriticManager mgr;
    mgr.on_configure(parent, "FollowPath", cm, &params);
    const auto & loaded_names = mgr.*CriticAccess::names();
    ASSERT_EQ(loaded_names.size(), 3u);
    EXPECT_EQ(loaded_names[0], "PathFollowCritic");
    EXPECT_EQ(loaded_names[1], "PathAlignCritic");
    EXPECT_EQ(loaded_names[2], "TerminalApproachCritic");

    // First evaluate in OVERLAP_SHARED (d_goal = 0.50 m in (0.35, 0.65], displaced y = -0.05 m)
    // where all three real critics contribute > 0 cost in declared order:
    mppi::models::State st;
    st.reset(2, 10);
    st.pose = pose({-0.50, -0.05, 0.0});
    st.speed.linear.x = 0.15;
    mppi::models::Trajectories tr;
    tr.reset(2, 10);
    for (size_t j = 0; j < 10; ++j) {
      const float x = -0.50f + 0.05f * static_cast<float>(j + 1);
      tr.x(0, j) = x;
      tr.y(0, j) = -0.02f;
      tr.yaws(0, j) = 0.05f;
      tr.x(1, j) = x;
      tr.y(1, j) = -0.15f;
      tr.yaws(1, j) = -0.35f;
    }
    const auto & early_sc = atlas->at("early_lateral_capture");
    auto mp = mppi::utils::toTensor(early_sc.message());
    auto g = early_sc.message().poses.back().pose;
    xt::xtensor<float, 1> costs = xt::zeros<float>({2});
    float dt = 0.05f;
    mppi::CriticData data = {
      st, tr, mp, g, costs, dt, false, nullptr, nullptr, std::nullopt, std::nullopt};
    mgr.evalTrajectoriesScores(data);
    ASSERT_EQ(mgr.executed_order.size(), 3u);
    EXPECT_EQ(mgr.executed_order[0], "PathFollowCritic");
    EXPECT_EQ(mgr.executed_order[1], "PathAlignCritic");
    EXPECT_EQ(mgr.executed_order[2], "TerminalApproachCritic");
    ASSERT_EQ(mgr.cost_deltas.size(), 3u);
    EXPECT_GT(mgr.cost_deltas[0], 0.0f);  // PathFollowCritic active at 0.50 m > 0.35 m
    EXPECT_GT(mgr.cost_deltas[1], 0.0f);  // PathAlignCritic active at 0.50 m > 0.35 m
    EXPECT_GT(mgr.cost_deltas[2], 0.0f);  // TerminalApproachCritic active in OVERLAP_SHARED
    EXPECT_LT(costs(0), costs(1));
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

    // Restore default min_turning_radius (0.56 m) and reset for next block
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.min_turning_radius", 0.56)})[0]
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

  // 2. Short single-gear terminal path handoff poses (short_cusp_exit: stress and
  //    short_terminal_attempt_1..3, plus Review #1's worked example [-0.20, 0.05, 0.0] on
  //    early_lateral_capture) and a true 2-segment gear-reversal sub-min_approach cusp scenario
  //    (sub_min_cusp: reverse -> cusp vertex at index 4 (-0.12, 0.0, 0.0) -> 12 cm forward stub
  //    to [0, 0, 0]) latch the reference path and complete in closed loop inside the 0.35 m
  //    stock-critic cutoff with 0 aborts under Ackermann plant, falsifiably abort when
  //    short-approach guards are disabled, and reject kinematically unreachable heading/lateral
  //    errors.
  {
    Scenario short_exit = atlas->at("short_cusp_exit");
    short_exit.starts["short_terminal_attempt_1"] = {-0.248, 0.430, -0.5624};
    short_exit.starts["short_terminal_attempt_2"] = {-0.197, 0.404, -0.4995};
    short_exit.starts["short_terminal_attempt_3"] = {-0.200, 0.398, -0.5112};
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
      {-0.12, 0.0, 0.0},      // Cusp vertex at index 4 (findFirstPathInversion returns index 5)
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
    // short single-gear terminal legs, the < 16 cm 2-segment post-cusp stub with -0.34 rad yaw
    // error, and Review #1's worked example abort with ENTRY_INFEASIBLE
    for (const std::string & fail_key : {
        std::string("short_terminal_attempt_1"),
        std::string("short_terminal_attempt_2"),
        std::string("short_terminal_attempt_3"),
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
        std::string("short_terminal_attempt_1"),
        std::string("short_terminal_attempt_2"),
        std::string("short_terminal_attempt_3"),
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

    // Short-approach reference_path post-latch drift envelope consistency & ratcheting
    // (single-gear straight_short at kinematically reachable pose (-0.22, 0.062, -0.20)):
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
      auto latch_p = pose({-0.22, 0.062, -0.20});
      auto local_p = handler.transformPath(latch_p);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.05;
      EXPECT_NO_THROW(optimizer.evalControl(latch_p, speed, local_p, goal_msg, goal_checker.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      ASSERT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());

      // +38 mm post-latch excursion (total cross_track = 0.100 m <= 0.062 + 0.045 = 0.107 m):
      auto coast_p = pose({-0.21, 0.100, -0.18});
      local_p = handler.transformPath(coast_p);
      EXPECT_NO_THROW(optimizer.evalControl(coast_p, speed, local_p, goal_msg, goal_checker.get()));
      EXPECT_TRUE(optimizer.terminalCritic()->isLatched());

      // +55 mm post-latch excursion (total cross_track = 0.117 m > 0.107 m) -> DRIFT_EXCEEDED:
      auto over_p = pose({-0.21, 0.117, -0.18});
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
      auto converged_p = pose({-0.19, 0.010, -0.05});
      local_p = handler.transformPath(converged_p);
      EXPECT_NO_THROW(
        optimizer.evalControl(converged_p, speed, local_p, goal_msg, goal_checker.get()));
      auto rebound_p = pose({-0.19, 0.070, -0.05});
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

    // Section 1: Separate lateral tracking error (DRIFT_EXCEEDED) from longitudinal pre-start
    // rollback (ROLLBACK_EXCEEDED):
    // (1a) Unexplained 70 mm backward retreat (> 45 mm max_tracking_drift_) in established
    //      same-gear motion (speed.linear.x = +0.10 in forward gear +1) must abort with
    //      reason=ROLLBACK_EXCEEDED:
    controller.reset();
    controller.setPlan(early_scenario.message());
    auto latch_pose = pose({-0.55, -0.06, 0.0});
    EXPECT_NO_THROW(controller.computeVelocityCommands(latch_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    auto backed_pose = pose({-0.62, -0.06, 0.0});
    try {
      controller.computeVelocityCommands(backed_pose, speed, nullptr);
      FAIL() << "Expected ROLLBACK_EXCEEDED when retreating 7 cm in established forward motion";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=ROLLBACK_EXCEEDED"), std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());

    // (1b) State-aware cusp braking transient: when latching with opposite-gear braking velocity
    //      (speed.linear.x = -0.12 while gear = +1), a 70 mm longitudinal coast behind
    //      latched_curve_[0] with zero lateral departure survives (EXPECT_NO_THROW), whereas a
    //      65 mm lateral departure during that coast aborts with DRIFT_EXCEEDED, and an excessive
    //      150 mm (> 135 mm) rollback aborts with ROLLBACK_EXCEEDED:
    {
      geometry_msgs::msg::Twist braking_speed;
      braking_speed.linear.x = -0.12;
      controller.reset();
      controller.setPlan(early_scenario.message());
      EXPECT_NO_THROW(controller.computeVelocityCommands(latch_pose, braking_speed, nullptr));
      EXPECT_TRUE(tc->isLatched());
      braking_speed.linear.x = -0.06;
      EXPECT_NO_THROW(controller.computeVelocityCommands(backed_pose, braking_speed, nullptr));
      EXPECT_TRUE(tc->isLatched());
      EXPECT_GT(tc->lastPreStartRollbackM(), 0.055);
      EXPECT_LT(tc->lastCurveLateralErr(), 0.015);

      auto lat_depart_pose = pose({-0.62, -0.130, 0.0});
      try {
        controller.computeVelocityCommands(lat_depart_pose, braking_speed, nullptr);
        FAIL() << "Expected DRIFT_EXCEEDED on 65 mm lateral departure during cusp braking coast";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
      }

      controller.reset();
      controller.setPlan(early_scenario.message());
      braking_speed.linear.x = -0.12;
      auto closer_latch_pose = pose({-0.48, -0.06, 0.0});
      EXPECT_NO_THROW(
        controller.computeVelocityCommands(closer_latch_pose, braking_speed, nullptr));
      auto excessive_rollback_pose = pose({-0.63, -0.06, 0.0});
      try {
        controller.computeVelocityCommands(excessive_rollback_pose, braking_speed, nullptr);
        FAIL() << "Expected ROLLBACK_EXCEEDED on 150 mm (> 135 mm) cusp rollback";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=ROLLBACK_EXCEEDED"), std::string::npos) << ex.what();
      }
    }

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

TEST_F(ControllerScenarios, TerminalApproachOptionA11ReviewPunchlistRegressions)
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

  // (1) Section 1, Sections A/B & Section G (Items 4, 6, 8, 16, 24, 26, 30, 47):
  //     - Recorded M6 post-cusp longitudinal braking coast sequence (m6_cusp_recorded, extracted
  //       from 20261114_fa_critic_v3/raw/arm2_candidate_a11_fa_critic/suite2_matrix/
  //       nav2_matrix_6_reverse_right_90_course/nav2_course_summary.json, episodes[0], Repeat 1,
  //       k=316..327, elapsed_s=16.920..17.470 s, wall_s=1794709652.842..1794709653.393)
  //       must survive the demonstrated ~75 mm opposite-gear braking coast behind curve[0]
  //       (lastPreStartRollbackM() > 0.015 m on every tick and > 0.070 m at coast apex k=327)
  //       without throwing DRIFT_EXCEEDED and with 0 redraws even when max_redraws = 1.
  //     - Recorded M8 post-cusp longitudinal braking coast sequence (m8_cusp_recorded, extracted
  //       from 20261114_fa_critic_v3/raw/arm2_candidate_a11_fa_critic/suite2_matrix/
  //       nav2_matrix_8_lateral_shift_right_course/nav2_course_summary.json, episodes[0], Repeat 1,
  //       k=140..153, elapsed_s=8.235..8.885 s, wall_s=1794709904.710..1794709905.360)
  //       must survive the demonstrated ~63 mm opposite-gear braking coast behind curve[0]
  //       (lastPreStartRollbackM() > 0.020 m on every tick and > 0.060 m at coast apex k=148)
  //       with 0 redraws when max_redraws = 1.
  //     - Post-reversal renewed-retreat contract (Sections A & B, 100 -> 20 -> 35 -> 80 mm):
  //       coasting to 100 mm behind curve[0], establishing reversal motion while still 20 mm
  //       behind curve[0] (inCuspBrakingTransient() == true, isReversalMotionEstablished() == true,
  //       minPostReversalRollbackM() == 0.020 m), surviving an allowed 15 mm fluctuation
  //       (20 -> 35 mm) and recovering onto the curve, while rejecting a 60 mm renewed retreat
  //       (20 -> 80 mm, below both 100+25=125 mm and the 135 mm ceiling) with ROLLBACK_EXCEEDED.
  //     - Closed-loop simulated-plant M6 cusp traversal entering the cusp at 0.22..0.28 m/s with
  //       motor_tau_s = 0.176 s (Item 47), naturally crossing the cusp via PathHandler and
  //       producing a 5..9 cm post-cusp braking coast behind curve[0] before reversing to the goal.
  {
    const auto & m6_base = atlas->at("m6_cusp_recorded");
    // Exact recorded states from 20261114_fa_critic_v3 M6 Repeat 1 (episodes[0], k=316..327,
    // elapsed_s=16.920..17.470s, wall_s=1794709652.842..1794709653.393):
    const std::vector<std::pair<Pose, std::pair<double, double>>> m6_coast_states = {
      {{-0.944200, -0.758500, 1.923700}, {0.1610, -0.2412}},   // k=316, t=16.920s
      {{-0.945300, -0.752000, 1.913900}, {0.1213, -0.2824}},   // k=317, t=16.970s
      {{-0.949800, -0.741300, 1.897100}, {0.0903, -0.2779}},   // k=318, t=17.020s
      {{-0.952200, -0.735100, 1.888200}, {0.0738, -0.2269}},   // k=319, t=17.070s
      {{-0.955100, -0.725200, 1.869400}, {0.0627, -0.1929}},   // k=320, t=17.120s
      {{-0.957000, -0.718300, 1.859500}, {0.0500, -0.1539}},   // k=321, t=17.170s
      {{-0.957700, -0.712600, 1.841100}, {0.0300, -0.0910}},   // k=322, t=17.220s
      {{-0.958700, -0.706900, 1.832400}, {-0.0001, -0.0005}},  // k=323, t=17.270s
      {{-0.958500, -0.707000, 1.821600}, {-0.0327, -0.0660}},  // k=324, t=17.320s
      {{-0.959300, -0.703600, 1.817200}, {-0.0630, -0.1297}},  // k=325, t=17.370s
      {{-0.959900, -0.702900, 1.807200}, {-0.0864, -0.1818}},  // k=326, t=17.420s
      {{-0.961000, -0.699100, 1.803100}, {-0.1011, -0.2006}}};  // k=327, t=17.470s

    for (const bool mirror : {false, true}) {
      for (const bool reverse_dir : {false, true}) {
        SCOPED_TRACE(
          std::string("m6_cusp_recorded/") + (mirror ? "mirror/" : "nom/") +
          (reverse_dir ? "rev_dir" : "fwd_dir"));
        const Scenario sc = m6_base.variant(mirror, reverse_dir);
        grid(sc);
        auto parent = node(true, 500, 0.325, 7, 0.35);
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.infeasible_policy",
          rclcpp::ParameterValue(std::string("abort")));
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
        auto gc = makeGoalChecker(parent);
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
        // First evaluate a pre-cusp tick on the full 2-segment path at pre_cusp_fast (k=306) so
        // score() records the pre-cusp gear before PathHandler prunes across the cusp vertex.
        {
          auto pre_cusp_actual = pose(sc.starts.at("pre_cusp_fast"));
          auto pre_cusp_local = handler.transformPath(pre_cusp_actual);
          ASSERT_NE(handler.inversion(), 0u);
          geometry_msgs::msg::Twist pre_speed;
          pre_speed.linear.x = (reverse_dir ? -1.0 : 1.0) * 0.2235;
          pre_speed.angular.z = (mirror ? -1.0 : 1.0) * (-0.3755);
          EXPECT_NO_THROW(
            optimizer.evalControl(
              pre_cusp_actual, pre_speed, pre_cusp_local, goal_msg, gc.get()));
          EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
        }
        // Advance ScenarioPathHandler onto the post-cusp terminal segment (inversion == 0)
        auto cusp_vertex_pose = pose(sc.path[7]);
        while (handler.inversion() != 0u) {
          (void)handler.transformPath(cusp_vertex_pose);
        }
        ASSERT_EQ(handler.inversion(), 0u);

        for (size_t idx = 0u; idx < m6_coast_states.size(); ++idx) {
          Pose p = m6_coast_states[idx].first;
          if (mirror) {
            p[1] = -p[1];
            p[2] = -p[2];
          }
          p[2] = controller_test::wrap(p[2] + (reverse_dir ? M_PI : 0.0));
          auto actual = pose(p);
          auto local_path = handler.transformPath(actual);
          geometry_msgs::msg::Twist speed;
          speed.linear.x = (reverse_dir ? -1.0 : 1.0) * m6_coast_states[idx].second.first;
          speed.angular.z = (mirror ? -1.0 : 1.0) * m6_coast_states[idx].second.second;
          EXPECT_NO_THROW(
            optimizer.evalControl(actual, speed, local_path, goal_msg, gc.get()))
            << "M6 post-cusp braking coast aborted at step " << idx;
          EXPECT_TRUE(optimizer.terminalCritic()->isLatched()) << "step " << idx;
          EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed()) << "step " << idx;
          EXPECT_TRUE(optimizer.terminalCritic()->inCuspBrakingTransient()) << "step " << idx;
          // Every recorded coast pose k=316..327 lies behind curve[0] (> 15 mm rollback)
          EXPECT_GT(optimizer.terminalCritic()->lastPreStartRollbackM(), 0.015) << "step " << idx;
          EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0) << "step " << idx;
        }
        EXPECT_GT(optimizer.terminalCritic()->lastPreStartRollbackM(), 0.070);

        // Genuine lateral departure (70 mm perpendicular to post-cusp segment) during cusp coast
        // must still abort with reason=DRIFT_EXCEEDED and 0 redraws (even with max_redraws = 1):
        {
          Pose lat_p = m6_coast_states.back().first;
          lat_p[0] -= 0.070;
          if (mirror) {
            lat_p[1] = -lat_p[1];
            lat_p[2] = -lat_p[2];
          }
          lat_p[2] = controller_test::wrap(lat_p[2] + (reverse_dir ? M_PI : 0.0));
          auto actual = pose(lat_p);
          auto local_path = handler.transformPath(actual);
          geometry_msgs::msg::Twist speed;
          speed.linear.x = (reverse_dir ? -1.0 : 1.0) * m6_coast_states.back().second.first;
          try {
            optimizer.evalControl(actual, speed, local_path, goal_msg, gc.get());
            FAIL() << "Expected DRIFT_EXCEEDED on genuine 70 mm lateral departure on M6";
          } catch (const nav2_core::InvalidPath & ex) {
            EXPECT_NE(
              std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
          }
          EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
          EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
        }

        // Sections A & B: Post-reversal renewed-retreat contract (100 -> 20 -> 35 -> 80 mm)
        // while still behind latched_curve_[0] = sc.path[8] (inCuspBrakingTransient() == true):
        {
          // After PathHandler prunes across the cusp (inversion_locale == 8), the post-cusp
          // segment is [sc.path[8], sc.path[9]] = [(-0.951166, -0.812774, 1.633401),
          // (-0.950000, -0.850000, 1.570796)] with unit tangent (ux0, uy0) = (0.031307, -0.999510).
          // A pose at rollback_m behind curve[0] is curve[0] - rollback_m * (ux0, uy0).
          auto makePreStartPose = [&](double rollback_m) -> Pose {
              constexpr double ux0 = 0.031307154;
              constexpr double uy0 = -0.999509811;
              Pose p = {
                -0.951166 - ux0 * rollback_m,
                -0.812774 - uy0 * rollback_m,
                1.633401};
              if (mirror) {
                p[1] = -p[1];
                p[2] = -p[2];
              }
              p[2] = controller_test::wrap(p[2] + (reverse_dir ? M_PI : 0.0));
              return p;
            };

          auto runToSmallRetreat035 = [&]() {
              handler.setPath(sc.message());
              optimizer.resetCritics();
              // Pre-cusp tick so prev_logged_gear_ records the pre-cusp gear
              auto pre_p = pose(sc.starts.at("pre_cusp_fast"));
              auto pre_lp = handler.transformPath(pre_p);
              geometry_msgs::msg::Twist pre_sp;
              pre_sp.linear.x = (reverse_dir ? -1.0 : 1.0) * 0.22;
              EXPECT_NO_THROW(optimizer.evalControl(pre_p, pre_sp, pre_lp, goal_msg, gc.get()));
              auto cusp_v = pose(sc.path[7]);
              while (handler.inversion() != 0u) {
                (void)handler.transformPath(cusp_v);
              }
              // Step 1: Coast to 100 mm behind curve[0] in pre-cusp gear (vx = +0.18 -> +0.02 m/s)
              for (const auto & [rb_m, vx_val] :
                std::vector<std::pair<double, double>>{{0.020, 0.18}, {0.100, 0.02}})
              {
                auto act = pose(makePreStartPose(rb_m));
                auto lp = handler.transformPath(act);
                geometry_msgs::msg::Twist sp;
                sp.linear.x = (reverse_dir ? -1.0 : 1.0) * vx_val;
                EXPECT_NO_THROW(optimizer.evalControl(act, sp, lp, goal_msg, gc.get()));
              }
              EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
              EXPECT_TRUE(optimizer.terminalCritic()->inCuspBrakingTransient());
              EXPECT_FALSE(optimizer.terminalCritic()->isReversalMotionEstablished());
              EXPECT_NEAR(optimizer.terminalCritic()->lastPreStartRollbackM(), 0.100, 2e-3);

              // Step 2: Establish reversal motion (|vx| = 0.06 m/s >= 0.015 m/s in post-cusp gear)
              // while STILL 20 mm behind curve[0] (along_track_s == -0.020 < 0.01 m, so
              // inCuspBrakingTransient() remains true!):
              {
                auto act_020 = pose(makePreStartPose(0.020));
                auto lp_020 = handler.transformPath(act_020);
                geometry_msgs::msg::Twist rev_sp;
                rev_sp.linear.x = (reverse_dir ? -1.0 : 1.0) * (-0.06);
                EXPECT_NO_THROW(optimizer.evalControl(act_020, rev_sp, lp_020, goal_msg, gc.get()));
                EXPECT_TRUE(optimizer.terminalCritic()->inCuspBrakingTransient());
                EXPECT_TRUE(optimizer.terminalCritic()->isReversalMotionEstablished());
                EXPECT_NEAR(optimizer.terminalCritic()->minPostReversalRollbackM(), 0.020, 2e-3);
              }

              // Step 3: Allowed small 15 mm fluctuation (20 mm -> 35 mm <= 20 + 25 = 45 mm)
              // must succeed without replenishing minPostReversalRollbackM() (remains 20 mm):
              {
                auto act_035 = pose(makePreStartPose(0.035));
                auto lp_035 = handler.transformPath(act_035);
                geometry_msgs::msg::Twist rev_sp;
                rev_sp.linear.x = (reverse_dir ? -1.0 : 1.0) * (-0.04);
                EXPECT_NO_THROW(optimizer.evalControl(act_035, rev_sp, lp_035, goal_msg, gc.get()));
                EXPECT_TRUE(optimizer.terminalCritic()->inCuspBrakingTransient());
                EXPECT_TRUE(optimizer.terminalCritic()->isReversalMotionEstablished());
                EXPECT_NEAR(optimizer.terminalCritic()->minPostReversalRollbackM(), 0.020, 2e-3);
              }
            };

          // Case A: Successful recovery from the allowed 35 mm fluctuation onto the post-cusp
          // curve (20 mm ahead of curve[0], along_track_s = 0.020 m >= 0.01 m) clears transient:
          runToSmallRetreat035();
          {
            auto act_recover = pose(makePreStartPose(-0.020));
            auto lp_recover = handler.transformPath(act_recover);
            geometry_msgs::msg::Twist rev_sp;
            rev_sp.linear.x = (reverse_dir ? -1.0 : 1.0) * (-0.08);
            EXPECT_NO_THROW(
              optimizer.evalControl(act_recover, rev_sp, lp_recover, goal_msg, gc.get()));
            EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
            EXPECT_FALSE(optimizer.terminalCritic()->inCuspBrakingTransient());
            EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
          }

          // Case B: From the same post-reversal state (100 -> 20 -> 35 mm, with
          // inCuspBrakingTransient() == true and minPostReversalRollbackM() == 20 mm),
          // a renewed retreat to 80 mm behind curve[0] (60 mm retreat > 25 mm allowance,
          // even though 80 mm < 100 + 25 = 125 mm and 80 mm < 135 mm kMaxCuspBrakingRollbackM)
          // MUST fail with reason=ROLLBACK_EXCEEDED:
          runToSmallRetreat035();
          ASSERT_TRUE(optimizer.terminalCritic()->inCuspBrakingTransient());
          ASSERT_TRUE(optimizer.terminalCritic()->isReversalMotionEstablished());
          {
            auto act_080 = pose(makePreStartPose(0.080));
            auto lp_080 = handler.transformPath(act_080);
            geometry_msgs::msg::Twist rev_sp;
            rev_sp.linear.x = (reverse_dir ? -1.0 : 1.0) * (-0.04);
            try {
              optimizer.evalControl(act_080, rev_sp, lp_080, goal_msg, gc.get());
              FAIL() << "Expected ROLLBACK_EXCEEDED on 100 -> 20 -> 80 mm renewed retreat";
            } catch (const nav2_core::InvalidPath & ex) {
              EXPECT_NE(
                std::string(ex.what()).find("reason=ROLLBACK_EXCEEDED"), std::string::npos)
                << ex.what();
            }
            EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
            EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
          }
        }
      }
    }

    // Recorded M8 post-cusp longitudinal braking coast sequence (m8_cusp_recorded, extracted
    // from 20261114_fa_critic_v3/raw/arm2_candidate_a11_fa_critic/suite2_matrix/
    // nav2_matrix_8_lateral_shift_right_course/nav2_course_summary.json, episodes[0], Repeat 1,
    // k=140..153, elapsed_s=8.235..8.885 s, wall_s=1794709904.710..1794709905.360):
    {
      const auto & m8_cusp_sc = atlas->at("m8_cusp_recorded");
      const std::vector<std::pair<Pose, std::pair<double, double>>> m8_coast_states = {
        {{-0.233300, 0.463600, -0.668200}, {-0.1192, 0.3544}},  // k=140, t=8.235s
        {{-0.240200, 0.469300, -0.649900}, {-0.0955, 0.2920}},  // k=141, t=8.285s
        {{-0.244700, 0.473000, -0.640500}, {-0.0663, 0.2041}},  // k=142, t=8.335s
        {{-0.249300, 0.476900, -0.625800}, {-0.0182, 0.0560}},  // k=143, t=8.385s
        {{-0.252700, 0.479600, -0.618400}, {0.0316, 0.0600}},   // k=144, t=8.435s
        {{-0.258800, 0.482900, -0.601800}, {0.0735, 0.1033}},   // k=145, t=8.485s
        {{-0.263200, 0.485500, -0.594100}, {0.0994, 0.1619}},   // k=146, t=8.535s
        {{-0.263600, 0.485300, -0.586300}, {0.1123, 0.2033}},   // k=147, t=8.585s
        {{-0.266000, 0.486900, -0.584300}, {0.1230, 0.2138}},   // k=148, t=8.635s
        {{-0.259200, 0.481400, -0.587800}, {0.1247, 0.2166}},   // k=149, t=8.685s
        {{-0.258900, 0.480900, -0.590900}, {0.1269, 0.2150}},   // k=150, t=8.735s
        {{-0.251200, 0.473300, -0.593600}, {0.1271, 0.2145}},   // k=151, t=8.785s
        {{-0.250800, 0.471800, -0.595400}, {0.1198, 0.2227}},   // k=152, t=8.835s
        {{-0.241500, 0.463300, -0.598200}, {0.1140, 0.2325}}};  // k=153, t=8.885s

      grid(m8_cusp_sc);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
      auto gc = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(m8_cusp_sc.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_msg = m8_cusp_sc.message().poses.back().pose;
      // Evaluate pre-cusp tick k=130 on the 2-segment path first
      {
        auto pre_actual = pose(m8_cusp_sc.starts.at("pre_cusp_fast"));
        auto pre_lp = handler.transformPath(pre_actual);
        ASSERT_NE(handler.inversion(), 0u);
        geometry_msgs::msg::Twist pre_sp;
        pre_sp.linear.x = -0.1819;
        pre_sp.angular.z = 0.4785;
        EXPECT_NO_THROW(optimizer.evalControl(pre_actual, pre_sp, pre_lp, goal_msg, gc.get()));
        EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
      }
      auto cusp_vertex_pose = pose(m8_cusp_sc.path[6]);
      while (handler.inversion() != 0u) {
        (void)handler.transformPath(cusp_vertex_pose);
      }
      ASSERT_EQ(handler.inversion(), 0u);

      double max_m8_rb = 0.0;
      for (size_t idx = 0u; idx < m8_coast_states.size(); ++idx) {
        auto actual = pose(m8_coast_states[idx].first);
        auto local_path = handler.transformPath(actual);
        geometry_msgs::msg::Twist speed;
        speed.linear.x = m8_coast_states[idx].second.first;
        speed.angular.z = m8_coast_states[idx].second.second;
        EXPECT_NO_THROW(
          optimizer.evalControl(actual, speed, local_path, goal_msg, gc.get()))
          << "M8 post-cusp braking coast aborted at step " << idx;
        EXPECT_TRUE(optimizer.terminalCritic()->isLatched()) << "step " << idx;
        EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed()) << "step " << idx;
        EXPECT_GT(optimizer.terminalCritic()->lastPreStartRollbackM(), 0.020) << "step " << idx;
        max_m8_rb = std::max(max_m8_rb, optimizer.terminalCritic()->lastPreStartRollbackM());
        EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0) << "step " << idx;
      }
      EXPECT_GT(max_m8_rb, 0.060);
    }

    // Item 47: True closed-loop simulated-plant M6 cusp traversal entering the cusp at
    // 0.22..0.28 m/s with motor_tau_s = 0.176 s:
    // starts on the pre-cusp segment (m6_base.path[6], 5 cm before the cusp vertex m6_base.path[7])
    // at vx = 0.25 m/s, steps continuously through stepAckermannPose() + handler.transformPath()
    // without teleporting, naturally crosses the cusp at |vx| in [0.22, 0.28] m/s, produces a
    // 5..9 cm physical braking coast behind latched_curve_[0], and reaches the goal with 0 redraws.
    {
      grid(m6_base);
      auto parent = node(true, 500, 0.325, 7, 0.35);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy",
        rclcpp::ParameterValue(std::string("abort")));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
      auto gc = makeGoalChecker(parent);
      mppi::ParametersHandler params(parent);
      params.start();
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioPathHandler handler;
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(m6_base.message());
      ScenarioOptimizer optimizer;
      optimizer.configure(parent, cm, &params, true);
      optimizer.setSeededGaussianBank(42u);

      const auto goal_msg = m6_base.message().poses.back().pose;
      // Start 1 tick pre-cusp along the recorded approach (between k=310 and k=311,
      // elapsed_s=16.645s, wall_s=1794709652.567) so tick 0 is pre-cusp
      // (HANDOFF_ALONG_TRACK_BLOCKED) and tick 1 naturally crosses the cusp at vx in [0.22, 0.28].
      auto actual = pose({-0.9285, -0.8065, 2.0268});
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.27;
      speed.angular.z = -0.32;
      AckermannPlantState plant{speed.linear.x, speed.angular.z / speed.linear.x};
      bool saw_pre_cusp = false;
      bool saw_post_cusp = false;
      double cusp_entry_speed = 0.0;
      double max_rollback_observed = 0.0;
      bool reached = false;
      for (size_t tick = 0u; tick < 180u; ++tick) {
        if (saw_post_cusp && gc->isGoalReached(actual.pose, goal_msg, speed)) {
          reached = true;
          break;
        }
        const bool was_pre_cusp = (handler.inversion() != 0u);
        auto local_path = handler.transformPath(actual);
        if (handler.inversion() != 0u) {
          saw_pre_cusp = true;
        } else if (was_pre_cusp) {
          saw_post_cusp = true;
          cusp_entry_speed = speed.linear.x;
        }
        geometry_msgs::msg::TwistStamped cmd;
        ASSERT_NO_THROW(
          cmd = optimizer.evalControl(actual, speed, local_path, goal_msg, gc.get()))
          << "Closed-loop M6 cusp aborted at tick " << tick;
        if (saw_post_cusp) {
          max_rollback_observed = std::max(
            max_rollback_observed, optimizer.terminalCritic()->lastPreStartRollbackM());
        }
        stepAckermannPose(actual, speed, cmd.twist, plant, 0.05, 0.56, 0.176, 0.05, 0.945, 1.21);
      }
      EXPECT_TRUE(saw_pre_cusp);
      EXPECT_TRUE(saw_post_cusp);
      EXPECT_GE(cusp_entry_speed, 0.22);
      EXPECT_LE(cusp_entry_speed, 0.28);
      EXPECT_GE(max_rollback_observed, 0.050);
      EXPECT_LE(max_rollback_observed, 0.095);
      EXPECT_TRUE(reached);
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
    }
  }

  // (2) Section 2, Section C & Section F (Items 13, 17, 18, 24):
  //     - Recorded WP1 synthetic-curve sequence (wp1_overlap_recorded, extracted from
  //       20261114_fa_critic_v3/raw/arm2_candidate_a11_fa_critic/suite1_office_wp1/
  //       office_wp1_course/nav2_course_summary.json, episodes[2], Repeat 3, k=295..354,
  //       elapsed_s=15.885..18.835 s, wall_s=1794708831.319..1794708835.526) spanning from
  //       d_goal = 0.6384 m across the 0.35 m cutoff (at k=319, d_goal = 0.3450 m) down to
  //       d_goal = 0.0512 m:
  //       * guidanceOwnership() == OVERLAP_SHARED for all ticks with d_goal > 0.35 m;
  //       * guidanceOwnership() == TERMINAL_EXCLUSIVE for all ticks with d_goal <= 0.35 m;
  //       * non-zero total critic cost AND non-zero TerminalApproachCritic cost contribution
  //         on every single tick across the 0.35 m cutoff (zero guidance-gap ticks);
  //       * 0 redraws even when max_redraws = 1 (isLatchConsumed() == true).
  //     - Bounded Ackermann joinability validation (Section C, Items 13 & 18): when lateral error
  //       is well within the corridor (<= max_tracking_drift_ = 0.045 m) and heading error is
  //       within the broad corridor allowance, but remaining runway and heading make joining the
  //       latched curve at min_turning_radius = 0.56 m kinematically infeasible, score() rejects
  //       with DRIFT_EXCEEDED and 0 redraws (retaining velocity and steering state across handoff).
  {
    using Ownership = mppi::critics::GuidanceOwnership;
    const auto & wp1_sc = atlas->at("wp1_overlap_recorded");
    grid(wp1_sc);
    auto parent = node(true, 500, 0.325, 7, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_entry_path_turn_angle",
      rclcpp::ParameterValue(0.785));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    auto gc = makeGoalChecker(parent);
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(wp1_sc.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    optimizer.setSeededGaussianBank(42u);

    const auto goal_msg = wp1_sc.message().poses.back().pose;
    // Exact recorded states from 20261114_fa_critic_v3 WP1 Repeat 3 (episodes[2], k=295..354):
    const std::vector<std::pair<Pose, std::pair<double, double>>> wp1_states = {
      {{2.803300, -0.526000, -1.389300}, {0.2830, -0.0160}},  // k=295, dg=0.6384m (OVERLAP)
      {{2.809500, -0.565900, -1.400900}, {0.2734, 0.0619}},   // k=298, dg=0.5980m (OVERLAP)
      {{2.816700, -0.607900, -1.406100}, {0.2632, 0.0906}},   // k=301, dg=0.5554m (OVERLAP)
      {{2.819200, -0.644500, -1.407200}, {0.2626, 0.1037}},   // k=304, dg=0.5189m (OVERLAP)
      {{2.822200, -0.680000, -1.397000}, {0.2580, 0.0957}},   // k=307, dg=0.4835m (OVERLAP)
      {{2.828800, -0.713500, -1.381800}, {0.2423, 0.0782}},   // k=310, dg=0.4493m (OVERLAP)
      {{2.833700, -0.750000, -1.367100}, {0.2279, 0.0324}},   // k=313, dg=0.4125m (OVERLAP)
      {{2.837500, -0.782500, -1.359100}, {0.2133, -0.0021}},  // k=316, dg=0.3799m (OVERLAP)
      {{2.842500, -0.805300, -1.356100}, {0.2020, -0.0343}},  // k=318, dg=0.3566m (OVERLAP)
      {{2.844700, -0.816700, -1.353100}, {0.2008, -0.0551}},  // k=319, dg=0.3450m (EXCLUSIVE)
      {{2.851000, -0.846200, -1.354300}, {0.2037, -0.1256}},  // k=322, dg=0.3148m (EXCLUSIVE)
      {{2.857900, -0.877900, -1.365200}, {0.2024, -0.1800}},  // k=325, dg=0.2824m (EXCLUSIVE)
      {{2.864000, -0.904000, -1.378500}, {0.2031, -0.2333}},  // k=328, dg=0.2556m (EXCLUSIVE)
      {{2.869300, -0.934300, -1.407500}, {0.2112, -0.2766}},  // k=331, dg=0.2248m (EXCLUSIVE)
      {{2.873200, -0.961200, -1.442300}, {0.2181, -0.2973}},  // k=334, dg=0.1977m (EXCLUSIVE)
      {{2.877000, -0.992100, -1.490200}, {0.2162, -0.2469}},  // k=337, dg=0.1667m (EXCLUSIVE)
      {{2.879000, -1.021800, -1.529400}, {0.2035, -0.2044}},  // k=340, dg=0.1373m (EXCLUSIVE)
      {{2.880000, -1.056200, -1.568800}, {0.1872, -0.1839}},  // k=343, dg=0.1039m (EXCLUSIVE)
      {{2.871800, -1.086600, -1.614900}, {0.1536, -0.1885}},  // k=347, dg=0.0795m (EXCLUSIVE)
      {{2.864500, -1.136600, -1.691700}, {0.0688, -0.2117}}};  // k=354, dg=0.0512m (EXCLUSIVE)

    size_t overlap_shared_ticks = 0u;
    size_t terminal_exclusive_ticks = 0u;
    for (size_t idx = 0u; idx < wp1_states.size(); ++idx) {
      auto actual = pose(wp1_states[idx].first);
      auto local_path = handler.transformPath(actual);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = wp1_states[idx].second.first;
      speed.angular.z = wp1_states[idx].second.second;
      EXPECT_NO_THROW(
        optimizer.evalControl(actual, speed, local_path, goal_msg, gc.get()))
        << "WP1 overlap-to-goal sequence aborted at step " << idx;
      EXPECT_TRUE(optimizer.terminalCritic()->isLatched()) << "step " << idx;
      EXPECT_FALSE(optimizer.terminalCritic()->isReferencePathLatched()) << "step " << idx;
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed()) << "step " << idx;
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0) << "step " << idx;
      // Non-zero total critic cost AND non-zero TerminalApproachCritic cost contribution on
      // every single tick across the 0.35 m cutoff:
      EXPECT_GT(xt::sum(optimizer.costs())(), 0.0f) << "step " << idx;
      EXPECT_GT(optimizer.evaluateTerminalCriticOnlyCost(), 0.0f) << "step " << idx;
      const double d_goal = std::hypot(
        goal_msg.position.x - actual.pose.position.x,
        goal_msg.position.y - actual.pose.position.y);
      if (d_goal > 0.35) {
        EXPECT_EQ(
          optimizer.terminalCritic()->guidanceOwnership(),
          Ownership::OVERLAP_SHARED) << "step " << idx;
        ++overlap_shared_ticks;
      } else {
        EXPECT_EQ(
          optimizer.terminalCritic()->guidanceOwnership(),
          Ownership::TERMINAL_EXCLUSIVE) << "step " << idx;
        ++terminal_exclusive_ticks;
      }
    }
    EXPECT_EQ(overlap_shared_ticks, 9u);
    EXPECT_EQ(terminal_exclusive_ticks, 11u);

    // Leaving the shared-control joinable corridor (+150 mm lateral offset in overlap ring)
    // must escalate with DRIFT_EXCEEDED and 0 redraws (even with max_redraws = 1):
    {
      handler.setPath(wp1_sc.message());
      optimizer.terminalCritic()->onNewPlan(wp1_sc.message());
      auto entry_actual = pose(wp1_states[0].first);
      auto local_path = handler.transformPath(entry_actual);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = wp1_states[0].second.first;
      speed.angular.z = wp1_states[0].second.second;
      EXPECT_NO_THROW(optimizer.evalControl(entry_actual, speed, local_path, goal_msg, gc.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());

      auto bad_overlap = pose({2.6500, -0.6812, -1.3786});
      local_path = handler.transformPath(bad_overlap);
      speed.linear.x = 0.24;
      try {
        optimizer.evalControl(bad_overlap, speed, local_path, goal_msg, gc.get());
        FAIL() << "Expected DRIFT_EXCEEDED when leaving joinable overlap corridor on WP1";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
      }
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
    }

    // Section C (Items 13 & 18): Bounded Ackermann joinability check across overlap & handoff
    // Separately exercise actual lateral departure and insufficient Ackermann joining runway,
    // retaining optimizer velocity and steering state across score() calls. Use diagnostics from
    // the failing evaluation; clearActiveCurve() deliberately clears the live error getters.
    {
      // (a) OVERLAP_SHARED: 56.8 mm from the latched synthetic curve exceeds its 45 mm limit.
      handler.setPath(wp1_sc.message());
      optimizer.terminalCritic()->onNewPlan(wp1_sc.message());
      auto entry_actual = pose(wp1_states[0].first);
      auto local_path = handler.transformPath(entry_actual);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = wp1_states[0].second.first;
      speed.angular.z = wp1_states[0].second.second;
      EXPECT_NO_THROW(optimizer.evalControl(entry_actual, speed, local_path, goal_msg, gc.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      ASSERT_EQ(optimizer.terminalCritic()->guidanceOwnership(), Ownership::OVERLAP_SHARED);

      auto unjoinable_overlap = pose({2.837500, -0.782500, -2.150000});
      const double overlap_lateral = optimizer.terminalCritic()->projectOntoCurve(
        unjoinable_overlap.pose.position.x, unjoinable_overlap.pose.position.y,
        optimizer.terminalCritic()->latchedCurve(), 1.0).lateral_dist;
      local_path = handler.transformPath(unjoinable_overlap);
      speed.linear.x = 0.2133;
      speed.angular.z = -0.0021;
      try {
        optimizer.evalControl(unjoinable_overlap, speed, local_path, goal_msg, gc.get());
        FAIL() << "Expected DRIFT_EXCEEDED on lateral departure in OVERLAP_SHARED";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=DRIFT_EXCEEDED"), std::string::npos) << ex.what();
      }
      const auto overlap_message = optimizer.terminalCritic()->lastAbortMessage();
      const auto overlap_field = overlap_message.find("curve_lat_err=");
      ASSERT_NE(overlap_field, std::string::npos);
      EXPECT_NEAR(std::stod(overlap_message.substr(overlap_field + 14u)), overlap_lateral, 1e-4);
      EXPECT_GT(overlap_lateral, 1e-4);
      // The previous post-clear assertion read zero. This fixture actually departs laterally.
      EXPECT_NEAR(overlap_lateral, 0.0568, 1e-4);
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);

      // (b) Crossing the 0.35 m cutoff into TERMINAL_EXCLUSIVE on WP1 (step 0..8 in
      //     OVERLAP_SHARED, retaining velocity and steering state, then crossing at k=319
      //     (2.8447, -0.8167) at d_goal = 0.3450 m <= 0.35 m, lateral_dist = 61.5 mm below
      //     the live 104.9 mm envelope, with yaw = -2.18 rad (|theta0_goal| = 0.35 rad << M_PI/4,
      //     |curve_yaw_err| = 0.65 rad < M_PI/4, so the old corridor check passed, but the 2-arc
      //     Ackermann joinability check at R_min = 0.56 m requires 0.56 m > runway 0.35 m)):
      handler.setPath(wp1_sc.message());
      optimizer.terminalCritic()->onNewPlan(wp1_sc.message());
      for (size_t idx = 0u; idx <= 8u; ++idx) {
        auto act = pose(wp1_states[idx].first);
        auto lp = handler.transformPath(act);
        speed.linear.x = wp1_states[idx].second.first;
        speed.angular.z = wp1_states[idx].second.second;
        EXPECT_NO_THROW(optimizer.evalControl(act, speed, lp, goal_msg, gc.get()));
      }
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
      ASSERT_EQ(optimizer.terminalCritic()->guidanceOwnership(), Ownership::OVERLAP_SHARED);

      auto unjoinable_cutoff = pose({2.844700, -0.816700, -2.180000});
      const double cutoff_lateral = optimizer.terminalCritic()->projectOntoCurve(
        unjoinable_cutoff.pose.position.x, unjoinable_cutoff.pose.position.y,
        optimizer.terminalCritic()->latchedCurve(), 1.0).lateral_dist;
      local_path = handler.transformPath(unjoinable_cutoff);
      speed.linear.x = wp1_states[9].second.first;
      speed.angular.z = wp1_states[9].second.second;
      try {
        optimizer.evalControl(unjoinable_cutoff, speed, local_path, goal_msg, gc.get());
        FAIL() << "Expected CURVE_UNJOINABLE when crossing 0.35 m cutoff with unjoinable heading";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=CURVE_UNJOINABLE"), std::string::npos) << ex.what();
      }
      const auto cutoff_message = optimizer.terminalCritic()->lastAbortMessage();
      const auto cutoff_field = cutoff_message.find("curve_lat_err=");
      ASSERT_NE(cutoff_field, std::string::npos);
      EXPECT_NEAR(std::stod(cutoff_message.substr(cutoff_field + 14u)), cutoff_lateral, 1e-4);
      EXPECT_GT(cutoff_lateral, 1e-4);
      // Actual error is below the live 0.1049 m limit, but was never below 25 mm.
      EXPECT_NEAR(cutoff_lateral, 0.0615, 1e-4);
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
    }

    // Obstacle appearing on the latched WP1 curve during overlap must escalate with
    // LATCHED_CURVE_BLOCKED and 0 redraws:
    {
      handler.setPath(wp1_sc.message());
      optimizer.terminalCritic()->onNewPlan(wp1_sc.message());
      auto entry_actual = pose(wp1_states[0].first);
      auto local_path = handler.transformPath(entry_actual);
      geometry_msgs::msg::Twist speed;
      speed.linear.x = wp1_states[0].second.first;
      EXPECT_NO_THROW(optimizer.evalControl(entry_actual, speed, local_path, goal_msg, gc.get()));
      ASSERT_TRUE(optimizer.terminalCritic()->isLatched());

      Scenario blocked_wp1 = wp1_sc;
      blocked_wp1.obstacles.push_back({2.88, -1.18, 2.95, -1.10});
      grid(blocked_wp1);
      auto step1_actual = pose(wp1_states[1].first);
      local_path = handler.transformPath(step1_actual);
      try {
        optimizer.evalControl(step1_actual, speed, local_path, goal_msg, gc.get());
        FAIL() << "Expected LATCHED_CURVE_BLOCKED when obstacle blocks WP1 curve in overlap";
      } catch (const nav2_core::InvalidPath & ex) {
        EXPECT_NE(
          std::string(ex.what()).find("reason=LATCHED_CURVE_BLOCKED"), std::string::npos)
          << ex.what();
      }
      EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
      EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
      grid(wp1_sc);
    }
  }

  // (2b) Section 7.10 & Section 7.2: Direct passive_overlap_ref test:
  //      Latch reference_path inside cutoff (d_goal = 0.32 m <= 0.35 m), then step back into
  //      the hysteresis band (d_goal = 0.40 m in (0.35, 0.42]) where stock critics steer:
  //      - A 55 mm cross-track shift (> 45 mm max_tracking_drift_, <= 95 mm viable corridor)
  //        in the hysteresis band must NOT self-abort with DRIFT_EXCEEDED;
  //      - An obstacle appearing on the reference path while at d_goal = 0.40 m (> 0.35 m)
  //        must unlatch into PREPARE without throwing InvalidPath while stock critics steer,
  //        and throw once inside cutoff (d_goal <= 0.35 m).
  {
    using Ownership = mppi::critics::GuidanceOwnership;
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    auto gc = makeGoalChecker(parent);
    mppi::ParametersHandler params(parent);
    params.start();
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    handler.initialize(parent, "FollowPath", cm, buffer, &params);
    handler.setPath(early_sc.message());
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    const auto goal_msg = early_sc.message().poses.back().pose;
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    // Latch reference_path inside cutoff (d_goal = 0.34 m <= 0.35 m)
    auto p_latch = pose({-0.34, 0.0, 0.0});
    auto lp_latch = handler.transformPath(p_latch);
    EXPECT_NO_THROW(optimizer.evalControl(p_latch, speed, lp_latch, goal_msg, gc.get()));
    ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
    ASSERT_TRUE(optimizer.terminalCritic()->isReferencePathLatched());
    EXPECT_EQ(optimizer.terminalCritic()->guidanceOwnership(), Ownership::TERMINAL_EXCLUSIVE);

    // Step into (0.35, 0.42] band (d_goal = hypot(-0.35, 0.060) = 0.3551 m, rollback = 10 mm)
    // with 60 mm (> 45 mm) lateral shift: must NOT throw DRIFT_EXCEEDED while in PREPARE
    auto p_band = pose({-0.35, 0.060, 0.0});
    auto lp_band = handler.transformPath(p_band);
    EXPECT_NO_THROW(optimizer.evalControl(p_band, speed, lp_band, goal_msg, gc.get()));
    EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(optimizer.terminalCritic()->guidanceOwnership(), Ownership::PREPARE);
    EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);

    // Place an obstacle on the reference path goal footprint while still at d_goal = 0.3551 m:
    // passive reference_path in 0.35..0.70 m unlatches into PREPARE without throwing InvalidPath
    Scenario blocked_early = early_sc;
    blocked_early.obstacles.push_back({0.10, -0.02, 0.13, 0.02});
    grid(blocked_early);
    EXPECT_NO_THROW(optimizer.evalControl(p_band, speed, lp_band, goal_msg, gc.get()));
    EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
    EXPECT_EQ(optimizer.terminalCritic()->guidanceOwnership(), Ownership::PREPARE);
    EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);

    // Once stepping inside cutoff (d_goal = 0.30 m <= 0.35 m), throws InvalidPath
    auto p_inside = pose({-0.30, 0.0, 0.0});
    auto lp_inside = handler.transformPath(p_inside);
    EXPECT_THROW(
      optimizer.evalControl(p_inside, speed, lp_inside, goal_msg, gc.get()),
      nav2_core::InvalidPath);
    EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
    grid(early_sc);
  }
}

TEST_F(ControllerScenarios, TerminalApproachOptionA11ReviewPunchlistPart2Regressions)
{
  // (2c) Section 7.1: Sub-min_approach_distance (< 0.16 m) turning-radius enforcement:
  //      1) Pose (-0.14, 0.0, 0.70) has d_goal = 0.14 m < 0.16 m and |yaw_err| = 0.70 rad
  //         (kinematically impossible at R_min = 0.5605 m) -> must reject with ENTRY_INFEASIBLE.
  //      2) Reference path with a sharp sub-R_min kink inside 0.16 m -> must reject
  //         reference_path admission with ENTRY_INFEASIBLE.
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    controller.setPlan(early_sc.message());
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.05;
    try {
      controller.computeVelocityCommands(pose({-0.14, 0.0, 0.70}), speed, nullptr);
      FAIL() << "Expected ENTRY_INFEASIBLE on sub-0.16 m pose (-0.14, 0.0, 0.70)";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos) << ex.what();
    }

    Scenario kinked_ref = early_sc;
    kinked_ref.path = {
      {-0.14, 0.0, 0.0},
      {-0.09, 0.0, 0.0},
      {-0.05, 0.03, 0.55},
      {0.0, 0.0, 0.0}};
    kinked_ref.gears = {1, 1, 1};
    controller.reset();
    controller.setPlan(kinked_ref.message());
    try {
      controller.computeVelocityCommands(pose({-0.14, 0.0, 0.0}), speed, nullptr);
      FAIL() << "Expected ENTRY_INFEASIBLE on sub-R_min kinked reference path inside 0.16 m";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=ENTRY_INFEASIBLE"), std::string::npos) << ex.what();
    }

    // Interior segment with 0.20 rad step (< goal_yaw_tolerance = 0.25 rad, but > max_seg_dth =
    // 0.089 rad at R_min = 0.56 m over 0.04 m) must also reject with ENTRY_INFEASIBLE:
    Scenario interior_kink_ref = early_sc;
    interior_kink_ref.path = {
      {-0.12, 0.0, 0.0},
      {-0.08, 0.0, 0.0},
      {-0.04, 0.0, 0.20},
      {0.0, 0.0, 0.0}};
    interior_kink_ref.gears = {1, 1, 1};
    controller.reset();
    controller.setPlan(interior_kink_ref.message());
    EXPECT_THROW(
      controller.computeVelocityCommands(pose({-0.12, 0.0, 0.0}), speed, nullptr),
      nav2_core::InvalidPath);

    // Parallel-offset poses (-0.14, 0.080, 0.0) (goal_dist = 0.161 m) and (-0.11, 0.080, 0.0)
    // (goal_dist = 0.136 m < 0.16 m) cannot reach |y_end| <= 0.045 m at R_min = 0.56 m
    // (best |y_end| >= 0.0625 m) -> both must reject with ENTRY_INFEASIBLE:
    for (const double lon_x : {-0.14, -0.11}) {
      controller.reset();
      controller.setPlan(early_sc.message());
      EXPECT_THROW(
        controller.computeVelocityCommands(pose({lon_x, 0.080, 0.0}), speed, nullptr),
        nav2_core::InvalidPath);
    }

    // Active opposite-gear cusp-braking handoffs (< 0.16 m) from m8_lateral_shift (r1/r4)
    // and m6_reverse_right_90 (r2/r4/r5): when PathHandler hands off inside 0.16 m while
    // the robot is still coasting in the pre-cusp gear (vx * gear < -0.24 m/s) within the
    // tracking corridor (|y0_goal| <= 0.045 m) and the 2-phase cusp-coast apex + approach arc
    // reaches the goal box, reference_path must latch and hold through the pre-apex coast
    // without ENTRY_INFEASIBLE, ROLLBACK_EXCEEDED, or DRIFT_EXCEEDED.
    Scenario m8_cusp_stub = early_sc;
    m8_cusp_stub.path = {
      {-0.071, 0.0057, -0.16},
      {-0.035, 0.0014, -0.08},
      {0.0, 0.0, 0.0}};
    m8_cusp_stub.gears = {1, 1};
    controller.reset();
    controller.setPlan(m8_cusp_stub.message());
    geometry_msgs::msg::Twist cusp_speed;
    cusp_speed.linear.x = -0.26;
    // m8_r1 handoff state: lon = 0.094 m, y0 = -0.031 m, th0 = -0.581 rad, vx = -0.26 m/s
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.094, -0.031, -0.581}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());
    // Coast across y = 0 to the cusp apex (-0.185, +0.025, -0.466) and then reverse into forward:
    cusp_speed.linear.x = -0.12;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.145, 0.000, -0.510}), cusp_speed, nullptr));
    cusp_speed.linear.x = -0.02;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.185, 0.025, -0.466}), cusp_speed, nullptr));
    cusp_speed.linear.x = 0.08;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.160, 0.018, -0.420}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // m8_r4 handoff state: lon = 0.136 m, y0 = -0.0432 m, th0 = -0.626 rad, vx = -0.272 m/s
    Scenario m8_r4_stub = early_sc;
    m8_r4_stub.path = {
      {-0.094, 0.0085, -0.18},
      {-0.047, 0.0021, -0.09},
      {0.0, 0.0, 0.0}};
    m8_r4_stub.gears = {1, 1};
    controller.reset();
    controller.setPlan(m8_r4_stub.message());
    cusp_speed.linear.x = -0.272;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.136, -0.0432, -0.626}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // m8_v2_r1 (2-pose 4.56 cm terminal stub, |y0_goal| = 0.046 m > 0.045 m at entry, coasting
    // inward at vx = -0.2553 m/s to |y_apex| = 0.020 m <= 0.045 m):
    Scenario m8_v2_r1_stub = early_sc;
    m8_v2_r1_stub.path = {
      {-0.0456, 0.0020, -0.14},
      {0.0, 0.0, 0.0}};
    m8_v2_r1_stub.gears = {1};
    controller.reset();
    controller.setPlan(m8_v2_r1_stub.message());
    cusp_speed.linear.x = -0.2553;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.079, -0.046, -0.470}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // m8_v2_r4 (6-pose 22.3 cm post-cusp curve, handoff at goal_dist = 0.247 m >= 0.16 m with
    // cross_track = 0.0016 m, yaw_err = 0.3093 rad, vx = -0.2029 m/s, followed by backward coast
    // to apex and full-lock R_min = 0.56 m forward turn-in reaching curve_lat_err = 0.0474 m at
    // goal_dist = 0.084 m): must hold latch without DRIFT_EXCEEDED:
    Scenario m8_v2_r4_curve = early_sc;
    m8_v2_r4_curve.path = {
      {-0.214, 0.043, -0.36},
      {-0.171, 0.026, -0.28},
      {-0.128, 0.014, -0.20},
      {-0.085, 0.005, -0.12},
      {-0.043, 0.001, -0.05},
      {0.0, 0.0, 0.0}};
    m8_v2_r4_curve.gears = {1, 1, 1, 1, 1};
    controller.reset();
    controller.setPlan(m8_v2_r4_curve.message());
    cusp_speed.linear.x = -0.2029;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.2409, 0.0545, -0.7082}), cusp_speed, nullptr));
    ASSERT_TRUE(controller.terminalCritic()->isLatched());
    cusp_speed.linear.x = -0.01;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.2767, 0.0815, -0.6127}), cusp_speed, nullptr));
    cusp_speed.linear.x = 0.08;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.2114, 0.0250, -0.6113}), cusp_speed, nullptr));
    cusp_speed.linear.x = 0.14;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.1343, -0.0196, -0.4521}), cusp_speed, nullptr));
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(pose({-0.0668, -0.0455, -0.3275}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // m6_r5 reverse-gear (gear = -1) handoff state: lon = 0.089 m, y0 = +0.0242 m,
    // th0 = +0.497 rad (nose yaw = M_PI + 0.497 = -2.6446 rad), vx = +0.247 m/s
    // (vx * gear = -0.247 m/s < 0), followed by coast to apex and reverse approach:
    const Scenario rev_early_sc = early_sc.variant(false, true);
    controller.reset();
    controller.setPlan(rev_early_sc.message());
    cusp_speed.linear.x = 0.247;
    const double m6_nose_yaw = controller_test::wrap(M_PI + 0.497);
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(
        pose({-0.089, 0.0242, m6_nose_yaw}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());
    cusp_speed.linear.x = 0.04;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(
        pose({-0.145, -0.002, controller_test::wrap(M_PI + 0.387)}), cusp_speed, nullptr));
    cusp_speed.linear.x = -0.08;
    EXPECT_NO_THROW(
      controller.computeVelocityCommands(
        pose({-0.125, -0.005, controller_test::wrap(M_PI + 0.330)}), cusp_speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // Conversely, an active opposite-gear cusp-braking pose with excessive heading error
    // (-0.09, 0.0, 0.85, vx = -0.25 m/s) cannot reach the goal box even after coasting to apex
    // -> must reject with ENTRY_INFEASIBLE:
    controller.reset();
    controller.setPlan(m8_cusp_stub.message());
    cusp_speed.linear.x = -0.25;
    EXPECT_THROW(
      controller.computeVelocityCommands(pose({-0.090, 0.0, 0.85}), cusp_speed, nullptr),
      nav2_core::InvalidPath);
  }

  // (3) Section 3: Stock cutoff 0.50 m / configured cutoff 0.35 m with infeasible pose at 0.45 m
  //     must treat 0.50 m as the effective handoff cutoff (failing immediately rather than
  //     leaving a 0.35..0.50 m guidance gap where both stock critics and terminal critic are off).
  {
    const auto & late_sc = atlas->at("late_lateral_capture");
    grid(late_sc);
    for (const std::string & pol : {std::string("fallback"), std::string("replan_signal")}) {
      SCOPED_TRACE("cutoff_mismatch_050_035/" + pol);
      auto parent = node(true, 64, 0.558, 6, 0.50);
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.infeasible_policy", rclcpp::ParameterValue(pol));
      parent->declare_parameter(
        "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
      auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
      ScenarioController controller;
      controller.configureWithCompiledCritics(parent, buffer, cm);
      controller.setPlan(late_sc.message());
      // d_goal = hypot(-0.41, -0.185) = 0.4498 m <= 0.50 m stock cutoff (both stock critics off!)
      auto pose_045 = pose({-0.41, -0.185, 0.0});
      geometry_msgs::msg::Twist speed;
      speed.linear.x = 0.10;
      EXPECT_THROW(
        controller.computeVelocityCommands(pose_045, speed, nullptr),
        nav2_core::InvalidPath) << "Expected immediate handoff failure at 0.45 m under " << pol;
    }
  }

  // (4) Section 7.2: Unlatched infeasible pose at (-0.60, 0.03, yaw 0.45) in the 0.35..0.65 m
  //     overlap ring under policy="abort" must NOT abort before the critic owns steering; if
  //     stock critics align the robot by <= 0.35 m it latches and succeeds, whereas if the robot
  //     is still unaligned at <= 0.35 m it throws ENTRY_INFEASIBLE at the handoff boundary.
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    controller.setPlan(early_sc.message());
    auto overlap_pose = pose({-0.60, 0.03, 0.45});
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_pose, speed, nullptr));
    EXPECT_FALSE(controller.terminalCritic()->isLatched());

    // Stock critics align the robot onto the straight reference path by d_goal = 0.30 m <= 0.35 m:
    // authoritative handoff check latches cleanly with 0 aborts!
    // Capture [TERMINAL_APPROACH_LATCH] and [TERMINAL_APPROACH_ESCALATE] via rcutils_logging
    // (Section 4 Items 27 & 28) and verify all 6 required diagnostic fields are emitted:
    static std::vector<std::string> captured_tac_logs;
    captured_tac_logs.clear();
    rcutils_logging_output_handler_t prev_handler = rcutils_logging_get_output_handler();
    rcutils_logging_set_output_handler(
      [](const rcutils_log_location_t *, int, const char *, rcutils_time_point_value_t,
      const char * format, va_list * args) {
        char buf[1024];
        va_list copy;
        va_copy(copy, *args);
        std::vsnprintf(buf, sizeof(buf), format, copy);
        va_end(copy);
        std::string line(buf);
        if (line.find("[TERMINAL_APPROACH_") != std::string::npos) {
          captured_tac_logs.push_back(std::move(line));
        }
      });
    auto aligned_handoff_pose = pose({-0.30, 0.01, 0.05});
    EXPECT_NO_THROW(controller.computeVelocityCommands(aligned_handoff_pose, speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());

    // Conversely, if the robot enters the overlap ring at (-0.60, 0.03, 0.45) and remains
    // unaligned when crossing d_goal <= 0.35 m ((-0.30, 0.18, 0.45)), it throws ENTRY_INFEASIBLE
    // at the handoff cutoff:
    controller.reset();
    controller.setPlan(early_sc.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(overlap_pose, speed, nullptr));
    EXPECT_THROW(
      controller.computeVelocityCommands(pose({-0.30, 0.18, 0.45}), speed, nullptr),
      nav2_core::InvalidPath);
    rcutils_logging_set_output_handler(prev_handler);

    std::string latch_log;
    std::string escalate_log;
    for (const auto & ln : captured_tac_logs) {
      if (ln.find("[TERMINAL_APPROACH_LATCH]") != std::string::npos) {
        latch_log = ln;
      }
      if (ln.find("[TERMINAL_APPROACH_ESCALATE]") != std::string::npos) {
        escalate_log = ln;
      }
    }
    ASSERT_FALSE(latch_log.empty());
    ASSERT_FALSE(escalate_log.empty());
    for (const char * field : {
        "cross_track=", "curve_lat_err=", "rollback=",
        "ownership=", "lat_limit=", "eff_cutoff="})
    {
      EXPECT_NE(latch_log.find(field), std::string::npos) << latch_log;
      EXPECT_NE(escalate_log.find(field), std::string::npos) << escalate_log;
      EXPECT_NE(controller.terminalCritic()->lastAbortMessage().find(field), std::string::npos)
        << controller.terminalCritic()->lastAbortMessage();
    }
  }

  // (5) Section 7.6: Live weight change (align_weight) while latched must NOT abort with
  //     PARAM_CHANGE_WHILE_LATCHED.
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    controller.setPlan(early_sc.message());
    auto inside_pose = pose({-0.30, -0.05, 0.0});
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;
    EXPECT_NO_THROW(controller.computeVelocityCommands(inside_pose, speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.align_weight", 48.0)})[0].successful);
    EXPECT_NO_THROW(controller.computeVelocityCommands(inside_pose, speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());
  }

  // (6) Section 7.3: When already within (goal_xy_tol, goal_yaw_tol) at the goal, do NOT
  //     re-solve or abort on curve lateral drift while waiting to settle.
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    controller.setPlan(early_sc.message());
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.05;
    EXPECT_NO_THROW(controller.computeVelocityCommands(pose({-0.30, 0.0, 0.0}), speed, nullptr));
    ASSERT_TRUE(controller.terminalCritic()->isLatched());
    // Pose inside goal box (d_goal = hypot(-0.05, 0.06) = 0.078 m <= 0.10 m, |yaw| = 0.10 <= 0.25)
    // with 60 mm (> 45 mm max_tracking_drift_) lateral offset: holds latch with 0 redraws.
    EXPECT_NO_THROW(controller.computeVelocityCommands(pose({-0.05, 0.06, 0.10}), speed, nullptr));
    EXPECT_TRUE(controller.terminalCritic()->isLatched());
    EXPECT_EQ(controller.terminalCritic()->redrawCount(), 0);
  }

  // (7) Section 2 Main #5 & Section 7 Tests #5 (Items 14, 48): Once latch_consumed_ is true and
  //     the latch is cleared under policy="fallback" in the overlap ring (d_goal > 0.35 m),
  //     subsequent ticks on the same plan must NOT re-solve and re-latch a second curve.
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("fallback")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(0));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    controller.setPlan(early_sc.message());
    auto * tc = controller.terminalCritic();
    ASSERT_NE(tc, nullptr);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    // Tick 0 (d_goal = 0.553 m > 0.35 m): latches capture curve and sets latch_consumed_ = true
    EXPECT_NO_THROW(controller.computeVelocityCommands(pose({-0.55, -0.06, 0.0}), speed, nullptr));
    EXPECT_TRUE(tc->isLatched());
    EXPECT_TRUE(tc->isLatchConsumed());

    // Tick 1 (d_goal = 0.536 m > 0.35 m): drift by 70 mm beyond the overlap join corridor under
    // policy="fallback" -> unlatches without throwing, keeping latch_consumed_ == true
    EXPECT_NO_THROW(controller.computeVelocityCommands(pose({-0.52, -0.13, 0.0}), speed, nullptr));
    EXPECT_FALSE(tc->isLatched());
    EXPECT_TRUE(tc->isLatchConsumed());

    // Tick 2 (d_goal = 0.483 m > 0.35 m): robot returns to a feasible pose (-0.48, -0.05, 0.0) on
    // the SAME plan -> latch_consumed_ must block re-solving / re-latching!
    EXPECT_NO_THROW(controller.computeVelocityCommands(pose({-0.48, -0.05, 0.0}), speed, nullptr));
    EXPECT_FALSE(tc->isLatched()) << "latch_consumed_ must block re-latching on the same plan";
    EXPECT_TRUE(tc->isLatchConsumed());
  }

  // (8) Follow-Up Review Finding 1 (Consumed-latch guard vs configured, permitted redraws):
  //     Exercise BOTH local-goal-shift (L2035-2070) and score-detected gear-flip (L2305-2336)
  //     branches through score() with an active latch, max_redraws = 1, unused redraw budget,
  //     and a feasible replacement curve:
  //     - Verify an actual replacement curve is latched and scored (> 0 cost), not just a counter.
  //     - Verify the spent-budget case (a subsequent qualifying failure is rejected).
  //     - Retain direct max_redraws = 0 regressions (no replacement solved).
  {
    const auto & early_sc = atlas->at("early_lateral_capture");
    grid(early_sc);
    auto parent = node(true, 64, 0.558, 6, 0.35);
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.infeasible_policy",
      rclcpp::ParameterValue(std::string("abort")));
    parent->declare_parameter(
      "FollowPath.TerminalApproachCritic.max_redraws", rclcpp::ParameterValue(1));
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    buffer->setUsingDedicatedThread(true);
    auto setGlobalToMapTransform = [&](double tx, double ty) {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = rclcpp::Time(0, 0, parent->get_clock()->get_clock_type());
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id = "global_test";
        tf_msg.transform.translation.x = tx;
        tf_msg.transform.translation.y = ty;
        tf_msg.transform.rotation.w = 1.0;
        buffer->setTransform(tf_msg, "unittest", true);
      };
    setGlobalToMapTransform(0.0, 0.0);

    ScenarioController controller;
    controller.configureWithCompiledCritics(parent, buffer, cm);
    nav_msgs::msg::Path global_plan = early_sc.message();
    global_plan.header.frame_id = "global_test";
    for (auto & ps : global_plan.poses) {
      ps.header.frame_id = "global_test";
    }
    controller.setPlan(global_plan);
    auto * tc = controller.terminalCritic();
    ASSERT_NE(tc, nullptr);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.10;

    // (8a) Local-goal-shift branch (max_redraws = 1):
    // Tick 0: latch initial capture curve at (-0.30, -0.06, 0.0) inside cutoff (d_goal = 0.3059 m)
    auto init_pose = pose({-0.30, -0.06, 0.0});
    auto init_cmd = controller.computeVelocityCommands(init_pose, speed, nullptr);
    ASSERT_TRUE(tc->isLatched());
    EXPECT_FALSE(tc->isReferencePathLatched());
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 0);
    EXPECT_GT(std::fabs(init_cmd.twist.linear.x), 1e-3);

    // Tick 1: 0.30 m (+0.24 m, -0.18 m) local goal shift (> 0.25 m goal_reset_pos_tol_) with a
    // feasible replacement pose -> must release latch_consumed_ for the authorized replacement,
    // latch a new capture curve, and score it (> 0 cost)!
    setGlobalToMapTransform(0.24, -0.18);
    auto shift1_pose = pose({-0.30 + 0.24, -0.06 - 0.18, 0.0});
    geometry_msgs::msg::TwistStamped shift1_cmd;
    EXPECT_NO_THROW(shift1_cmd = controller.computeVelocityCommands(shift1_pose, speed, nullptr));
    EXPECT_TRUE(tc->isLatched())
      << "Permitted local-goal-shift redraw (max_redraws=1) must latch a replacement curve";
    EXPECT_FALSE(tc->isReferencePathLatched());
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    ASSERT_FALSE(tc->latchedCurve().empty());
    EXPECT_NEAR(tc->latchedCurve().back().x, 0.24f, 0.05f);
    EXPECT_NEAR(tc->latchedCurve().back().y, -0.18f, 0.05f);
    EXPECT_GT(std::fabs(shift1_cmd.twist.linear.x), 1e-3);
    {
      mppi::models::State st_s;
      st_s.reset(2, 10);
      st_s.pose.header.frame_id = "map";
      st_s.pose.pose = shift1_pose.pose;
      st_s.speed.linear.x = 0.10f;
      mppi::models::Trajectories tr_s;
      tr_s.reset(2, 10);
      for (size_t k = 0; k < 10; ++k) {
        tr_s.x(0, k) = static_cast<float>(shift1_pose.pose.position.x) + 0.025f * k;
        tr_s.y(0, k) = static_cast<float>(shift1_pose.pose.position.y) + 0.005f * k;
        tr_s.yaws(0, k) = 0.0f;
        tr_s.x(1, k) = static_cast<float>(shift1_pose.pose.position.x) + 0.015f * k;
        tr_s.y(1, k) = static_cast<float>(shift1_pose.pose.position.y) - 0.12f;
        tr_s.yaws(1, k) = -0.30f;
      }
      mppi::models::Path mp_s;
      mp_s.reset(7);
      for (size_t i = 0; i < 7; ++i) {
        mp_s.x(i) = -0.06f + 0.05f * static_cast<float>(i);
        mp_s.y(i) = -0.18f;
        mp_s.yaws(i) = 0.0f;
      }
      geometry_msgs::msg::Pose shifted_goal;
      shifted_goal.position.x = 0.24;
      shifted_goal.position.y = -0.18;
      shifted_goal.orientation.w = 1.0;
      xt::xtensor<float, 1> shift_costs = xt::zeros<float>({2});
      float dt_s = 0.05f;
      mppi::CriticData shift_data = {
        st_s, tr_s, mp_s, shifted_goal, shift_costs, dt_s,
        false, nullptr, nullptr, std::nullopt, std::nullopt};
      tc->score(shift_data);
      EXPECT_GT(shift_costs(0) + shift_costs(1), 0.0f);
      EXPECT_LT(shift_costs(0), shift_costs(1));
    }

    // Tick 2 (spent-budget case): a second 0.30 m local goal shift after redraw budget is spent
    // (redrawCount() == 1 == max_redraws) must NOT receive another replacement and must abort!
    setGlobalToMapTransform(0.48, -0.36);
    auto shift2_pose = pose({-0.30 + 0.48, -0.06 - 0.36, 0.0});
    try {
      controller.computeVelocityCommands(shift2_pose, speed, nullptr);
      FAIL() << "Expected LOCAL_GOAL_SHIFT_EXCEEDED once redraw budget (1/1) is spent";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LOCAL_GOAL_SHIFT_EXCEEDED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // Direct max_redraws = 0 regression on the same local-goal shift:
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 0)})[0].successful);
    setGlobalToMapTransform(0.0, 0.0);
    controller.reset();
    controller.setPlan(global_plan);
    EXPECT_NO_THROW(controller.computeVelocityCommands(init_pose, speed, nullptr));
    ASSERT_TRUE(tc->isLatched());
    setGlobalToMapTransform(0.24, -0.18);
    try {
      controller.computeVelocityCommands(shift1_pose, speed, nullptr);
      FAIL() << "Expected LOCAL_GOAL_SHIFT_EXCEEDED on first shift when max_redraws=0";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=LOCAL_GOAL_SHIFT_EXCEEDED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // (8b) Score-detected gear-flip branch (L2305-2336) through score() with max_redraws = 1:
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 1)})[0].successful);
    setGlobalToMapTransform(0.0, 0.0);
    controller.reset();
    controller.setPlan(early_sc.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(init_pose, speed, nullptr));
    ASSERT_TRUE(tc->isLatched());
    ASSERT_FALSE(tc->latchedCurve().empty());
    EXPECT_LT(tc->latchedCurve().front().x, 0.0f);
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 0);

    // Construct CriticData representing a score()-detected gear flip (path_gear = -1.0 while
    // latched_gear_ = +1.0, without onNewPlan()) at a feasible reverse pose (0.30, 0.06, 0):
    mppi::models::State st;
    st.reset(2, 10);
    st.pose.header.frame_id = "map";
    st.pose.pose.position.x = 0.30;
    st.pose.pose.position.y = 0.06;
    st.pose.pose.orientation.w = 1.0;
    st.speed.linear.x = -0.10f;
    mppi::models::Trajectories tr;
    tr.reset(2, 10);
    for (size_t k = 0; k < 10; ++k) {
      tr.x(0, k) = 0.30f - 0.025f * static_cast<float>(k);
      tr.y(0, k) = 0.06f - 0.005f * static_cast<float>(k);
      tr.yaws(0, k) = 0.0f;
      tr.x(1, k) = 0.30f - 0.015f * static_cast<float>(k);
      tr.y(1, k) = 0.16f;
      tr.yaws(1, k) = 0.30f;
    }
    mppi::models::Path rev_mp;
    rev_mp.reset(7);
    for (size_t i = 0; i < 7; ++i) {
      rev_mp.x(i) = 0.30f - 0.05f * static_cast<float>(i);
      rev_mp.y(i) = 0.0f;
      rev_mp.yaws(i) = 0.0f;
    }
    geometry_msgs::msg::Pose goal_pose;
    goal_pose.orientation.w = 1.0;
    xt::xtensor<float, 1> flip_costs = xt::zeros<float>({2});
    float model_dt = 0.05f;
    mppi::CriticData flip_data = {
      st, tr, rev_mp, goal_pose, flip_costs, model_dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};

    // Tick 1 (score-detected gear flip +1 -> -1 with unused redraw budget 0/1):
    // Must release latch_consumed_, latch a reverse replacement curve, and score rollouts (> 0):
    EXPECT_NO_THROW(tc->score(flip_data));
    EXPECT_TRUE(tc->isLatched())
      << "Permitted score-detected gear-flip redraw (max_redraws=1) must latch replacement curve";
    EXPECT_FALSE(tc->isReferencePathLatched());
    ASSERT_FALSE(tc->latchedCurve().empty());
    EXPECT_GT(tc->latchedCurve().front().x, 0.25f);
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    EXPECT_GT(flip_costs(0) + flip_costs(1), 0.0f);
    EXPECT_LT(flip_costs(0), flip_costs(1));

    // Tick 2 (spent-budget case): flip gear back (-1 -> +1) once redraw budget (1/1) is spent ->
    // must NOT grant another replacement and must throw GEAR_FLIP_WHILE_LATCHED!
    st.pose.pose.position.x = -0.30;
    st.pose.pose.position.y = -0.06;
    st.speed.linear.x = 0.10f;
    mppi::models::Path fwd_mp;
    fwd_mp.reset(7);
    for (size_t i = 0; i < 7; ++i) {
      fwd_mp.x(i) = -0.30f + 0.05f * static_cast<float>(i);
      fwd_mp.y(i) = 0.0f;
      fwd_mp.yaws(i) = 0.0f;
    }
    flip_costs.fill(0.0f);
    mppi::CriticData flip_back_data = {
      st, tr, fwd_mp, goal_pose, flip_costs, model_dt,
      false, nullptr, nullptr, std::nullopt, std::nullopt};
    try {
      tc->score(flip_back_data);
      FAIL() << "Expected GEAR_FLIP_WHILE_LATCHED once redraw budget (1/1) is spent";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=GEAR_FLIP_WHILE_LATCHED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 1);

    // Direct max_redraws = 0 regression on score-detected gear flip:
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 0)})[0].successful);
    controller.reset();
    controller.setPlan(early_sc.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(init_pose, speed, nullptr));
    ASSERT_TRUE(tc->isLatched());
    st.pose.pose.position.x = 0.30;
    st.pose.pose.position.y = 0.06;
    st.speed.linear.x = -0.10f;
    flip_costs.fill(0.0f);
    try {
      tc->score(flip_data);
      FAIL() <<
        "Expected GEAR_FLIP_WHILE_LATCHED on first score-detected gear flip when max_redraws=0";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=GEAR_FLIP_WHILE_LATCHED"),
        std::string::npos) << ex.what();
    }
    EXPECT_FALSE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);

    // (8c) onNewPlan() (setPlan) permitted gear-flip redraw (max_redraws = 1) + spent-budget:
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 1)})[0].successful);
    controller.reset();
    controller.setPlan(early_sc.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(init_pose, speed, nullptr));
    ASSERT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);
    nav_msgs::msg::Path rev_plan = early_sc.message();
    rev_plan.poses.resize(7);
    for (size_t i = 0; i < 7; ++i) {
      rev_plan.poses[i].header.frame_id = "map";
      rev_plan.poses[i].pose.position.x = 0.30 - 0.05 * static_cast<double>(i);
      rev_plan.poses[i].pose.position.y = 0.0;
      rev_plan.poses[i].pose.orientation.w = 1.0;
    }
    controller.setPlan(rev_plan);
    EXPECT_FALSE(tc->isLatched());
    EXPECT_FALSE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    auto rev_pose = pose({0.30, 0.06, 0.0});
    geometry_msgs::msg::Twist rev_speed;
    rev_speed.linear.x = -0.10;
    EXPECT_NO_THROW(controller.computeVelocityCommands(rev_pose, rev_speed, nullptr));
    EXPECT_TRUE(tc->isLatched())
      << "onNewPlan() permitted gear-flip redraw must latch replacement curve on next score()";
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    // Spent-budget second setPlan() gear flip must escalate GEAR_FLIP_ON_SAME_GOAL:
    controller.setPlan(early_sc.message());
    try {
      controller.computeVelocityCommands(init_pose, speed, nullptr);
      FAIL() << "Expected GEAR_FLIP_ON_SAME_GOAL once redraw budget (1/1) is spent";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=GEAR_FLIP_ON_SAME_GOAL"),
        std::string::npos) << ex.what();
    }

    // (8d) Consecutive dynamic set_parameters() calls when max_redraws = 1 must not reset
    // redraw_count_ in addPostCallback and must escalate PARAM_CHANGE_WHILE_LATCHED on 2nd change:
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_redraws", 1),
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_residual_y", 0.025)})[0]
      .successful);
    controller.reset();
    controller.setPlan(early_sc.message());
    EXPECT_NO_THROW(controller.computeVelocityCommands(init_pose, speed, nullptr));
    ASSERT_TRUE(tc->isLatched());
    EXPECT_EQ(tc->redrawCount(), 0);
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_residual_y", 0.030)})[0]
      .successful);
    EXPECT_FALSE(tc->isLatched());
    EXPECT_FALSE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    ASSERT_TRUE(
      parent->set_parameters({
        rclcpp::Parameter("FollowPath.TerminalApproachCritic.max_residual_y", 0.025)})[0]
      .successful);
    EXPECT_TRUE(tc->isLatchConsumed());
    EXPECT_EQ(tc->redrawCount(), 1);
    try {
      controller.computeVelocityCommands(init_pose, speed, nullptr);
      FAIL() << "Expected PARAM_CHANGE_WHILE_LATCHED on 2nd parameter change once budget (1/1) "
        "is spent";
    } catch (const nav2_core::InvalidPath & ex) {
      EXPECT_NE(
        std::string(ex.what()).find("reason=PARAM_CHANGE_WHILE_LATCHED"),
        std::string::npos) << ex.what();
    }
  }
}
}  // namespace

TEST_F(ControllerScenarios, TerminalGoalRegionKeepsGuidanceAndRejectsUnsafeArrivals)
{
  // Reconstruct the finite-runway defect using breadth M8 r1/r4/r7 and M3 log magnitudes.
  // These are controlled straight-terminal geometries, not a claim to replay unrecorded
  // latched curves. In particular the recorded path yaw error is not necessarily goal yaw.
  const std::vector<Pose> arrivals = {
    {-0.0734, 0.0531, 0.3474}, {-0.0445, 0.0542, 0.3220},
    {-0.0046, 0.0356, 0.3177}, {-0.0300, 0.0438, 0.3500}};
  for (const bool mirror : {false, true}) {
    for (const bool reverse : {false, true}) {
      for (size_t case_idx = 0; case_idx < arrivals.size() + 9u; ++case_idx) {
        SCOPED_TRACE(
          std::to_string(mirror) + "/" + std::to_string(reverse) + "/" +
          std::to_string(case_idx));
        Scenario base;
        base.id = "terminal_goal_region";
        base.path = {{-0.25, 0.0, 0.0}, {-0.15, 0.0, 0.0}, {-0.05, 0.0, 0.0}, {0, 0, 0}};
        base.gears = {1, 1, 1};
        base.starts = {{"entry", {-0.24, 0.022, 0.0}}};
        Pose arrival = case_idx < arrivals.size() ? arrivals[case_idx] : arrivals.back();
        const bool lateral_failure = case_idx == arrivals.size();
        const bool unreachable = case_idx == arrivals.size() + 1u;
        const bool blocked = case_idx == arrivals.size() + 2u;
        const bool narrow_corridor = case_idx == arrivals.size() + 3u;
        const bool witness_blocked = case_idx == arrivals.size() + 4u;
        const bool bad_yaw = case_idx == arrivals.size() + 5u;
        const bool huge_radius = case_idx == arrivals.size() + 6u;
        const bool tight_margin = case_idx == arrivals.size() + 7u;
        const bool ratchet_failure = case_idx == arrivals.size() + 8u;
        if (witness_blocked && reverse) {continue;}
        if (lateral_failure) {arrival[1] = 0.09;}
        if (unreachable) {arrival[2] = 0.9;}
        if (narrow_corridor) {arrival = arrivals.front();}
        if (witness_blocked || tight_margin) {arrival = arrivals[2];}
        if (ratchet_failure) {arrival = {-0.196, -0.050, 0.700};}
        // The wider constructed examples need their full steering sagitta. This is not a
        // replay of their recorded latch envelope. The r7 counterexample needs no wider gate.
        if (case_idx == 0u || case_idx == 1u) {base.starts["entry"][1] = 0.04;}
        base.starts["arrival"] = arrival;
        const Scenario sc = base.variant(mirror, reverse);
        grid(sc);
        auto parent = node(true, 500, 0.325, 7, 0.35);
        parent->declare_parameter("FollowPath.TerminalApproachCritic.enabled", true);
        parent->declare_parameter("FollowPath.TerminalApproachCritic.max_redraws", 0);
        parent->declare_parameter(
          "FollowPath.TerminalApproachCritic.min_turning_radius", huge_radius ? 1e38 : 0.56);
        parent->declare_parameter("FollowPath.TerminalApproachCritic.infeasible_policy", "abort");
        parent->declare_parameter("goal_check.xy_goal_tolerance", 0.10);
        parent->declare_parameter("goal_check.yaw_goal_tolerance", 0.25);
        parent->declare_parameter("goal_check.trans_stopped_velocity", 0.04);
        parent->declare_parameter("goal_check.rot_stopped_velocity", 0.12);
        parent->declare_parameter("goal_check.stateful", false);
        nav2_controller::StoppedGoalChecker gc;
        gc.initialize(parent, "goal_check", cm);
        mppi::ParametersHandler params(parent);
        params.start();
        ScenarioOptimizer optimizer;
        optimizer.configure(parent, cm, &params, true);
        const auto path = sc.message();
        const auto goal = path.poses.back().pose;
        geometry_msgs::msg::Twist speed;
        speed.linear.x = reverse ? -0.15 : 0.15;
        optimizer.terminalCritic()->onNewPlan(path);
        ASSERT_NO_THROW(optimizer.evalControl(pose(sc.starts.at("entry")), speed, path, goal, &gc));
        ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
        const auto original_curve = optimizer.terminalCritic()->latchedCurve();
        auto actual = pose(sc.starts.at("arrival"));
        if (bad_yaw) {
          actual.pose.orientation.z = std::numeric_limits<double>::quiet_NaN();
        }
        if (tight_margin) {
          ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
                "FollowPath.TerminalApproachCritic.goal_region_xy_ratio", 0.1)).successful);
          ASSERT_TRUE(optimizer.terminalCritic()->isLatched());
        }
        if (blocked) {
          unsigned int mx, my;
          ASSERT_TRUE(cm->getCostmap()->worldToMap(0.0, 0.0, mx, my));
          cm->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
        }
        if (witness_blocked) {
          Scenario with_obstacle = sc;
          with_obstacle.obstacles = mirror ?
            std::vector<controller_test::Box>{{0.30, -0.22, 0.33, -0.19, 0.1}} :
          std::vector<controller_test::Box>{{0.30, 0.19, 0.33, 0.22, 0.1}};
          grid(with_obstacle);
        }
        if (lateral_failure || unreachable || blocked || narrow_corridor || witness_blocked ||
          bad_yaw || huge_radius || tight_margin || ratchet_failure)
        {
          testing::internal::CaptureStderr();
          EXPECT_THROW(optimizer.evalControl(actual, speed, path, goal, &gc),
            nav2_core::InvalidPath);
          const auto log = testing::internal::GetCapturedStderr();
          const auto message = optimizer.terminalCritic()->lastAbortMessage();
          const std::string reason = blocked ? "LATCHED_CURVE_BLOCKED" :
            (lateral_failure ? "DRIFT_EXCEEDED" : "CURVE_UNJOINABLE");
          EXPECT_NE(message.find("reason=" + reason), std::string::npos) << message;
          const auto pos = message.find("curve_lat_err=");
          ASSERT_NE(pos, std::string::npos);
          // Inspect the failing evaluation, not lastCurveLateralErr() after latch clearing.
          EXPECT_NEAR(std::stod(message.substr(pos + 14u)), std::fabs(arrival[1]), 1e-4);
          const auto log_pos = log.find("curve_lat_err=");
          ASSERT_NE(log_pos, std::string::npos) << log;
          EXPECT_NEAR(std::stod(log.substr(log_pos + 14u)), std::fabs(arrival[1]), 1e-4);
          EXPECT_EQ(log.find("lat_limit=0.0000"), std::string::npos) << log;
          EXPECT_FALSE(optimizer.terminalCritic()->isLatched());
        } else {
          ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, &gc));
          EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
          EXPECT_GT(optimizer.evaluateTerminalCriticOnlyCost(), 0.0f);
          // Walk the bounded physical arc through score(), retaining optimizer state. An
          // admission that merely defers an unavoidable lateral abort must fail this test.
          for (int step = 1; step <= 12; ++step) {
            const double yaw = arrival[2] + (0.18 - arrival[2]) * step / 12.0;
            Pose next = {
              arrival[0] + 0.56 * (std::sin(arrival[2]) - std::sin(yaw)),
              arrival[1] + 0.56 * (std::cos(yaw) - std::cos(arrival[2])), yaw};
            if (mirror) {next[1] = -next[1]; next[2] = -next[2];}
            if (reverse) {next[2] = controller_test::wrap(next[2] + M_PI);}
            ASSERT_NO_THROW(optimizer.evalControl(pose(next), speed, path, goal, &gc));
            EXPECT_TRUE(optimizer.terminalCritic()->isLatched());
          }
          const auto & after = optimizer.terminalCritic()->latchedCurve();
          ASSERT_EQ(after.size(), original_curve.size());
          for (size_t i = 0u; i < after.size(); ++i) {
            EXPECT_FLOAT_EQ(after[i].x, original_curve[i].x);
            EXPECT_FLOAT_EQ(after[i].y, original_curve[i].y);
            EXPECT_FLOAT_EQ(after[i].theta, original_curve[i].theta);
          }
        }
        EXPECT_EQ(optimizer.terminalCritic()->redrawCount(), 0);
        EXPECT_TRUE(optimizer.terminalCritic()->isLatchConsumed());
        optimizer.shutdown();
      }
    }
  }
}

TEST_F(ControllerScenarios, TerminalDiagnosticsPublishFailingEvaluationBeforeClear)
{
  using Diagnostic = nav2_critics_msgs::msg::TerminalApproachDiagnostics;
  for (const std::string mode : {"accepted", "inside_goal", "lateral", "blocked", "witness_blocked",
      "tight_margin", "sample_budget", "entry", "empty_candidates", "pose_admission",
      "stock_mismatch", "capture_blocked"})
  {
    SCOPED_TRACE(mode);
    const bool succeeds = mode == "accepted" || mode == "inside_goal";
    const bool entry = mode == "entry" || mode == "empty_candidates" || mode == "pose_admission" ||
      mode == "capture_blocked";
    Scenario sc;
    sc.id = "terminal_diagnostics";
    sc.path = {{-0.25, 0, 0}, {-0.15, 0, 0}, {-0.05, 0, 0}, {0, 0, 0}};
    sc.gears = {1, 1, 1};
    grid(sc);
    auto parent = node(true, 64, 0.325, 7, 0.35);
    parent->declare_parameter("FollowPath.TerminalApproachCritic.enabled", true);
    parent->declare_parameter("FollowPath.TerminalApproachCritic.infeasible_policy", "abort");
    parent->declare_parameter("FollowPath.TerminalApproachCritic.min_turning_radius",
      mode == "sample_budget" ? 1e38 : 0.56);
    mppi::ParametersHandler params(parent);
    params.start();
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    auto observer = std::make_shared<rclcpp::Node>("terminal_diagnostic_observer");
    auto sub = observer->create_subscription<Diagnostic>(
      "/scenario_controller/terminal_approach_diagnostics", rclcpp::QoS(100).reliable(),
      [](Diagnostic::ConstSharedPtr) {});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (sub->get_publisher_count() == 0u && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_GT(sub->get_publisher_count(), 0u);
    auto receive = [&]() {
        std::vector<Diagnostic> messages;
        auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (std::chrono::steady_clock::now() < end) {
          Diagnostic message;
          rclcpp::MessageInfo info;
          if (sub->take(message, info)) {
            messages.push_back(message);
          } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        return messages;
      };
    const auto path = sc.message();
    const auto goal = path.poses.back().pose;
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.15;
    auto * critic = optimizer.terminalCritic();
    critic->onNewPlan(path);
    if (!entry) {
      ASSERT_NO_THROW(optimizer.evalControl(pose({-0.24, 0.022, 0}), speed, path, goal, nullptr));
      ASSERT_TRUE(critic->isLatched());
    }
    const auto initial = receive();
    if (!entry) {
      ASSERT_EQ(initial.size(), 3u);
      EXPECT_TRUE(initial.front().latched);
      EXPECT_FALSE(initial.front().curve.empty());
      EXPECT_EQ(initial.front().event, "EVALUATED");
      for (size_t i = 0; i < initial.size(); ++i) {
        EXPECT_EQ(initial[i].evaluation_id, initial[0].evaluation_id + i);
        EXPECT_EQ(initial[i].sequence, initial[0].sequence + i);
        EXPECT_EQ(initial[i].feasibility_evaluated, i == 0u);
        EXPECT_EQ(initial[i].reuse_obstacles_cache, i != 0u);
        if (i != 0u) {
          EXPECT_TRUE(initial[i].checks.empty());
          EXPECT_TRUE(initial[i].curve.empty());
        }
      }
    }
    Pose arrival{-0.0046, 0.0356, 0.3177};
    if (mode == "inside_goal") {arrival = {0.09, 0.0, 0.20};}
    if (mode == "lateral") {arrival[1] = 0.09;}
    if (mode == "entry") {arrival = {-0.14, 0, 0.9};}
    if (mode == "empty_candidates") {arrival = {-0.20, 0.10, 1.2};}
    if (mode == "pose_admission") {arrival = {0.20, 0, 0.9};}
    if (mode == "capture_blocked") {arrival = {-0.30, 0.04, 0.0};}
    if (mode == "blocked" || mode == "capture_blocked") {
      unsigned int mx, my;
      ASSERT_TRUE(cm->getCostmap()->worldToMap(0, 0, mx, my));
      cm->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
    if (mode == "witness_blocked") {
      sc.obstacles = {{0.30, 0.19, 0.33, 0.22, 0.1}};
      grid(sc);
    }
    if (mode == "tight_margin") {
      ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
            "FollowPath.TerminalApproachCritic.goal_region_xy_ratio", 0.1)).successful);
    }
    if (mode == "stock_mismatch") {
      ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
            "FollowPath.PathAlignCritic.threshold_to_consider", 0.01)).successful);
    }
    if (succeeds) {
      ASSERT_NO_THROW(optimizer.evalControl(pose(arrival), speed, path, goal, nullptr));
    } else {
      EXPECT_THROW(optimizer.evalControl(pose(arrival), speed, path, goal, nullptr),
        nav2_core::InvalidPath);
    }
    const auto events = receive();
    ASSERT_EQ(events.size(), succeeds ? 3u : 1u);
    const auto & event = events.front();
    EXPECT_EQ(event.event, succeeds ? "EVALUATED" : "ABORT");
    EXPECT_EQ(event.schema_version, 1u);
    EXPECT_FALSE(event.instance_id.empty());
    EXPECT_EQ(event.header.frame_id, path.header.frame_id);
    EXPECT_TRUE(event.evaluation_context_valid);
    EXPECT_DOUBLE_EQ(event.robot_pose.position.x, arrival[0]);
    EXPECT_DOUBLE_EQ(event.velocity.linear.x, 0.15);
    EXPECT_FALSE(event.checks_truncated);
    EXPECT_FALSE(event.curve_truncated);
    EXPECT_EQ(event.redraw_count, 0);
    EXPECT_EQ(event.replan_count, 0);
    if (!initial.empty()) {
      EXPECT_GT(event.sequence, initial.back().sequence);
      EXPECT_EQ(event.goal_id, initial.front().goal_id);
      EXPECT_EQ(event.latch_id, initial.front().latch_id);
    }
    if (succeeds) {
      EXPECT_EQ(event.goal_region_result, mode == "accepted" ? "REACHED_INTERIOR" :
        "MISSED_INTERIOR");
      if (mode == "inside_goal") {
        const auto check = std::find_if(event.checks.begin(), event.checks.end(),
            [](const auto & c) {return c.name == "goal/already_inside";});
        ASSERT_NE(check, event.checks.end());
        EXPECT_TRUE(check->passed);
        EXPECT_TRUE(event.reason.empty());
      }
      EXPECT_TRUE(critic->isLatched());
      for (size_t i = 1; i < events.size(); ++i) {
        EXPECT_EQ(events[i].event, "EVALUATED");
        EXPECT_EQ(events[i].evaluation_id, event.evaluation_id + i);
        EXPECT_EQ(events[i].sequence, event.sequence + i);
        EXPECT_FALSE(events[i].feasibility_evaluated);
        EXPECT_TRUE(events[i].reuse_obstacles_cache);
        EXPECT_TRUE(events[i].checks.empty());
        EXPECT_TRUE(events[i].curve.empty());
        EXPECT_EQ(events[i].goal_region_result, "NOT_EVALUATED");
      }
    } else if (mode == "stock_mismatch") {
      EXPECT_EQ(event.reason, "STOCK_THRESHOLD_MISMATCH");
      EXPECT_FALSE(event.goal_tolerance_valid);
      ASSERT_EQ(event.checks.size(), 1u);
      EXPECT_EQ(event.checks.front().name, "stock/threshold_difference_le");
      EXPECT_GT(event.checks.front().value, event.checks.front().limit);
    } else if (entry) {
      EXPECT_EQ(event.reason, "ENTRY_INFEASIBLE");
      EXPECT_FALSE(event.latched);
      EXPECT_FALSE(event.before_clear);
      EXPECT_TRUE(event.curve.empty());
      EXPECT_TRUE(event.curve_frame.empty());
      EXPECT_FALSE(event.checks.empty());
      if (mode == "capture_blocked") {
        const auto accepted = std::find_if(event.checks.begin(), event.checks.end(),
            [](const auto & c) {return c.name.find("/accepted_geometric") != std::string::npos;});
        const auto blocked = std::find_if(event.checks.begin(), event.checks.end(),
            [](const auto & c) {
              const bool capture = c.name.find("capture/") == 0u;
              return capture && c.name.find("/collision_free") != std::string::npos;
            });
        ASSERT_NE(accepted, event.checks.end());
        ASSERT_NE(blocked, event.checks.end());
        EXPECT_LT(accepted - event.checks.begin(), blocked - event.checks.begin());
        EXPECT_TRUE(accepted->passed);
        EXPECT_FALSE(blocked->passed);
      } else if (mode != "entry") {
        size_t capture_checks = 0u;
        for (const auto & check : event.checks) {
          if (check.name.find("capture/") != 0u) {continue;}
          ++capture_checks;
          EXPECT_FALSE(check.passed) << check.name;
          if (check.name.find("R < min_turning_r") != std::string::npos) {
            EXPECT_DOUBLE_EQ(check.limit, static_cast<double>(0.56f));
          } else {
            // Bit check survives the production -ffast-math compiler policy.
            uint64_t bits;
            std::memcpy(&bits, &check.value, sizeof(bits));
            EXPECT_EQ(bits & 0x7ff0000000000000ULL, 0x7ff0000000000000ULL) << check.name;
            EXPECT_NE(bits & 0x000fffffffffffffULL, 0u) << check.name;
            std::memcpy(&bits, &check.limit, sizeof(bits));
            EXPECT_EQ(bits & 0x7ff0000000000000ULL, 0x7ff0000000000000ULL) << check.name;
            EXPECT_NE(bits & 0x000fffffffffffffULL, 0u) << check.name;
          }
        }
        EXPECT_GE(capture_checks, mode == "pose_admission" ? 1u : 4u);
      }
    } else {
      EXPECT_FALSE(critic->isLatched());
      EXPECT_TRUE(event.latched);  // Failing state survived clearActiveCurve().
      EXPECT_TRUE(event.before_clear);
      EXPECT_FALSE(event.curve.empty());
      EXPECT_GT(event.lateral_limit, 0.04);
      EXPECT_NEAR(event.lateral_error, std::fabs(arrival[1]), 1e-5);
      EXPECT_EQ(event.reason, mode == "lateral" ? "DRIFT_EXCEEDED" :
        (mode == "blocked" ? "LATCHED_CURVE_BLOCKED" : "CURVE_UNJOINABLE"));
      if (mode == "tight_margin") {
        EXPECT_EQ(event.goal_region_result, "MISSED_INTERIOR");
        EXPECT_TRUE(event.witness_sample_valid);
        EXPECT_EQ(event.witness_sample_index + 1u, event.witness_sample_count);
      }
      if (mode == "witness_blocked") {
        EXPECT_EQ(event.goal_region_result, "FOOTPRINT_BLOCKED");
        EXPECT_TRUE(event.collision_check_failed);
        EXPECT_FALSE(event.collision_reason.empty());
      }
      if (mode == "sample_budget") {
        EXPECT_EQ(event.goal_region_result, "SAMPLE_BUDGET");
        EXPECT_FALSE(event.witness_sample_valid);
      }
      if (mode == "lateral") {
        EXPECT_EQ(event.goal_region_result, "LATERAL_ENVELOPE");
        const auto check = std::find_if(event.checks.begin(), event.checks.end(),
            [](const auto & c) {return c.name == "curve/lateral_le";});
        ASSERT_NE(check, event.checks.end());
        EXPECT_FALSE(check->passed);
        EXPECT_GT(check->value, check->limit);
      }
    }
    critic->reset();
    const auto resets = receive();
    ASSERT_EQ(resets.size(), 1u);
    EXPECT_EQ(resets.front().event, "RESET");
    EXPECT_EQ(resets.front().reason, "TASK_RESET");
    EXPECT_EQ(resets.front().goal_region_result, "NOT_EVALUATED");
    EXPECT_FALSE(resets.front().evaluation_context_valid);
    const bool had_latch = succeeds || mode == "stock_mismatch";
    EXPECT_EQ(resets.front().before_clear, had_latch);
    EXPECT_EQ(resets.front().curve.empty(), !had_latch);
    EXPECT_EQ(resets.front().curve_frame.empty(), !had_latch);
    grid(sc);
    critic->onNewPlan(path);
    if (mode == "accepted") {
      ASSERT_NO_THROW(optimizer.evalControl(pose({-0.24, 0.022, 0}), speed, path, goal, nullptr));
      auto next = receive();
      ASSERT_FALSE(next.empty());
      EXPECT_GT(next.front().task_id, event.task_id);
      EXPECT_GT(next.front().plan_id, event.plan_id);
      EXPECT_TRUE(next.front().reason.empty());
      EXPECT_EQ(next.front().goal_region_result, "NOT_EVALUATED");
      critic->on_deactivate();
      optimizer.evalControl(pose({-0.24, 0.022, 0}), speed, path, goal, nullptr);
      EXPECT_TRUE(receive().empty());
      critic->on_activate();
      receive();  // Activation clears the old latch; lifecycle cases are checked below.
      ASSERT_NO_THROW(optimizer.evalControl(pose({-0.24, 0.022, 0}), speed, path, goal, nullptr));
      ASSERT_TRUE(critic->isLatched());
      receive();
      ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
            "FollowPath.TerminalApproachCritic.enabled", false)).successful);
      EXPECT_TRUE(receive().empty());
      optimizer.evalControl(pose({-0.24, 0.022, 0}), speed, path, goal, nullptr);
      EXPECT_TRUE(receive().empty());
    }
    optimizer.shutdown();
  }
}

TEST_F(ControllerScenarios, TerminalDiagnosticsSeparateTransitionsFromFinalState)
{
  using Diagnostic = nav2_critics_msgs::msg::TerminalApproachDiagnostics;
  for (const std::string mode : {"new_goal", "empty_plan", "same_goal", "plan_gear_flip",
      "in_place_redraw", "parameter", "disabled", "reconfigure",
      "redraw_failure", "gear_redraw_failure", "fallback_shift", "replan_shift", "lost_goal",
      "entry_replan", "new_goal_after_replan"})
  {
    SCOPED_TRACE(mode);
    Scenario sc;
    sc.id = "terminal_diagnostic_transitions";
    sc.path = {{-0.25, 0, 0}, {-0.15, 0, 0}, {-0.05, 0, 0}, {0, 0, 0}};
    sc.gears = {1, 1, 1};
    if (mode == "in_place_redraw") {
      sc = atlas->at("curved_capture");
    }
    grid(sc);
    auto parent = node(true, 64, 0.325, 7, 0.35);
    parent->declare_parameter("FollowPath.TerminalApproachCritic.enabled", true);
    const bool redraw = mode == "redraw_failure" || mode == "gear_redraw_failure" ||
      mode == "in_place_redraw" || mode == "plan_gear_flip";
    parent->declare_parameter("FollowPath.TerminalApproachCritic.max_redraws", redraw ? 1 : 0);
    parent->declare_parameter("FollowPath.TerminalApproachCritic.infeasible_policy",
      mode == "fallback_shift" ? "fallback" :
      ((mode == "replan_shift" || mode == "entry_replan" || mode == "new_goal_after_replan") ?
      "replan_signal" : "abort"));
    mppi::ParametersHandler params(parent);
    params.start();
    ScenarioOptimizer optimizer;
    optimizer.configure(parent, cm, &params, true);
    auto observer = std::make_shared<rclcpp::Node>("terminal_transition_observer");
    auto sub = observer->create_subscription<Diagnostic>(
      "/scenario_controller/terminal_approach_diagnostics", rclcpp::QoS(100).reliable(),
      [](Diagnostic::ConstSharedPtr) {});
    auto connect = [&]() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (sub->get_publisher_count() == 0u && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return sub->get_publisher_count() > 0u;
      };
    ASSERT_TRUE(connect());
    auto receive = [&]() {
        std::vector<Diagnostic> messages;
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (std::chrono::steady_clock::now() < end) {
          Diagnostic message;
          rclcpp::MessageInfo info;
          if (sub->take(message, info)) {messages.push_back(message);} else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
        }
        return messages;
      };
    auto path = sc.message();
    auto goal = path.poses.back().pose;
    auto actual = pose({-0.24, 0.022, 0});
    auto buffer = std::make_shared<tf2_ros::Buffer>(parent->get_clock());
    ScenarioPathHandler handler;
    if (mode == "in_place_redraw") {
      actual = pose(sc.starts.at("inside"));
      handler.initialize(parent, "FollowPath", cm, buffer, &params);
      handler.setPath(path);
      path = handler.transformPath(actual);
    }
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.15;
    auto * critic = optimizer.terminalCritic();
    critic->onNewPlan(path);
    if (mode == "entry_replan" || mode == "new_goal_after_replan") {
      ASSERT_NO_THROW(optimizer.evalControl(pose({-0.50, 0.10, 1.2}), speed, path, goal, nullptr));
      const auto entry_events = receive();
      ASSERT_EQ(entry_events.size(), 3u);
      const auto & entry_failure = entry_events.front();
      EXPECT_EQ(entry_failure.event, "EVALUATED");
      EXPECT_EQ(entry_failure.reason, "ENTRY_INFEASIBLE");
      EXPECT_FALSE(entry_failure.latched);
      EXPECT_FALSE(entry_failure.before_clear);
      EXPECT_TRUE(entry_failure.latch_consumed);
      EXPECT_TRUE(entry_failure.replan_allowance_consumed);
      EXPECT_TRUE(entry_failure.replan_pending);
      EXPECT_EQ(entry_failure.replan_count, 1);
      EXPECT_TRUE(entry_failure.curve.empty());
      EXPECT_TRUE(entry_failure.curve_frame.empty());
      EXPECT_TRUE(entry_failure.feasibility_evaluated);
      ASSERT_NO_THROW(optimizer.evalControl(pose({-0.50, 0.10, 1.2}), speed, path, goal, nullptr));
      const auto pending = receive();
      ASSERT_EQ(pending.size(), 3u);
      for (const auto & event : pending) {
        EXPECT_FALSE(event.feasibility_evaluated);
        EXPECT_TRUE(event.replan_pending);
        EXPECT_EQ(event.replan_count, 1);
      }
      if (mode == "entry_replan") {
        EXPECT_THROW(optimizer.evalControl(pose({-0.30, 0.10, 1.2}), speed, path, goal, nullptr),
          nav2_core::InvalidPath);
        const auto cutoff = receive();
        ASSERT_EQ(cutoff.size(), 1u);
        EXPECT_EQ(cutoff.front().reason, "REPLAN_NOT_READY_BY_CUTOFF");
        EXPECT_FALSE(cutoff.front().feasibility_evaluated);
        optimizer.shutdown();
        continue;
      }
      // Accept the same-goal replacement with its one replan allowance already consumed.
      critic->onNewPlan(path);
    }
    ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
    ASSERT_TRUE(critic->isLatched());
    const auto initial = receive();
    ASSERT_EQ(initial.size(), 3u);

    if (mode == "empty_plan" || mode == "same_goal" || mode == "plan_gear_flip") {
      if (mode == "plan_gear_flip") {
        for (auto & p : path.poses) {
          p.pose.position.x = -p.pose.position.x;
        }
        actual = pose({0.24, 0.022, 0});
        speed.linear.x = -0.15;
      }
      critic->onNewPlan(mode == "empty_plan" ? nav_msgs::msg::Path{} : path);
      const auto transition = receive();
      if (mode == "same_goal") {
        EXPECT_TRUE(transition.empty());
      } else {
        ASSERT_EQ(transition.size(), 1u);
        const auto & outgoing = transition.front();
        EXPECT_EQ(outgoing.event, mode == "empty_plan" ? "RESET" : "CURVE_CLEARED");
        EXPECT_EQ(outgoing.reason, mode == "empty_plan" ? "TASK_RESET" : "GEAR_FLIP_ON_SAME_GOAL");
        EXPECT_EQ(outgoing.plan_id, initial.front().plan_id);
        EXPECT_EQ(outgoing.goal_id, initial.front().goal_id);
        EXPECT_EQ(outgoing.task_id, initial.front().task_id);
        EXPECT_EQ(outgoing.latch_id, initial.front().latch_id);
        EXPECT_EQ(outgoing.redraw_count, 0);
        EXPECT_TRUE(outgoing.before_clear);
        EXPECT_EQ(outgoing.curve, initial.front().curve);
      }
      if (mode == "empty_plan") {
        critic->onNewPlan(path);
      }
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      const auto replacement = receive();
      ASSERT_EQ(replacement.size(), 3u);
      EXPECT_EQ(replacement.front().plan_id, initial.front().plan_id +
        (mode == "empty_plan" ? 2u : 1u));
      EXPECT_EQ(replacement.front().goal_id, initial.front().goal_id +
        (mode == "empty_plan" ? 1u : 0u));
      EXPECT_EQ(replacement.front().task_id, initial.front().task_id +
        (mode == "empty_plan" ? 1u : 0u));
      EXPECT_EQ(replacement.front().latch_id, initial.front().latch_id +
        (mode == "same_goal" ? 0u : 1u));
      EXPECT_EQ(replacement.front().redraw_count, mode == "plan_gear_flip" ? 1 : 0);
      EXPECT_TRUE(replacement.front().latched);
      optimizer.shutdown();
      continue;
    }
    if (mode == "in_place_redraw") {
      actual.pose.position.y -= 0.055;
      path = handler.transformPath(actual);
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      const auto replacement = receive();
      ASSERT_EQ(replacement.size(), 3u);
      const auto & event = replacement.front();
      EXPECT_EQ(event.event, "EVALUATED");
      EXPECT_FALSE(event.before_clear);
      EXPECT_TRUE(event.latched);
      EXPECT_TRUE(event.feasibility_evaluated);
      EXPECT_EQ(event.plan_id, initial.front().plan_id);
      EXPECT_EQ(event.goal_id, initial.front().goal_id);
      EXPECT_EQ(event.latch_id, initial.front().latch_id + 1u);
      EXPECT_EQ(event.redraw_count, 1);
      EXPECT_EQ(event.curve_frame, actual.header.frame_id);
      ASSERT_FALSE(event.curve.empty());
      EXPECT_NE(event.curve, initial.front().curve);
      EXPECT_TRUE(std::any_of(event.checks.begin(), event.checks.end(), [](const auto & check) {
          return check.name == "curve/lateral_le" && !check.passed;
        }));
      for (size_t i = 1; i < replacement.size(); ++i) {
        EXPECT_EQ(replacement[i].event, "EVALUATED");
        EXPECT_EQ(replacement[i].latch_id, event.latch_id);
        EXPECT_EQ(replacement[i].sequence, event.sequence + i);
        EXPECT_FALSE(replacement[i].feasibility_evaluated);
        EXPECT_TRUE(replacement[i].curve.empty());
      }
      optimizer.shutdown();
      continue;
    }

    if (mode == "disabled") {
      ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
            "FollowPath.TerminalApproachCritic.enabled", false)).successful);
      EXPECT_TRUE(receive().empty());
      critic->reset();
      EXPECT_TRUE(receive().empty());
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      EXPECT_TRUE(receive().empty());
      optimizer.shutdown();
      continue;
    }
    if (mode == "reconfigure") {
      critic->on_deactivate();
      critic->on_cleanup();
      critic->initialize();
      critic->on_activate();
      ASSERT_TRUE(connect());
      EXPECT_TRUE(receive().empty());
      critic->onNewPlan(path);
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      const auto fresh = receive();
      ASSERT_EQ(fresh.size(), 3u);
      EXPECT_NE(fresh.front().instance_id, initial.front().instance_id);
      EXPECT_EQ(fresh.front().sequence, 1u);
      EXPECT_EQ(fresh.front().evaluation_id, 1u);
      EXPECT_EQ(fresh.front().task_id, 0u);
      EXPECT_EQ(fresh.front().plan_id, 1u);
      EXPECT_EQ(fresh.front().goal_id, 1u);
      EXPECT_EQ(fresh.front().latch_id, 1u);
      optimizer.shutdown();
      continue;
    }

    const bool new_goal = mode == "new_goal" || mode == "new_goal_after_replan";
    const bool shifted = new_goal || mode == "redraw_failure" ||
      mode == "fallback_shift" || mode == "replan_shift";
    if (shifted) {
      for (auto & p : path.poses) {
        p.pose.position.x += 0.30;
      }
      goal = path.poses.back().pose;
      actual = pose(redraw ? Pose{0.16, 0, 0.9} : Pose{-0.20, 0.022, 0});
    }
    if (mode == "redraw_failure") {
      path.header.frame_id = actual.header.frame_id = "odom";
    }
    if (mode == "gear_redraw_failure") {
      for (auto & p : path.poses) {
        p.pose.position.x = -p.pose.position.x;
      }
      actual = pose({0.14, 0, 0.9});
    }
    if (new_goal) {
      critic->onNewPlan(path);
    } else if (mode == "parameter") {
      ASSERT_TRUE(parent->set_parameter(rclcpp::Parameter(
            "FollowPath.TerminalApproachCritic.max_residual_y", 0.03)).successful);
    } else {
      if (mode == "lost_goal") {
        for (auto & p : path.poses) {
          p.pose.position.x -= 0.15;
        }
      }
      if (redraw || mode == "lost_goal") {
        EXPECT_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr),
          nav2_core::InvalidPath);
      } else {
        ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      }
    }
    const auto events = receive();
    ASSERT_FALSE(events.empty());
    const auto & cleared = events.front();
    EXPECT_EQ(cleared.event, "CURVE_CLEARED");
    EXPECT_EQ(cleared.reason, new_goal ? "NEW_GOAL" :
      (mode == "parameter" ? "PARAM_CHANGE_WHILE_LATCHED" :
      (mode == "gear_redraw_failure" ? "GEAR_FLIP_WHILE_LATCHED" :
      (mode == "lost_goal" ? "LATCHED_GOAL_LOST_INSIDE_CUTOFF" : "LOCAL_GOAL_SHIFT_EXCEEDED"))));
    EXPECT_TRUE(cleared.latched);
    EXPECT_TRUE(cleared.before_clear);
    EXPECT_FALSE(cleared.evaluation_context_valid);
    EXPECT_FALSE(cleared.goal_tolerance_valid);
    EXPECT_EQ(cleared.goal_region_result, "NOT_EVALUATED");
    EXPECT_FALSE(cleared.curve.empty());
    EXPECT_EQ(cleared.curve_frame, "map");
    EXPECT_EQ(cleared.sequence, initial.back().sequence + 1u);
    EXPECT_EQ(cleared.latch_id, initial.front().latch_id);
    EXPECT_EQ(cleared.plan_id, initial.front().plan_id);
    EXPECT_EQ(cleared.goal_id, initial.front().goal_id);
    EXPECT_EQ(cleared.redraw_count, initial.front().redraw_count);
    EXPECT_EQ(cleared.replan_count, initial.front().replan_count);
    EXPECT_EQ(cleared.replan_allowance_consumed, initial.front().replan_allowance_consumed);
    for (size_t i = 1; i < events.size(); ++i) {
      EXPECT_EQ(events[i].sequence, cleared.sequence + i);
    }
    if (redraw) {
      ASSERT_EQ(events.size(), 2u);
      const auto & failure = events.back();
      EXPECT_EQ(failure.event, "ABORT");
      EXPECT_EQ(failure.reason, "ENTRY_INFEASIBLE");
      EXPECT_EQ(failure.redraw_count, 1);
      EXPECT_DOUBLE_EQ(failure.gear, mode == "gear_redraw_failure" ? -1.0 : 1.0);
      EXPECT_FALSE(failure.latched);
      EXPECT_FALSE(failure.before_clear);
      EXPECT_TRUE(failure.curve.empty());
      EXPECT_TRUE(failure.curve_frame.empty());
      EXPECT_EQ(failure.header.frame_id, actual.header.frame_id);
      EXPECT_TRUE(failure.evaluation_context_valid);
      EXPECT_TRUE(failure.feasibility_evaluated);
    } else if (mode == "fallback_shift" || mode == "replan_shift") {
      ASSERT_EQ(events.size(), 4u);
      const auto & final = events[1];
      EXPECT_EQ(final.event, "EVALUATED");
      EXPECT_EQ(final.reason, "LOCAL_GOAL_SHIFT_EXCEEDED");
      EXPECT_FALSE(final.latched);
      EXPECT_FALSE(final.before_clear);
      EXPECT_EQ(final.replan_count, mode == "replan_shift" ? 1 : 0);
      EXPECT_EQ(final.replan_pending, mode == "replan_shift");
      EXPECT_TRUE(final.curve_frame.empty());
      EXPECT_FALSE(final.feasibility_evaluated);
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      const auto waiting = receive();
      ASSERT_EQ(waiting.size(), 3u);
      for (const auto & event : waiting) {
        EXPECT_FALSE(event.feasibility_evaluated);
        EXPECT_FALSE(event.latched);
        EXPECT_TRUE(event.latch_consumed);
      }
    } else if (mode == "lost_goal") {
      ASSERT_EQ(events.size(), 2u);
      EXPECT_EQ(events.back().event, "ABORT");
      EXPECT_EQ(events.back().reason, "LATCHED_GOAL_LOST_INSIDE_CUTOFF");
      EXPECT_FALSE(events.back().feasibility_evaluated);
      EXPECT_FALSE(events.back().goal_tolerance_valid);
    } else {
      ASSERT_EQ(events.size(), 1u);
    }
    if (new_goal) {
      if (mode == "new_goal_after_replan") {
        EXPECT_EQ(cleared.replan_count, 1);
        EXPECT_TRUE(cleared.replan_allowance_consumed);
      }
      actual = pose({0.06, 0.022, 0});
      ASSERT_NO_THROW(optimizer.evalControl(actual, speed, path, goal, nullptr));
      const auto replacement = receive();
      ASSERT_EQ(replacement.size(), 3u);
      EXPECT_EQ(replacement.front().plan_id, initial.front().plan_id + 1u);
      EXPECT_EQ(replacement.front().goal_id, initial.front().goal_id + 1u);
      EXPECT_EQ(replacement.front().latch_id, initial.front().latch_id + 1u);
      EXPECT_TRUE(replacement.front().latched);
      EXPECT_EQ(replacement.front().replan_count, 0);
      EXPECT_FALSE(replacement.front().replan_allowance_consumed);
      optimizer.shutdown();
      continue;
    }
    critic->reset();
    const auto reset = receive();
    ASSERT_EQ(reset.size(), 1u);
    EXPECT_EQ(reset.front().event, "RESET");
    EXPECT_EQ(reset.front().reason, "TASK_RESET");
    EXPECT_EQ(reset.front().goal_region_result, "NOT_EVALUATED");
    EXPECT_FALSE(reset.front().before_clear);
    EXPECT_TRUE(reset.front().curve.empty());
    EXPECT_TRUE(reset.front().curve_frame.empty());
    optimizer.shutdown();
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
