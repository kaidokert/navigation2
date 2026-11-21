// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Optional MPPI diagnostics (FollowPath.trace_cycles / trace_rejections): what they publish, and
// that turning them on does not change any command.

#include <chrono>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_critics_msgs/msg/mppi_cycle.hpp"
#include "nav2_critics_msgs/msg/mppi_rejection.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/optimizer.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{
using nav2_critics_msgs::msg::MppiCycle;
using nav2_critics_msgs::msg::MppiRejection;

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

class TraceOptimizer : public mppi::Optimizer
{
public:
  void configure(
    rclcpp_lifecycle::LifecycleNode::SharedPtr node,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm, mppi::ParametersHandler * params,
    bool with_critics)
  {
    initialize(node, "FollowPath", cm, params);
    auto & critics = critic_manager_.*CriticAccess::critics();
    auto & names = critic_manager_.*CriticAccess::names();
    critics.clear();
    names.clear();
    if (with_critics) {
      critics.emplace_back(std::make_unique<mppi::critics::PathFollowCritic>());
      critics.emplace_back(std::make_unique<mppi::critics::ObstaclesCritic>());
      names = {"PathFollowCritic", "ObstaclesCritic"};
      for (size_t i = 0; i < critics.size(); ++i) {
        critics[i]->on_configure(node, "FollowPath", "FollowPath." + names[i], cm, params);
      }
    }
    on_activate();
    freezeNoise();
  }
  // Frozen noise bank, identical for every optimizer built here. A fallback reset draws a new
  // random bank, so comparisons refreeze it before every cycle.
  void freezeNoise()
  {
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
  void stepNoise()
  {
    auto & vx = noise_generator_.*NoiseAccess::vx();
    auto & wz = noise_generator_.*NoiseAccess::wz();
    for (size_t i = 0; i < vx.shape(0); ++i) {
      for (size_t j = 0; j < vx.shape(1); ++j) {
        // The fast rows stall at index 1 (index 0 is the measured speed), so the shifted
        // and unshifted forecasts of a fast row collide at different indices.
        vx(i, j) = i % 4 == 0 ? -0.45f : (j == 1 ? -0.3f : 0.2f);
        wz(i, j) = 0.0f;
      }
    }
  }
  void warm(float vx, float wz)
  {
    control_sequence_.vx.fill(vx);
    control_sequence_.wz.fill(wz);
  }
  const auto & tape() const {return control_sequence_;}
  const auto & history() const {return control_history_;}
  bool productionShift() const {return settings_.shift_control_sequence;}
  bool checkWith(const mppi::models::ControlSequence & s, bool shift, CollisionReport * r) const
  {
    return sequenceCollisionFree(s, shift, r);
  }
  // The production final check, exactly as getValidatedControlFromSequenceAsTwist calls it.
  bool production(const mppi::models::ControlSequence & s)
  {
    const auto keep = control_sequence_;
    control_sequence_ = s;
    const bool ok = controlSequenceCollisionFree();
    control_sequence_ = keep;
    return ok;
  }
  // Pose and speed the checks read, without running the optimizer.
  void setState(const geometry_msgs::msg::PoseStamped & pose, double vx)
  {
    state_.pose = pose;
    state_.speed = geometry_msgs::msg::Twist{};
    state_.speed.linear.x = vx;
  }
  using Optimizer::CollisionReport;
};

class TraceTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>("trace_map", "", "trace_map", true);
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    cm->set_parameter(rclcpp::Parameter("global_frame", "map"));
    cm->set_parameter(rclcpp::Parameter("footprint", std::string(
        "[[0.276,0.135],[0.276,-0.135],[-0.050,-0.135],[-0.050,0.135]]")));
    cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.012));
    ASSERT_EQ(cm->on_configure(rclcpp_lifecycle::State{}), nav2_util::CallbackReturn::SUCCESS);
    cm->pause();
  }
  static void TearDownTestSuite()
  {
    cm->on_cleanup(rclcpp_lifecycle::State{});
    cm.reset();
  }
  void SetUp() override
  {
    cm->getLayeredCostmap()->resizeMap(200, 200, 0.02, -2.0, -2.0);
    auto * g = cm->getCostmap();
    std::fill(g->getCharMap(), g->getCharMap() + 40000, 0);
  }
  void wall(double x0, double y0, double x1, double y1)
  {
    auto * g = cm->getCostmap();
    for (double x = x0; x <= x1; x += 0.01) {
      for (double y = y0; y <= y1; y += 0.01) {
        unsigned int mx, my;
        if (g->worldToMap(x, y, mx, my)) {g->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);}
      }
    }
  }
  static rclcpp_lifecycle::LifecycleNode::SharedPtr node(
    const std::string & name, bool trace, bool critics_params = true)
  {
    rclcpp::NodeOptions options;
    std::vector<rclcpp::Parameter> p = {
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
      rclcpp::Parameter("FollowPath.AckermannConstraints.min_turning_r", 0.325),
      rclcpp::Parameter("FollowPath.trace_cycles", trace),
      rclcpp::Parameter("FollowPath.trace_rejections", trace)};
    if (!critics_params) {
      // Critic-free fixtures: no control-cost term either, so the update is the plain
      // average of the frozen bank and a forced rejection stays forced.
      p.push_back(rclcpp::Parameter("FollowPath.gamma", 0.0));
    }
    if (critics_params) {
      p.push_back(rclcpp::Parameter("FollowPath.PathFollowCritic.cost_weight", 66.05));
      p.push_back(rclcpp::Parameter("FollowPath.ObstaclesCritic.consider_footprint", true));
      p.push_back(
        rclcpp::Parameter("FollowPath.ObstaclesCritic.collision_check_footprint_only", true));
    }
    options.parameter_overrides(p);
    return std::make_shared<rclcpp_lifecycle::LifecycleNode>(name, options);
  }
  static geometry_msgs::msg::PoseStamped pose(double x, double y, double yaw)
  {
    geometry_msgs::msg::PoseStamped p;
    p.header.frame_id = "map";
    p.pose.position.x = x;
    p.pose.position.y = y;
    p.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(yaw);
    return p;
  }
  static nav_msgs::msg::Path straightPath(double length)
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = "map";
    for (double x = 0.0; x <= length + 1e-9; x += 0.05) {path.poses.push_back(pose(x, 0.0, 0.0));}
    return path;
  }
  // Exact geometric test, independent of the optimizer: does the padded rectangle at (x, y, yaw)
  // contain the centre of any lethal cell, or touch it within half a cell?
  bool bodyHitsLethal(double x, double y, double yaw) const
  {
    auto * g = cm->getCostmap();
    const double c = std::cos(yaw), s = std::sin(yaw), half = g->getResolution() / 2.0;
    for (unsigned int my = 0; my < g->getSizeInCellsY(); ++my) {
      for (unsigned int mx = 0; mx < g->getSizeInCellsX(); ++mx) {
        if (g->getCost(mx, my) < nav2_costmap_2d::LETHAL_OBSTACLE) {continue;}
        double wx, wy;
        g->mapToWorld(mx, my, wx, wy);
        const double lx = c * (wx - x) + s * (wy - y);
        const double ly = -s * (wx - x) + c * (wy - y);
        if (lx >= -0.062 - half && lx <= 0.288 + half && std::abs(ly) <= 0.147 + half) {
          return true;
        }
      }
    }
    return false;
  }
  inline static std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
};

struct Sink
{
  explicit Sink(const std::string & ns)
  {
    node = std::make_shared<rclcpp::Node>("trace_sink_" + ns);
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(100)).reliable();
    cycles_sub = node->create_subscription<MppiCycle>(
      "/" + ns + "/mppi_cycles", qos, [this](MppiCycle::SharedPtr m) {cycles.push_back(*m);});
    dumps_sub = node->create_subscription<MppiRejection>(
      "/" + ns + "/mppi_rejections", qos,
      [this](MppiRejection::SharedPtr m) {dumps.push_back(*m);});
  }
  void spin(size_t want_cycles)
  {
    for (int i = 0; i < 200 && cycles.size() < want_cycles; ++i) {
      rclcpp::spin_some(node);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (int i = 0; i < 20; ++i) {  // drain anything else in flight
      rclcpp::spin_some(node);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  rclcpp::Node::SharedPtr node;
  rclcpp::Subscription<MppiCycle>::SharedPtr cycles_sub;
  rclcpp::Subscription<MppiRejection>::SharedPtr dumps_sub;
  std::vector<MppiCycle> cycles;
  std::vector<MppiRejection> dumps;
};

void waitForMatch(const rclcpp_lifecycle::LifecycleNode::SharedPtr & n, const std::string & ns)
{
  for (const auto & topic : {"/" + ns + "/mppi_cycles", "/" + ns + "/mppi_rejections"}) {
    for (int i = 0; i < 300 && n->count_subscribers(topic) == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(n->count_subscribers(topic), 1u) << topic;
  }
}

TEST_F(TraceTest, ForcedRejectionPublishesOneDumpWithTheCheckedCollisionIndex)
{
  wall(0.75, -0.5, 0.80, 0.5);  // lethal wall across the straight line ahead
  auto n = node("trace_on_reject", true, false);  // no critics: the mean stays at the warm start
  mppi::ParametersHandler params(n);
  TraceOptimizer opt;
  opt.configure(n, cm, &params, false);
  Sink sink("trace_on_reject");
  waitForMatch(n, "trace_on_reject");
  opt.stepNoise();
  opt.warm(0.3f, 0.0f);
  geometry_msgs::msg::Twist speed;
  speed.linear.x = 0.3;
  const auto path = straightPath(1.5);
  EXPECT_THROW(
    opt.evalControl(pose(0.0, 0.0, 0.0), speed, path, path.poses.back().pose, nullptr),
    nav2_core::NoValidControl);
  sink.spin(1);
  ASSERT_EQ(sink.cycles.size(), 1u);
  ASSERT_EQ(sink.dumps.size(), 1u);
  const auto & c = sink.cycles[0];
  const auto & d = sink.dumps[0];
  EXPECT_EQ(c.result, MppiCycle::RESULT_REJECTED_COLLISION);
  EXPECT_EQ(c.cycle, d.cycle);
  EXPECT_EQ(c.first_collision_index, d.first_collision_index);
  EXPECT_EQ(c.cmd_vx, 0.0);
  ASSERT_GE(d.first_collision_index, 1);
  const size_t k = static_cast<size_t>(d.first_collision_index);
  ASSERT_LT(k, d.forecast_x.size());
  // Independent geometry: the reported pose hits the wall and the pose before it does not.
  EXPECT_TRUE(bodyHitsLethal(d.forecast_x[k], d.forecast_y[k], d.forecast_yaw[k]));
  EXPECT_FALSE(bodyHitsLethal(d.forecast_x[k - 1], d.forecast_y[k - 1], d.forecast_yaw[k - 1]));
  // The reported cell is lethal and lies in the wall.
  EXPECT_GE(c.collision_cell_cost, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_GE(c.collision_cell_x, 0.74);
  EXPECT_LE(c.collision_cell_x, 0.81);
  // The dump carries the controller's grid and the batch.
  EXPECT_EQ(d.costs.size(), static_cast<size_t>(d.size_x) * d.size_y);
  EXPECT_EQ(d.costmap_hash, c.costmap_hash);
  EXPECT_EQ(d.rollout_vx.size(), static_cast<size_t>(d.batch_size) * d.time_steps);
  EXPECT_EQ(d.softmax_weights.size(), d.batch_size);
  EXPECT_EQ(d.rollout_final_check_first_collision.size(), d.batch_size);
  EXPECT_EQ(c.final_vx.size(), 41u);
  // Every rollout's recorded result is the production check on that row's controls.
  ASSERT_TRUE(opt.productionShift());
  int32_t passing = 0, clear_unshifted = 0;
  mppi::models::ControlSequence row;
  row.reset(d.time_steps);
  for (size_t i = 0; i < d.batch_size; ++i) {
    for (size_t j = 0; j < d.time_steps; ++j) {
      row.vx(j) = d.rollout_vx[i * d.time_steps + j];
      row.wz(j) = d.rollout_wz[i * d.time_steps + j];
    }
    TraceOptimizer::CollisionReport r;
    const bool ok = opt.checkWith(row, true, &r);
    EXPECT_EQ(ok, opt.production(row)) << "rollout " << i;
    EXPECT_EQ(d.rollout_final_check_first_collision[i], ok ? -2 : r.first_index) << "rollout " << i;
    passing += ok ? 1 : 0;
    clear_unshifted += opt.checkWith(row, false, nullptr) ? 1 : 0;
  }
  EXPECT_EQ(c.rollouts_pass_final_check, passing);
  EXPECT_EQ(c.rollouts_clear_unshifted, clear_unshifted);
  // Some rollouts pass and some do not, so the counts and the rank are informative.
  EXPECT_GT(passing, 0);
  EXPECT_LT(passing, static_cast<int32_t>(d.batch_size));
  // Rank = position, by total cost (stable), of the first passing rollout.
  std::vector<size_t> order(d.batch_size);
  for (size_t i = 0; i < order.size(); ++i) {order[i] = i;}
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return d.total_costs[a] < d.total_costs[b];
    });
  int32_t rank = -1;
  for (size_t r = 0; r < order.size(); ++r) {
    if (d.rollout_final_check_first_collision[order[r]] == -2) {
      rank = static_cast<int32_t>(r);
      break;
    }
  }
  EXPECT_EQ(c.best_passing_rollout_rank, rank);
  EXPECT_EQ(c.cheapest_rollout_final_collision, d.rollout_final_check_first_collision[order[0]]);
  // The production check on the recorded final sequence fails at the recorded index.
  mppi::models::ControlSequence fin;
  fin.reset(41);
  for (size_t j = 0; j < 41; ++j) {
    fin.vx(j) = c.final_vx[j];
    fin.wz(j) = c.final_wz[j];
  }
  TraceOptimizer::CollisionReport rf;
  EXPECT_FALSE(opt.checkWith(fin, true, &rf));
  EXPECT_EQ(rf.first_index, c.first_collision_index);
}

TEST_F(TraceTest, AcceptedCyclePublishesARecordAndNoDump)
{
  auto n = node("trace_on_accept", true);
  mppi::ParametersHandler params(n);
  TraceOptimizer opt;
  opt.configure(n, cm, &params, true);
  Sink sink("trace_on_accept");
  waitForMatch(n, "trace_on_accept");
  geometry_msgs::msg::Twist speed;
  speed.linear.x = 0.2;
  const auto path = straightPath(1.5);
  const auto cmd = opt.evalControl(pose(0.0, 0.0, 0.0), speed, path, path.poses.back().pose,
      nullptr);
  sink.spin(1);
  ASSERT_EQ(sink.cycles.size(), 1u);
  EXPECT_EQ(sink.dumps.size(), 0u);
  EXPECT_EQ(sink.cycles[0].result, MppiCycle::RESULT_ACCEPTED);
  EXPECT_EQ(sink.cycles[0].first_collision_index, -2);
  EXPECT_DOUBLE_EQ(sink.cycles[0].cmd_vx, cmd.twist.linear.x);
  EXPECT_DOUBLE_EQ(sink.cycles[0].cmd_wz, cmd.twist.angular.z);
  EXPECT_EQ(sink.cycles[0].rollouts_pass_final_check, -1);
}

TEST_F(TraceTest, OffPublishesNothing)
{
  wall(0.75, -0.5, 0.80, 0.5);
  auto n = node("trace_off", false, false);
  mppi::ParametersHandler params(n);
  TraceOptimizer opt;
  opt.configure(n, cm, &params, false);
  Sink sink("trace_off");
  opt.warm(0.3f, 0.0f);
  geometry_msgs::msg::Twist speed;
  speed.linear.x = 0.3;
  const auto path = straightPath(1.5);
  EXPECT_THROW(
    opt.evalControl(pose(0.0, 0.0, 0.0), speed, path, path.poses.back().pose, nullptr),
    nav2_core::NoValidControl);
  sink.spin(0);
  EXPECT_EQ(n->count_publishers("/trace_off/mppi_cycles"), 0u);
  EXPECT_EQ(n->count_publishers("/trace_off/mppi_rejections"), 0u);
  EXPECT_TRUE(sink.cycles.empty());
  EXPECT_TRUE(sink.dumps.empty());
}

struct IdentityCounts
{
  int rejected{0}, final_rejected{0}, accepted{0};
};

// Run the same ticks on a tracing-off and a tracing-on optimizer; every command, exception
// and memory word must match bit for bit.
IdentityCounts runIdentity(
  TraceTest & t, const std::string & tag, bool critics, bool stepped,
  rclcpp_lifecycle::LifecycleNode::SharedPtr n_off, rclcpp_lifecycle::LifecycleNode::SharedPtr n_on,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm, const nav_msgs::msg::Path & path,
  const std::function<geometry_msgs::msg::PoseStamped(double, double, double)> & pose)
{
  (void)t;
  mppi::ParametersHandler p_off(n_off), p_on(n_on);
  TraceOptimizer off, on;
  off.configure(n_off, cm, &p_off, critics);
  on.configure(n_on, cm, &p_on, critics);
  IdentityCounts k;
  for (int tick = 0; tick < 40; ++tick) {
    const double x = 0.015 * tick, y = 0.02 * std::sin(0.3 * tick);
    const double yaw = 0.15 * std::sin(0.2 * tick);
    geometry_msgs::msg::Twist speed;
    speed.linear.x = 0.25;
    speed.angular.z = 0.1 * std::cos(0.2 * tick);
    for (auto * o : {&off, &on}) {
      if (stepped) {o->stepNoise();} else {o->freezeNoise();}
      // Same memory every fifth tick so the forced rejections recur.
      if (tick % 5 == 0) {o->warm(0.3f, stepped ? 0.0f : 0.25f);}
    }
    geometry_msgs::msg::TwistStamped c_off, c_on;
    bool t_off = false, t_on = false;
    std::string w_off, w_on;
    try {
      c_off = off.evalControl(pose(x, y, yaw), speed, path, path.poses.back().pose, nullptr);
    } catch (const nav2_core::NoValidControl & e) {t_off = true; w_off = e.what();}
    try {
      c_on = on.evalControl(pose(x, y, yaw), speed, path, path.poses.back().pose, nullptr);
    } catch (const nav2_core::NoValidControl & e) {t_on = true; w_on = e.what();}
    EXPECT_EQ(t_off, t_on) << tag << " tick " << tick;
    EXPECT_EQ(w_off, w_on) << tag << " tick " << tick;
    (t_on ? k.rejected : k.accepted)++;
    k.final_rejected += w_on.find("not collision-free") != std::string::npos ? 1 : 0;
    EXPECT_EQ(c_off.twist.linear.x, c_on.twist.linear.x) << tag << " tick " << tick;
    EXPECT_EQ(c_off.twist.angular.z, c_on.twist.angular.z) << tag << " tick " << tick;
    for (size_t j = 0; j < off.tape().vx.size(); ++j) {
      EXPECT_EQ(off.tape().vx(j), on.tape().vx(j)) << tag << " tick " << tick << " step " << j;
      EXPECT_EQ(off.tape().wz(j), on.tape().wz(j)) << tag << " tick " << tick << " step " << j;
    }
    for (size_t j = 0; j < off.history().size(); ++j) {
      EXPECT_EQ(off.history()[j].vx, on.history()[j].vx) << tag << " tick " << tick;
      EXPECT_EQ(off.history()[j].wz, on.history()[j].wz) << tag << " tick " << tick;
    }
  }
  return k;
}

TEST_F(TraceTest, CommandsAndMemoryIdenticalWithTracingOnAndOff)
{
  const auto path = straightPath(2.0);
  const auto mk = [](double x, double y, double yaw) {return pose(x, y, yaw);};
  // Critics loaded (capture path, fallback path): a wall beside the route.
  wall(1.10, 0.10, 1.15, 0.60);
  const auto a = runIdentity(*this, "critics", true, false, node("ident_off", false),
      node("ident_on", true), cm, path, mk);
  // No critics, stepped bank, a wall across the route: the final check rejects, which is
  // the path that runs the per-rollout diagnostics and builds the dump.
  SetUp();
  wall(0.95, -0.5, 1.00, 0.5);
  const auto b = runIdentity(*this, "final", false, true, node("ident_off_f", false, false),
      node("ident_on_f", true, false), cm, path, mk);
  EXPECT_GT(a.accepted + b.accepted, 0);
  EXPECT_GT(b.final_rejected, 0);
}

TEST_F(TraceTest, ProductionCheckIsTheReportingCheck)
{
  wall(0.75, -0.5, 0.80, 0.5);
  auto n = node("trace_check", false, false);
  mppi::ParametersHandler params(n);
  TraceOptimizer opt;
  opt.configure(n, cm, &params, false);
  opt.setState(pose(0.0, 0.0, 0.0), 0.0);
  ASSERT_TRUE(opt.productionShift());
  mppi::models::ControlSequence s;
  s.reset(41);
  // From rest, a sequence that stalls at index 1 (index 0 is replaced by the measured
  // speed): the unshifted forecast spends a step on the stall, the shifted one does not,
  // so near the wall they disagree for some speed.
  int disagreements = 0, checked = 0;
  for (float v = 0.17f; v <= 0.30f; v += 0.0005f) {
    s.vx.fill(v);
    s.vx(1) = 0.0f;
    const bool shifted = opt.checkWith(s, true, nullptr);
    const bool unshifted = opt.checkWith(s, false, nullptr);
    disagreements += shifted != unshifted ? 1 : 0;
    EXPECT_EQ(opt.production(s), shifted) << v;
    for (bool shift : {false, true}) {
      TraceOptimizer::CollisionReport r;
      const bool with = opt.checkWith(s, shift, &r);
      EXPECT_EQ(with, opt.checkWith(s, shift, nullptr));
      EXPECT_EQ(with, r.first_index == -2);
      if (!with) {EXPECT_GE(r.first_index, -1);}
    }
    ++checked;
  }
  EXPECT_GT(disagreements, 0) << "fixture never separates the shift; " << checked << " speeds";
  // A failure before any pose is tested is never reported as passing.
  s.vx.fill(0.2f);
  s.vx(3) = std::numeric_limits<float>::quiet_NaN();
  TraceOptimizer::CollisionReport r;
  EXPECT_FALSE(opt.checkWith(s, true, &r));
  EXPECT_FALSE(opt.production(s));
  EXPECT_EQ(r.first_index, -4);
  s.vx(3) = 0.2f;
  auto bad = pose(0.0, 0.0, 0.0);
  bad.pose.orientation = geometry_msgs::msg::Quaternion{};
  bad.pose.orientation.w = 0.0;
  opt.setState(bad, 0.0);
  EXPECT_FALSE(opt.checkWith(s, true, &r));
  EXPECT_EQ(r.first_index, -4);
}

}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(0, nullptr);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
