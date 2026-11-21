// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include "nav2_mppi_controller/optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <cstring>
#include <type_traits>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_util/line_iterator.hpp"

namespace mppi
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

namespace
{
// std::isfinite may be optimized away by this package's -ffast-math policy.
// Inspect IEEE exponent bits so invalid inputs are still rejected in Release.
template<typename T>
bool finiteValue(T value)
{
  static_assert(std::numeric_limits<T>::is_iec559);
  using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  Bits bits;
  std::memcpy(&bits, &value, sizeof(value));
  constexpr Bits exponent = sizeof(T) == 4 ?
    static_cast<Bits>(0x7f800000U) : static_cast<Bits>(0x7ff0000000000000ULL);
  return (bits & exponent) != exponent;
}
}  // namespace


void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);
  critic_manager_.setCaptureCriticCosts(trace_rejections_);
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

  if (trace_cycles_) {
    trace_cycle_pub_ = node->create_publisher<nav2_critics_msgs::msg::MppiCycle>(
      "~/mppi_cycles", rclcpp::QoS(rclcpp::KeepLast(100)).reliable());
    RCLCPP_INFO(logger_, "Publishing per-cycle MPPI records to ~/mppi_cycles");
  }
  if (trace_rejections_) {
    trace_rejection_pub_ = node->create_publisher<nav2_critics_msgs::msg::MppiRejection>(
      "~/mppi_rejections", rclcpp::QoS(rclcpp::KeepLast(20)).reliable());
    RCLCPP_INFO(logger_, "Publishing MPPI rejection dumps to ~/mppi_rejections");
  }

  reset();
}

void Optimizer::shutdown()
{
  noise_generator_.shutdown();
  critic_manager_.on_cleanup();
}

void Optimizer::on_activate()
{
  critic_manager_.on_activate();
  if (trace_cycle_pub_) {trace_cycle_pub_->on_activate();}
  if (trace_rejection_pub_) {trace_rejection_pub_->on_activate();}
}

void Optimizer::on_deactivate()
{
  critic_manager_.on_deactivate();
  if (trace_cycle_pub_) {trace_cycle_pub_->on_deactivate();}
  if (trace_rejection_pub_) {trace_rejection_pub_->on_deactivate();}
}

void Optimizer::getParams()
{
  std::string motion_model_name;

  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);
  getParam(s.batch_size, "batch_size", 1000);
  getParam(s.iteration_count, "iteration_count", 1);
  getParam(s.temperature, "temperature", 0.3f);
  getParam(s.gamma, "gamma", 0.015f);
  getParam(s.base_constraints.vx_max, "vx_max", 0.5f);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35f);
  getParam(s.base_constraints.vy, "vy_max", 0.5f);
  getParam(s.base_constraints.wz, "wz_max", 1.9f);
  getParam(s.base_constraints.ax_max, "ax_max", 3.0f);
  getParam(s.base_constraints.ax_min, "ax_min", -3.0f);
  getParam(s.base_constraints.ay_max, "ay_max", 3.0f);
  getParam(s.base_constraints.ay_min, "ay_min", -3.0f);
  getParam(s.base_constraints.az_max, "az_max", 3.5f);
  getParam(s.sampling_std.vx, "vx_std", 0.2f);
  getParam(s.sampling_std.vy, "vy_std", 0.2f);
  getParam(s.sampling_std.wz, "wz_std", 0.4f);
  getParam(s.retry_attempt_limit, "retry_attempt_limit", 1);
  // GENERIC diagnostics, default off: per-cycle record and rejection dump. Read-only.
  getParam(trace_cycles_, "trace_cycles", false, ParameterType::Static);
  getParam(trace_rejections_, "trace_rejections", false, ParameterType::Static);

  s.base_constraints.ax_max = std::abs(s.base_constraints.ax_max);
  if (s.base_constraints.ax_min > 0.0) {
    s.base_constraints.ax_min = -1.0 * s.base_constraints.ax_min;
    RCLCPP_WARN(
      logger_,
      "Sign of the parameter ax_min is incorrect, consider setting it negative.");
  }

  if (s.base_constraints.ay_min > 0.0) {
    s.base_constraints.ay_min = -1.0 * s.base_constraints.ay_min;
    RCLCPP_WARN(
      logger_,
      "Sign of the parameter ay_min is incorrect, consider setting it negative.");
  }

  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  parameters_handler_->addPostCallback([this]() {reset();});

  double controller_frequency;
  getParentParam(controller_frequency, "controller_frequency", 0.0, ParameterType::Static);
  setOffset(controller_frequency);
}

void Optimizer::setOffset(double controller_frequency)
{
  const double controller_period = 1.0 / controller_frequency;
  constexpr double eps = 1e-6;

  if ((controller_period + eps) < settings_.model_dt) {
    RCLCPP_WARN(
      logger_,
      "Controller period is less then model dt, consider setting it equal");
  } else if (abs(controller_period - settings_.model_dt) < eps) {
    RCLCPP_INFO(
      logger_,
      "Controller period is equal to model dt. Control sequence "
      "shifting is ON");
    settings_.shift_control_sequence = true;
  } else {
    throw nav2_core::ControllerException(
            "Controller period more then model dt, set it equal to model dt");
  }
}

void Optimizer::reset(bool reset_dynamic_speed_limits)
{
  state_.reset(settings_.batch_size, settings_.time_steps);
  control_sequence_.reset(settings_.time_steps);
  control_history_[0] = {0.0f, 0.0f, 0.0f};
  control_history_[1] = {0.0f, 0.0f, 0.0f};
  control_history_[2] = {0.0f, 0.0f, 0.0f};
  control_history_[3] = {0.0f, 0.0f, 0.0f};

  if (reset_dynamic_speed_limits) {
    settings_.constraints = settings_.base_constraints;
  }

  costs_ = xt::zeros<float>({settings_.batch_size});
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());
  motion_model_->initialize(settings_.constraints, settings_.model_dt);

  RCLCPP_INFO(logger_, "Optimizer reset");
}

void Optimizer::resetCritics()
{
  critic_manager_.reset();
}

void Optimizer::onNewPlan(const nav_msgs::msg::Path & plan)
{
  critic_manager_.onNewPlan(plan);
}

bool Optimizer::isHolonomic() const
{
  return motion_model_->isHolonomic();
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & goal,
  nav2_core::GoalChecker * goal_checker)
{
  prepare(robot_pose, robot_speed, plan, goal, goal_checker);

  const bool tracing = trace_cycles_ || trace_rejections_;
  if (tracing) {
    ++trace_cycle_;
    trace_plan_stamp_ = plan.header.stamp;
    trace_plan_size_ = plan.poses.size();
    if (trace_rejections_) {trace_plan_ = plan;}
    trace_warm_ = control_sequence_;
    trace_opt_ = models::ControlSequence{};
    trace_smooth_ = models::ControlSequence{};
    trace_final_ = models::ControlSequence{};
    trace_report_ = CollisionReport{};
    trace_final_nonfinite_ = false;
  }

  bool diag = (diag_tick_++ % 4 == 0);
  const auto ts = settings_.time_steps;

  // [A] INPUT: robot feedback velocity + control_sequence_ carried from last tick
  if (diag && ts >= 3) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] A.input  robot_vx=%.4f robot_wz=%.3f  ctrl[0..2]=(%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f)",
      robot_speed.linear.x, robot_speed.angular.z,
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2));
  }

  try {
    do {
      optimize(diag);
    } while (fallback(critics_data_.fail_flag));
  } catch (const nav2_core::NoValidControl &) {
    if (tracing) {
      publishTrace(nav2_critics_msgs::msg::MppiCycle::RESULT_ALL_ROLLOUTS_COLLIDE, nullptr);
    }
    throw;
  }
  if (tracing) {trace_opt_ = control_sequence_;}

  // [E] POST-OPTIMIZE (after softmax + accel constraints)
  if (diag && ts >= 5) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] E.post-opt  ctrl[0..4]=(%.4f,%.4f,%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f)",
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.vx(3), control_sequence_.vx(4),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2));
  }

  utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);
  if (tracing) {trace_smooth_ = control_sequence_;}

  // [F] POST-SG FILTER
  if (diag && ts >= 5) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] F.post-SG   ctrl[0..4]=(%.4f,%.4f,%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f)",
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.vx(3), control_sequence_.vx(4),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2));
  }

  geometry_msgs::msg::TwistStamped control;
  try {
    control = getValidatedControlFromSequenceAsTwist(plan.header.stamp);
  } catch (const nav2_core::NoValidControl &) {
    if (tracing) {
      publishTrace(
        trace_final_nonfinite_ ?
        nav2_critics_msgs::msg::MppiCycle::RESULT_REJECTED_NONFINITE :
        nav2_critics_msgs::msg::MppiCycle::RESULT_REJECTED_COLLISION, nullptr);
    }
    throw;
  }
  if (tracing) {
    publishTrace(nav2_critics_msgs::msg::MppiCycle::RESULT_ACCEPTED, &control);
  }

  // [G] FINAL CMD_VEL
  if (diag) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] G.cmd_vel   vx=%.4f wz=%.3f",
      control.twist.linear.x, control.twist.angular.z);
  }

  if (settings_.shift_control_sequence) {
    shiftControlSequence();
  }

  return control;
}

/**
 * @brief Run MPPI optimization iterations to score trajectories and update controls
 * @param diag Whether diagnostic logging is enabled for this control tick
 */
void Optimizer::optimize(bool diag)
{
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    // Recentring changes each row's controls and rollout. Costs from an earlier
    // bank (including collision penalties) do not score this new trajectory.
    costs_.fill(0.0f);
    generateNoisedTrajectories();

    // [B] BATCH STATS after noise+predict
    if (diag) {
      auto cvx0 = xt::view(state_.cvx, xt::all(), 0);
      auto cvx1 = xt::view(state_.cvx, xt::all(), 1);
      auto cwz0 = xt::view(state_.cwz, xt::all(), 0);
      auto cwz1 = xt::view(state_.cwz, xt::all(), 1);
      RCLCPP_DEBUG(logger_,
        "[MPPI] B.batch  cvx0: mean=%.3f [%.3f,%.3f] neg=%d/%d"
        "  cvx1: mean=%.3f [%.3f,%.3f]  cwz_mean=(%.3f,%.3f)",
        static_cast<float>(xt::mean(cvx0)()),
        static_cast<float>(xt::amin(cvx0)()), static_cast<float>(xt::amax(cvx0)()),
        static_cast<int>(xt::sum(xt::cast<int>(cvx0 < 0.0f))()), settings_.batch_size,
        static_cast<float>(xt::mean(cvx1)()),
        static_cast<float>(xt::amin(cvx1)()), static_cast<float>(xt::amax(cvx1)()),
        static_cast<float>(xt::mean(cwz0)()), static_cast<float>(xt::mean(cwz1)()));
    }

    critics_data_.reuse_obstacles_cache = (i > 0u) || (retry_attempt_counter_ > 0u);
    critic_manager_.evalTrajectoriesScores(critics_data_);
    critics_data_.reuse_obstacles_cache = false;

    // [C] COSTS after critics (before control regularization + softmax)
    if (diag) {
      float c_min = static_cast<float>(xt::amin(costs_)());
      float c_max = static_cast<float>(xt::amax(costs_)());
      float c_mean = static_cast<float>(xt::mean(costs_)());
      float c_std = static_cast<float>(xt::stddev(costs_)());
      RCLCPP_DEBUG(logger_,
        "[MPPI] C.costs  min=%.1f max=%.1f mean=%.1f spread=%.1f std=%.2f",
        c_min, c_max, c_mean, c_max - c_min, c_std);
    }

    updateControlSequence(diag);
  }
}

bool Optimizer::fallback(bool fail)
{
  if (!fail) {
    retry_attempt_counter_ = 0;
    return false;
  }

  reset(false /*Don't reset zone-based speed limits after fallback*/);

  if (++retry_attempt_counter_ > settings_.retry_attempt_limit) {
    retry_attempt_counter_ = 0;
    throw nav2_core::NoValidControl("Optimizer fail to compute path");
  }

  // A permitted retry must score its fresh batch, not retain the failed batch flag.
  critics_data_.fail_flag = false;
  return true;
}

void Optimizer::prepare(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::Pose & goal,
  nav2_core::GoalChecker * goal_checker)
{
  state_.pose = robot_pose;
  state_.speed = robot_speed;
  path_ = utils::toTensor(plan);
  costs_.fill(0.0f);
  goal_ = goal;

  retry_attempt_counter_ = 0;
  critics_data_.fail_flag = false;
  critics_data_.goal_checker = goal_checker;
  critics_data_.motion_model = motion_model_;
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
  critic_manager_.onTickStart();
}

void Optimizer::shiftControlSequence()
{
  using namespace xt::placeholders;  // NOLINT
  control_sequence_.vx = xt::roll(control_sequence_.vx, -1);
  control_sequence_.wz = xt::roll(control_sequence_.wz, -1);


  xt::view(control_sequence_.vx, -1) =
    xt::view(control_sequence_.vx, -2);

  xt::view(control_sequence_.wz, -1) =
    xt::view(control_sequence_.wz, -2);


  if (isHolonomic()) {
    control_sequence_.vy = xt::roll(control_sequence_.vy, -1);
    xt::view(control_sequence_.vy, -1) =
      xt::view(control_sequence_.vy, -2);
  }
}

void Optimizer::generateNoisedTrajectories()
{
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
}

void Optimizer::applyControlSequenceConstraints()
{
  auto & s = settings_;

  if (isHolonomic()) {
    control_sequence_.vy = xt::clip(control_sequence_.vy, -s.constraints.vy, s.constraints.vy);
  }

  control_sequence_.vx = xt::clip(control_sequence_.vx, s.constraints.vx_min, s.constraints.vx_max);
  control_sequence_.wz = xt::clip(control_sequence_.wz, -s.constraints.wz, s.constraints.wz);

  float max_delta_vx = s.model_dt * s.constraints.ax_max;
  float min_delta_vx = s.model_dt * s.constraints.ax_min;
  float max_delta_vy = s.model_dt * s.constraints.ay_max;
  float min_delta_vy = s.model_dt * s.constraints.ay_min;
  float max_delta_wz = s.model_dt * s.constraints.az_max;
  float vx_last = control_sequence_.vx(0);
  float vy_last = control_sequence_.vy(0);
  float wz_last = control_sequence_.wz(0);
  for (unsigned int i = 1; i != control_sequence_.vx.shape(0); i++) {
    float & vx_curr = control_sequence_.vx(i);
    if (vx_last > 0) {
      vx_curr = std::clamp(vx_curr, vx_last + min_delta_vx, vx_last + max_delta_vx);
    } else {
      vx_curr = std::clamp(vx_curr, vx_last - max_delta_vx, vx_last - min_delta_vx);
    }
    vx_last = vx_curr;

    float & wz_curr = control_sequence_.wz(i);
    wz_curr = std::clamp(wz_curr, wz_last - max_delta_wz, wz_last + max_delta_wz);
    wz_last = wz_curr;

    if (isHolonomic()) {
      float & vy_curr = control_sequence_.vy(i);
      if (vy_last > 0) {
        vy_curr = std::clamp(vy_curr, vy_last + min_delta_vy, vy_last + max_delta_vy);
      } else {
        vy_curr = std::clamp(vy_curr, vy_last - max_delta_vy, vy_last - min_delta_vy);
      }
      vy_last = vy_curr;
    }
  }

  motion_model_->applyConstraints(control_sequence_);
}

void Optimizer::updateStateVelocities(
  models::State & state) const
{
  updateInitialStateVelocities(state);
  propagateStateVelocitiesFromInitials(state);
}

void Optimizer::updateInitialStateVelocities(
  models::State & state) const
{
  xt::noalias(xt::view(state.vx, xt::all(), 0)) = static_cast<float>(state.speed.linear.x);
  xt::noalias(xt::view(state.wz, xt::all(), 0)) = static_cast<float>(state.speed.angular.z);

  if (isHolonomic()) {
    xt::noalias(xt::view(state.vy, xt::all(), 0)) = static_cast<float>(state.speed.linear.y);
  }
}

void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & state) const
{
  motion_model_->predict(state);
}

void Optimizer::integrateStateVelocities(
  xt::xtensor<float, 2> & trajectory,
  const xt::xtensor<float, 2> & sequence) const
{
  float initial_yaw = static_cast<float>(tf2::getYaw(state_.pose.pose.orientation));

  const auto vx = xt::view(sequence, xt::all(), 0);
  const auto wz = xt::view(sequence, xt::all(), 1);

  auto traj_x = xt::view(trajectory, xt::all(), 0);
  auto traj_y = xt::view(trajectory, xt::all(), 1);
  auto traj_yaws = xt::view(trajectory, xt::all(), 2);

  xt::noalias(traj_yaws) = xt::cumsum(wz * settings_.model_dt, 0) + initial_yaw;

  auto yaw_cos = xt::roll(xt::eval(xt::cos(traj_yaws)), 1);
  auto yaw_sin = xt::roll(xt::eval(xt::sin(traj_yaws)), 1);
  xt::view(yaw_cos, 0) = cosf(initial_yaw);
  xt::view(yaw_sin, 0) = sinf(initial_yaw);

  auto && dx = xt::eval(vx * yaw_cos);
  auto && dy = xt::eval(vx * yaw_sin);

  if (isHolonomic()) {
    const auto vy = xt::view(sequence, xt::all(), 2);
    dx = dx - vy * yaw_sin;
    dy = dy + vy * yaw_cos;
  }

  xt::noalias(traj_x) = state_.pose.pose.position.x + xt::cumsum(dx * settings_.model_dt, 0);
  xt::noalias(traj_y) = state_.pose.pose.position.y + xt::cumsum(dy * settings_.model_dt, 0);
}

void Optimizer::integrateStateVelocities(
  models::Trajectories & trajectories,
  const models::State & state) const
{
  const float initial_yaw = static_cast<float>(tf2::getYaw(state.pose.pose.orientation));

  xt::noalias(trajectories.yaws) =
    xt::cumsum(state.wz * settings_.model_dt, {1}) + initial_yaw;

  auto yaw_cos = xt::roll(xt::eval(xt::cos(trajectories.yaws)), 1, 1);
  auto yaw_sin = xt::roll(xt::eval(xt::sin(trajectories.yaws)), 1, 1);
  xt::view(yaw_cos, xt::all(), 0) = cosf(initial_yaw);
  xt::view(yaw_sin, xt::all(), 0) = sinf(initial_yaw);

  auto && dx = xt::eval(state.vx * yaw_cos);
  auto && dy = xt::eval(state.vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - state.vy * yaw_sin;
    dy = dy + state.vy * yaw_cos;
  }

  xt::noalias(trajectories.x) = state.pose.pose.position.x +
    xt::cumsum(dx * settings_.model_dt, {1});
  xt::noalias(trajectories.y) = state.pose.pose.position.y +
    xt::cumsum(dy * settings_.model_dt, {1});
}

xt::xtensor<float, 2> Optimizer::getOptimizedTrajectory()
{
  const bool is_holo = isHolonomic();
  auto && sequence =
    xt::xtensor<float, 2>::from_shape({settings_.time_steps, is_holo ? 3u : 2u});
  auto && trajectories = xt::xtensor<float, 2>::from_shape({settings_.time_steps, 3});

  xt::noalias(xt::view(sequence, xt::all(), 0)) = control_sequence_.vx;
  xt::noalias(xt::view(sequence, xt::all(), 1)) = control_sequence_.wz;

  if (is_holo) {
    xt::noalias(xt::view(sequence, xt::all(), 2)) = control_sequence_.vy;
  }

  integrateStateVelocities(trajectories, sequence);
  return std::move(trajectories);
}

void Optimizer::updateControlSequence(bool diag)
{
  const bool is_holo = isHolonomic();
  auto & s = settings_;
  auto bounded_noises_vx = state_.cvx - control_sequence_.vx;
  auto bounded_noises_wz = state_.cwz - control_sequence_.wz;
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.vx, 2) * xt::sum(
    xt::view(control_sequence_.vx, xt::newaxis(), xt::all()) * bounded_noises_vx, 1, immediate);
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.wz, 2) * xt::sum(
    xt::view(control_sequence_.wz, xt::newaxis(), xt::all()) * bounded_noises_wz, 1, immediate);

  if (is_holo) {
    auto bounded_noises_vy = state_.cvy - control_sequence_.vy;
    xt::noalias(costs_) +=
      s.gamma / powf(s.sampling_std.vy, 2) * xt::sum(
      xt::view(control_sequence_.vy, xt::newaxis(), xt::all()) * bounded_noises_vy,
      1, immediate);
  }

  auto && costs_normalized = costs_ - xt::amin(costs_, immediate);
  auto && exponents = xt::eval(xt::exp(-1 / settings_.temperature * costs_normalized));
  auto && softmaxes = xt::eval(exponents / xt::sum(exponents, immediate));
  auto && softmaxes_extened = xt::eval(xt::view(softmaxes, xt::all(), xt::newaxis()));
  if (trace_rejections_) {
    trace_total_costs_ = costs_;
    trace_softmax_ = softmaxes;
  }

  // [D] SOFTMAX: how concentrated are the weights?
  if (diag) {
    float w_max = static_cast<float>(xt::amax(softmaxes)());
    size_t w_max_idx = 0;
    for (size_t k = 1; k < softmaxes.size(); ++k) {
      if (softmaxes(k) > softmaxes(w_max_idx)) {w_max_idx = k;}
    }
    float ess = 1.0f / static_cast<float>(xt::sum(xt::square(softmaxes))());
    RCLCPP_DEBUG(logger_,
      "[MPPI] D.softmax  w_max=%.4f ess=%.0f/%d  best[%zu]: cvx(%.3f,%.3f,%.3f) cwz(%.3f,%.3f)",
      w_max, ess, settings_.batch_size, w_max_idx,
      state_.cvx(w_max_idx, 0), state_.cvx(w_max_idx, 1), state_.cvx(w_max_idx, 2),
      state_.cwz(w_max_idx, 0), state_.cwz(w_max_idx, 1));
  }

  xt::noalias(control_sequence_.vx) = xt::sum(state_.cvx * softmaxes_extened, 0, immediate);
  xt::noalias(control_sequence_.wz) = xt::sum(state_.cwz * softmaxes_extened, 0, immediate);
  if (is_holo) {
    xt::noalias(control_sequence_.vy) = xt::sum(state_.cvy * softmaxes_extened, 0, immediate);
  }

  // [D2] POST-SOFTMAX (before accel constraints)
  if (diag && settings_.time_steps >= 5) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] D2.w-avg  ctrl[0..4]=(%.4f,%.4f,%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f)",
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.vx(3), control_sequence_.vx(4),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2));
  }

  applyControlSequenceConstraints();
}

bool Optimizer::controlSequenceFinite() const
{
  return sequenceFinite(control_sequence_);
}

bool Optimizer::sequenceFinite(const models::ControlSequence & sequence) const
{
  const size_t n = settings_.time_steps;
  if (!motion_model_ || n < 2 || !finiteValue(settings_.model_dt) || settings_.model_dt <= 0 ||
    sequence.vx.size() != n || sequence.wz.size() != n ||
    sequence.vy.size() != n)
  {
    return false;
  }
  for (size_t j = 0; j < n; ++j) {
    if (!finiteValue(sequence.vx(j)) || !finiteValue(sequence.wz(j)) ||
      (isHolonomic() && !finiteValue(sequence.vy(j))))
    {
      return false;
    }
  }
  return true;
}

bool Optimizer::controlSequenceCollisionFree() const
{
  return sequenceCollisionFree(control_sequence_, settings_.shift_control_sequence, nullptr);
}

bool Optimizer::sequenceCollisionFree(
  const models::ControlSequence & sequence, bool shift, CollisionReport * report) const
{
  // Adapted from PR #14's final filtered-sequence guard. It is unconditional
  // here: scoring sampled rollouts is not proof that the emitted mean is clear.
  if (report) {*report = CollisionReport{};}  // first_index -4: rejected before any pose
  if (!costmap_ || !costmap_ros_ || !sequenceFinite(sequence)) {return false;}
  const auto & p = state_.pose.pose.position;
  const auto & q = state_.pose.pose.orientation;
  if (!finiteValue(p.x) || !finiteValue(p.y) || !finiteValue(q.x) || !finiteValue(q.y) ||
    !finiteValue(q.z) || !finiteValue(q.w) ||
    q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 1e-12 ||
    !finiteValue(state_.speed.linear.x) || !finiteValue(state_.speed.angular.z) ||
    (isHolonomic() && !finiteValue(state_.speed.linear.y)))
  {
    return false;
  }

  const size_t n = settings_.time_steps;
  const size_t offset = shift ? 1 : 0;
  models::State predicted;
  // The first state column is measured velocity. Include one additional column
  // so every command in the emitted horizon contributes to the forecast.
  predicted.reset(1, n + 1);
  predicted.pose = state_.pose;
  predicted.speed = state_.speed;
  for (size_t j = 0; j <= n; ++j) {
    const size_t k = std::min(j + offset, n - 1);
    predicted.cvx(0, j) = sequence.vx(k);
    predicted.cwz(0, j) = sequence.wz(k);
    if (isHolonomic()) {predicted.cvy(0, j) = sequence.vy(k);}
  }
  updateStateVelocities(predicted);
  models::Trajectories trajectory;
  trajectory.reset(1, n + 1);
  integrateStateVelocities(trajectory, predicted);

  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
  const auto footprint = costmap_ros_->getRobotFootprint();
  const bool circle = costmap_ros_->getUseRadius();
  if (footprint.size() < 3) {return false;}
  const double radius = circle ?
    costmap_ros_->getLayeredCostmap()->getCircumscribedRadius() : 0.0;
  if (circle && (!finiteValue(radius) || radius <= 0.0)) {return false;}
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto in_bounds = [&](double x, double y) {
      return finiteValue(x) && finiteValue(y) &&
             x >= costmap_->getOriginX() && y >= costmap_->getOriginY() &&
             x < costmap_->getOriginX() + costmap_->getSizeInCellsX() * costmap_->getResolution() &&
             y < costmap_->getOriginY() + costmap_->getSizeInCellsY() * costmap_->getResolution();
    };
  const auto clear_pose = [&](double x, double y, double yaw) {
      if (!in_bounds(x, y) || !finiteValue(yaw)) {return false;}
      unsigned int mx, my;
      if (!costmap_->worldToMap(x, y, mx, my)) {return false;}
      const auto center_cost = costmap_->getCost(mx, my);
      if (center_cost >= (circle ? nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE :
        nav2_costmap_2d::LETHAL_OBSTACLE))
      {
        return false;
      }
      if (circle) {
        // Inflation is optional and may be smaller than the body. Check every
        // cell intersecting the padded circular body, including its interior.
        if (!in_bounds(x - radius, y - radius) || !in_bounds(x + radius, y + radius)) {
          return false;
        }
        unsigned int x0, y0, x1, y1;
        if (!costmap_->worldToMap(x - radius, y - radius, x0, y0) ||
          !costmap_->worldToMap(x + radius, y + radius, x1, y1))
        {
          return false;
        }
        const double resolution = costmap_->getResolution();
        for (unsigned int cy = y0; cy <= y1; ++cy) {
          const double bottom = costmap_->getOriginY() + cy * resolution;
          const double dy = std::max({bottom - y, 0.0, y - bottom - resolution});
          for (unsigned int cx = x0; cx <= x1; ++cx) {
            const double left = costmap_->getOriginX() + cx * resolution;
            const double dx = std::max({left - x, 0.0, x - left - resolution});
            if (dx * dx + dy * dy <= radius * radius &&
              costmap_->getCost(cx, cy) >= nav2_costmap_2d::LETHAL_OBSTACLE)
            {
              return false;
            }
          }
        }
        return true;
      }
      const double c = std::cos(yaw), sn = std::sin(yaw);
      for (const auto & vertex : footprint) {
        if (!in_bounds(x + c * vertex.x - sn * vertex.y, y + sn * vertex.x + c * vertex.y)) {
          return false;
        }
      }
      const double cost = checker.footprintCostAtPose(x, y, yaw, footprint);
      return cost >= 0 && cost < nav2_costmap_2d::LETHAL_OBSTACLE;
    };
  // Diagnostics only: the first lethal, unknown or off-map cell under a failing pose,
  // found the way the check rasterises the body (centre cell, then perimeter edges).
  const auto locate = [&](double x, double y, double yaw) {
      const auto off_map = [&](double wx, double wy) {
          report->cell_x = wx;
          report->cell_y = wy;
          report->cell_cost = nav2_costmap_2d::NO_INFORMATION;
        };
      const auto cell = [&](unsigned int mx, unsigned int my) {
          costmap_->mapToWorld(mx, my, report->cell_x, report->cell_y);
          report->cell_cost = costmap_->getCost(mx, my);
        };
      unsigned int mx, my;
      if (!in_bounds(x, y) || !finiteValue(yaw) || !costmap_->worldToMap(x, y, mx, my)) {
        off_map(x, y);
        return;
      }
      const auto center_cost = costmap_->getCost(mx, my);
      if (center_cost >= (circle ? nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE :
        nav2_costmap_2d::LETHAL_OBSTACLE))
      {
        cell(mx, my);
        return;
      }
      if (circle) {
        off_map(x, y);  // a circular body is not located beyond its centre here
        return;
      }
      const double c = std::cos(yaw), sn = std::sin(yaw);
      for (const auto & vertex : footprint) {  // the production pre-check, same arithmetic
        const double wx = x + c * vertex.x - sn * vertex.y;
        const double wy = y + sn * vertex.x + c * vertex.y;
        if (!in_bounds(wx, wy)) {
          off_map(wx, wy);
          return;
        }
      }
      // FootprintCollisionChecker::footprintCostAtPose: same vertex arithmetic, edges
      // i -> i + 1, then the closing edge vertex 0 -> last; it stops at the first LETHAL
      // cell and otherwise keeps the maximum (NO_INFORMATION also rejects).
      const double cos_th = cos(yaw);
      const double sin_th = sin(yaw);
      std::vector<std::pair<unsigned int, unsigned int>> v;
      for (const auto & vertex : footprint) {
        const double wx = x + (vertex.x * cos_th - vertex.y * sin_th);
        const double wy = y + (vertex.x * sin_th + vertex.y * cos_th);
        unsigned int vx, vy;
        if (!costmap_->worldToMap(wx, wy, vx, vy)) {
          off_map(wx, wy);
          return;
        }
        v.emplace_back(vx, vy);
      }
      std::vector<std::pair<size_t, size_t>> edges;
      for (size_t k = 0; k + 1 < v.size(); ++k) {edges.emplace_back(k, k + 1);}
      edges.emplace_back(0, v.size() - 1);
      bool unknown = false;
      unsigned int ux = 0, uy = 0;
      for (const auto & e : edges) {
        for (nav2_util::LineIterator line(v[e.first].first, v[e.first].second,
          v[e.second].first, v[e.second].second); line.isValid(); line.advance())
        {
          const auto cost = costmap_->getCost(line.getX(), line.getY());
          if (cost == nav2_costmap_2d::LETHAL_OBSTACLE) {
            cell(line.getX(), line.getY());
            return;
          }
          if (cost > nav2_costmap_2d::LETHAL_OBSTACLE && !unknown) {
            unknown = true;
            ux = line.getX();
            uy = line.getY();
          }
        }
      }
      if (unknown) {cell(ux, uy);}
    };
  if (report) {
    report->first_index = -2;
    report->forecast_x.resize(n + 1);
    report->forecast_y.resize(n + 1);
    report->forecast_yaw.resize(n + 1);
    for (size_t j = 0; j <= n; ++j) {
      report->forecast_x[j] = trajectory.x(0, j);
      report->forecast_y[j] = trajectory.y(0, j);
      report->forecast_yaw[j] = trajectory.yaws(0, j);
    }
  }
  // Never rely on forward prediction to notice an already-occupied footprint.
  if (!clear_pose(p.x, p.y, tf2::getYaw(q))) {
    if (report) {
      report->first_index = -1;
      locate(p.x, p.y, tf2::getYaw(q));
    }
    return false;
  }
  for (size_t j = 0; j <= n; ++j) {
    if (!clear_pose(trajectory.x(0, j), trajectory.y(0, j), trajectory.yaws(0, j))) {
      if (report) {
        report->first_index = static_cast<int>(j);
        locate(trajectory.x(0, j), trajectory.y(0, j), trajectory.yaws(0, j));
      }
      return false;
    }
  }
  return true;
}

geometry_msgs::msg::TwistStamped Optimizer::getValidatedControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp)
{
  const bool tracing = trace_cycles_ || trace_rejections_;
  // Validate raw controls before clamping can turn NaN/Inf into a finite bound.
  if (controlSequenceFinite()) {
    // Smoothing follows optimization and can invalidate the Ackermann cone.
    // No velocity or steering adjustment is permitted after this validation.
    applyControlSequenceConstraints();
    if (tracing) {trace_final_ = control_sequence_;}
    if (sequenceCollisionFree(
        control_sequence_, settings_.shift_control_sequence,
        tracing ? &trace_report_ : nullptr))
    {
      const auto control = getControlFromSequenceAsTwist(stamp);
      control_history_.back() = {static_cast<float>(control.twist.linear.x),
        static_cast<float>(control.twist.linear.y), static_cast<float>(control.twist.angular.z)};
      return control;
    }
  }
  if (tracing && !controlSequenceFinite()) {
    trace_final_ = control_sequence_;
    trace_final_nonfinite_ = true;
  }
  control_sequence_.reset(settings_.time_steps);
  control_history_.fill({0.0f, 0.0f, 0.0f});
  throw nav2_core::NoValidControl("Final MPPI control sequence is not collision-free");
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp)
{
  unsigned int offset = settings_.shift_control_sequence ? 1 : 0;

  auto vx = control_sequence_.vx(offset);
  auto wz = control_sequence_.wz(offset);

  if (isHolonomic()) {
    auto vy = control_sequence_.vy(offset);
    return utils::toTwistStamped(vx, vy, wz, stamp, costmap_ros_->getBaseFrameID());
  }

  return utils::toTwistStamped(vx, wz, stamp, costmap_ros_->getBaseFrameID());
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_shared<OmniMotionModel>();
  } else if (model == "Ackermann") {
    motion_model_ = std::make_shared<AckermannMotionModel>(parameters_handler_, name_);
  } else {
    throw nav2_core::ControllerException(
            std::string(
              "Model " + model + " is not valid! Valid options are DiffDrive, Omni, "
              "or Ackermann"));
  }
  motion_model_->initialize(settings_.constraints, settings_.model_dt);
}

void Optimizer::setSpeedLimit(double speed_limit, bool percentage)
{
  auto & s = settings_;
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    s.constraints.vx_max = s.base_constraints.vx_max;
    s.constraints.vx_min = s.base_constraints.vx_min;
    s.constraints.vy = s.base_constraints.vy;
    s.constraints.wz = s.base_constraints.wz;
  } else {
    if (percentage) {
      // Speed limit is expressed in % from maximum speed of robot
      double ratio = speed_limit / 100.0;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    } else {
      // Speed limit is expressed in absolute value
      double ratio = speed_limit / s.base_constraints.vx_max;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    }
  }
  motion_model_->initialize(settings_.constraints, settings_.model_dt);
}

models::Trajectories & Optimizer::getGeneratedTrajectories()
{
  return generated_trajectories_;
}

namespace
{
uint64_t fnv1a(uint64_t h, const void * data, size_t n)
{
  const auto * p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

uint64_t costmapHash(const nav2_costmap_2d::Costmap2D & cm)
{
  uint64_t h = 1469598103934665603ULL;
  const unsigned int sx = cm.getSizeInCellsX(), sy = cm.getSizeInCellsY();
  const double res = cm.getResolution(), ox = cm.getOriginX(), oy = cm.getOriginY();
  h = fnv1a(h, &sx, sizeof(sx));
  h = fnv1a(h, &sy, sizeof(sy));
  h = fnv1a(h, &res, sizeof(res));
  h = fnv1a(h, &ox, sizeof(ox));
  h = fnv1a(h, &oy, sizeof(oy));
  return fnv1a(h, cm.getCharMap(), static_cast<size_t>(sx) * sy);
}

void copySequence(
  const mppi::models::ControlSequence & s, std::vector<float> & vx, std::vector<float> & wz)
{
  vx.assign(s.vx.begin(), s.vx.end());
  wz.assign(s.wz.begin(), s.wz.end());
}
}  // namespace

void Optimizer::publishTrace(
  uint8_t result, const geometry_msgs::msg::TwistStamped * command)
{
  try {
    publishTraceUnguarded(result, command);
  } catch (const std::exception & e) {
    RCLCPP_WARN(logger_, "MPPI trace not published: %s", e.what());
  } catch (...) {
    RCLCPP_WARN(logger_, "MPPI trace not published");
  }
}

void Optimizer::publishTraceUnguarded(
  uint8_t result, const geometry_msgs::msg::TwistStamped * command)
{
  using nav2_critics_msgs::msg::MppiCycle;
  auto node = parent_.lock();
  if (!node) {return;}
  const bool shift = settings_.shift_control_sequence;
  const bool rejected = result == MppiCycle::RESULT_REJECTED_COLLISION;
  const bool want_cycle = trace_cycle_pub_ && trace_cycle_pub_->is_activated();
  const bool want_dump = rejected && trace_rejection_pub_ && trace_rejection_pub_->is_activated();
  if (!want_cycle && !want_dump) {return;}

  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
  const uint64_t hash = costmapHash(*costmap_);
  const auto stamp = node->get_clock()->now();
  const double yaw = tf2::getYaw(state_.pose.pose.orientation);

  // Stage checks: the production check on earlier sequences (no side effects).
  const auto stage = [&](const models::ControlSequence & seq, bool shifted) {
      if (seq.vx.size() != settings_.time_steps) {return -3;}
      CollisionReport r;
      return sequenceCollisionFree(seq, shifted, &r) ? -2 : r.first_index;
    };

  // Rejected cycles: run the production check on every rollout of the last iteration.
  const size_t batch = state_.cvx.shape(0);
  const size_t steps = state_.cvx.shape(1);
  std::vector<int32_t> rollout_first;
  int32_t clear_unshifted = -1, pass_final = -1, best_rank = -1, cheapest_first = -3;
  if (rejected && batch > 0 && steps == settings_.time_steps &&
    trace_total_costs_.size() == batch)
  {
    rollout_first.assign(batch, -2);
    clear_unshifted = 0;
    pass_final = 0;
    models::ControlSequence one;
    one.reset(steps);
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < steps; ++j) {
        one.vx(j) = state_.cvx(i, j);
        one.wz(j) = state_.cwz(i, j);
        one.vy(j) = isHolonomic() ? state_.cvy(i, j) : 0.0f;
      }
      CollisionReport r;
      if (sequenceCollisionFree(one, shift, &r)) {
        ++pass_final;
      } else {
        rollout_first[i] = r.first_index;
      }
      clear_unshifted += sequenceCollisionFree(one, false, nullptr) ? 1 : 0;
    }
    std::vector<size_t> order(batch);
    for (size_t i = 0; i < batch; ++i) {order[i] = i;}
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return trace_total_costs_(a) < trace_total_costs_(b);
      });
    cheapest_first = rollout_first[order[0]];
    for (size_t r = 0; r < batch; ++r) {
      if (rollout_first[order[r]] == -2) {
        best_rank = static_cast<int32_t>(r);
        break;
      }
    }
  }

  if (want_cycle) {
    auto m = std::make_unique<MppiCycle>();
    m->stamp = stamp;
    m->cycle = trace_cycle_;
    m->pose_x = state_.pose.pose.position.x;
    m->pose_y = state_.pose.pose.position.y;
    m->pose_yaw = yaw;
    m->speed_vx = state_.speed.linear.x;
    m->speed_wz = state_.speed.angular.z;
    m->plan_stamp = trace_plan_stamp_;
    m->plan_size = static_cast<uint32_t>(trace_plan_size_);
    m->costmap_hash = hash;
    m->costmap_origin_x = costmap_->getOriginX();
    m->costmap_origin_y = costmap_->getOriginY();
    copySequence(trace_warm_, m->warm_vx, m->warm_wz);
    copySequence(trace_opt_, m->opt_vx, m->opt_wz);
    copySequence(trace_smooth_, m->smooth_vx, m->smooth_wz);
    copySequence(trace_final_, m->final_vx, m->final_wz);
    m->result = result;
    m->cmd_vx = command ? command->twist.linear.x : 0.0;
    m->cmd_wz = command ? command->twist.angular.z : 0.0;
    m->first_collision_index = trace_report_.first_index;
    m->collision_cell_x = trace_report_.cell_x;
    m->collision_cell_y = trace_report_.cell_y;
    m->collision_cell_cost = trace_report_.cell_cost;
    m->opt_unshifted_first_collision = stage(trace_opt_, false);
    m->opt_shifted_first_collision = stage(trace_opt_, shift);
    m->smooth_unshifted_first_collision = stage(trace_smooth_, false);
    m->smooth_shifted_first_collision = stage(trace_smooth_, shift);
    m->rollouts = static_cast<uint32_t>(batch);
    m->rollouts_clear_unshifted = clear_unshifted;
    m->rollouts_pass_final_check = pass_final;
    m->best_passing_rollout_rank = best_rank;
    m->cheapest_rollout_final_collision = cheapest_first;
    trace_cycle_pub_->publish(std::move(m));
  }

  if (want_dump) {
    auto d = std::make_unique<nav2_critics_msgs::msg::MppiRejection>();
    d->stamp = stamp;
    d->cycle = trace_cycle_;
    d->size_x = costmap_->getSizeInCellsX();
    d->size_y = costmap_->getSizeInCellsY();
    d->resolution = costmap_->getResolution();
    d->origin_x = costmap_->getOriginX();
    d->origin_y = costmap_->getOriginY();
    const auto * cells = costmap_->getCharMap();
    d->costs.assign(cells, cells + static_cast<size_t>(d->size_x) * d->size_y);
    d->costmap_hash = hash;
    for (const auto & v : costmap_ros_->getRobotFootprint()) {
      d->footprint_x.push_back(v.x);
      d->footprint_y.push_back(v.y);
    }
    d->pose_x = state_.pose.pose.position.x;
    d->pose_y = state_.pose.pose.position.y;
    d->pose_yaw = yaw;
    d->speed_vx = state_.speed.linear.x;
    d->speed_wz = state_.speed.angular.z;
    for (const auto & p : trace_plan_.poses) {
      d->plan_x.push_back(p.pose.position.x);
      d->plan_y.push_back(p.pose.position.y);
      d->plan_yaw.push_back(tf2::getYaw(p.pose.orientation));
    }
    d->goal_x = goal_.position.x;
    d->goal_y = goal_.position.y;
    d->goal_yaw = tf2::getYaw(goal_.orientation);
    d->forecast_x = trace_report_.forecast_x;
    d->forecast_y = trace_report_.forecast_y;
    d->forecast_yaw = trace_report_.forecast_yaw;
    d->first_collision_index = trace_report_.first_index;
    d->batch_size = static_cast<uint32_t>(batch);
    d->time_steps = static_cast<uint32_t>(steps);
    d->rollout_vx.assign(state_.cvx.begin(), state_.cvx.end());
    d->rollout_wz.assign(state_.cwz.begin(), state_.cwz.end());
    d->critic_names = critic_manager_.capturedCriticNames();
    for (const auto & c : critic_manager_.capturedCriticCosts()) {
      d->critic_costs.insert(d->critic_costs.end(), c.begin(), c.end());
    }
    d->total_costs.assign(trace_total_costs_.begin(), trace_total_costs_.end());
    d->softmax_weights.assign(trace_softmax_.begin(), trace_softmax_.end());
    d->rollout_final_check_first_collision = rollout_first;
    trace_rejection_pub_->publish(std::move(d));
  }
}

}  // namespace mppi
