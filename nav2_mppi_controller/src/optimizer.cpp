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

#include <limits>
#include <cstring>
#include <cstdint>
#include "nav2_mppi_controller/optimizer.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_core/controller_exceptions.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_mppi_controller/tools/diagnostic_capture.hpp"

namespace mppi
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

namespace
{
class DirectionSamplingFailure : public nav2_core::NoValidControl
{
public:
  DirectionSamplingFailure()
  : nav2_core::NoValidControl("No finite samples in the real path direction") {}
};

template<typename Tensor>
void snapshotTensor(std::ostream & out, const Tensor & value)
{
  out << "{\"shape\":[";
  bool first = true;
  for (auto size : value.shape()) {
    if (!first) {out << ',';}
    out << size; first = false;
  }
  out << "],\"data\":["; first = true;
  for (auto scalar : value) {
    if (!std::isfinite(static_cast<double>(scalar))) {
      throw std::runtime_error("Non-finite optimizer snapshot input");
    }
    if (!first) {out << ',';}
    out << scalar; first = false;
  }
  out << "]}";
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
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

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
}

void Optimizer::on_deactivate()
{
  critic_manager_.on_deactivate();
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
  getParam(smooth_control_sequence_, "smooth_control_sequence", true, ParameterType::Static);
  getParam(condition_control_on_path_direction_, "condition_control_on_path_direction",
    false, ParameterType::Static);
  getParam(allow_path_direction_braking_, "allow_path_direction_braking",
    false, ParameterType::Static);
  getParam(condition_terminal_path_direction_, "condition_terminal_path_direction",
    false, ParameterType::Static);
  getParam(seed_short_cusp_on_handoff_, "seed_short_cusp_on_handoff",
    false, ParameterType::Static);
  getParam(preserve_short_cusp_feasibility_, "preserve_short_cusp_feasibility",
    false, ParameterType::Static);
  getParam(extend_short_cusp_completion_, "extend_short_cusp_completion",
    false, ParameterType::Static);
  if (extend_short_cusp_completion_ &&
    (!seed_short_cusp_on_handoff_ || !preserve_short_cusp_feasibility_))
  {
    throw nav2_core::ControllerException(
            "extend_short_cusp_completion requires short cusp seed and feasibility preservation");
  }
  getParam(initialize_long_cusp_entry_, "initialize_long_cusp_entry",
    false, ParameterType::Static);
  getParam(repair_long_cusp_warm_start_, "repair_long_cusp_warm_start",
    false, ParameterType::Static);
  getParam(preserve_filtered_collision_feasibility_, "preserve_filtered_collision_feasibility",
    false, ParameterType::Static);
  getParam(retain_collision_free_long_path_sample_, "retain_collision_free_long_path_sample",
    false, ParameterType::Static);
  if (retain_collision_free_long_path_sample_ && !preserve_filtered_collision_feasibility_) {
    throw nav2_core::ControllerException(
            "retain_collision_free_long_path_sample requires preserve_filtered_collision_feasibility");
  }
  if (repair_long_cusp_warm_start_ && !initialize_long_cusp_entry_) {
    throw nav2_core::ControllerException(
            "repair_long_cusp_warm_start requires initialize_long_cusp_entry");
  }
  if (initialize_long_cusp_entry_ &&
    (!seed_short_cusp_on_handoff_ || !preserve_short_cusp_feasibility_))
  {
    throw nav2_core::ControllerException(
            "initialize_long_cusp_entry requires short cusp seed and feasibility preservation");
  }
  if (preserve_short_cusp_feasibility_ && !seed_short_cusp_on_handoff_) {
    throw nav2_core::ControllerException(
            "preserve_short_cusp_feasibility requires seed_short_cusp_on_handoff");
  }
  getParam(motor_deadband_floor_, "motor_deadband_floor", 0.0f);
  getParam(motor_deadband_approach_radius_, "motor_deadband_approach_radius", 0.15f);
  getParam(motor_deadband_terminal_radius_, "motor_deadband_terminal_radius", 0.04f);
  getParam(motor_deadband_max_heading_error_, "motor_deadband_max_heading_error", 0.5236f);

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
  clearCuspCompletion();
  effective_cusp_tolerances_.reset();
  short_cusp_seed_requested_ = false;
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
  // Keep the shifted sequence carried into this update. A rejected noisy mean
  // must not restart a crawl-then-cruise delay if that sequence is still valid.
  std::optional<models::ControlSequence> warm_start;
  if (preserve_short_cusp_feasibility_ && active_cusp_direction_ != 0) {
    warm_start = control_sequence_;
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

  bool retry;
  do {
    try {
      optimize(diag);
    } catch (const DirectionSamplingFailure &) {
      // A warm sequence can lose all post-filter support in the active gear.
      // Retry from a fresh sequence using the existing bounded fallback, not
      // the same rejected batch on every controller tick. Other exceptions
      // retain their original propagation behavior.
      critics_data_.fail_flag = true;
    }
    retry = fallback(critics_data_.fail_flag, warm_start ? &*warm_start : nullptr);
    if (retry) {
      critics_data_.fail_flag = false;
      warm_start.reset();  // Fallback reset invalidates the prior sequence.
    }
  } while (retry);

  // [E] POST-OPTIMIZE (after softmax + accel constraints)
  if (diag && ts >= 5) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] E.post-opt  ctrl[0..4]=(%.4f,%.4f,%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f)",
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.vx(3), control_sequence_.vx(4),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2));
  }

  std::optional<models::ControlSequence> before_filter;
  if (preserve_filtered_collision_feasibility_) {
    before_filter = control_sequence_;
  }
  smoothControlSequence();

  // [F] POST-SG FILTER
  if (diag && ts >= 5) {
    RCLCPP_DEBUG(logger_,
      "[MPPI] F.post-SG   ctrl[0..4]=(%.4f,%.4f,%.4f,%.4f,%.4f) wz(%.3f,%.3f,%.3f) smoothing_enabled=%d",
      control_sequence_.vx(0), control_sequence_.vx(1), control_sequence_.vx(2),
      control_sequence_.vx(3), control_sequence_.vx(4),
      control_sequence_.wz(0), control_sequence_.wz(1), control_sequence_.wz(2),
      smooth_control_sequence_ ? 1 : 0);
  }

  auto control = getControlFromSequenceAsTwist(
    plan.header.stamp, warm_start ? &*warm_start : nullptr,
    before_filter ? &*before_filter : nullptr);

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

void Optimizer::smoothControlSequence()
{
  if (smooth_control_sequence_) {
    utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);
  } else {
    // Keep command history chronological even without filtering. Extraction
    // writes the final constrained command into the newest history entry.
    control_history_[0] = control_history_[1];
    control_history_[1] = control_history_[2];
    control_history_[2] = control_history_[3];
  }
}

void Optimizer::optimize(bool diag)
{
  // Explicit office diagnostic only. No new ROS parameters or class ABI fields.
  // Capture once per process after sustained stationary controls near a real endpoint.
  static const char * snapshot_path = std::getenv("MPPI_STALL_SNAPSHOT_PATH");
  static unsigned int stationary_cycles = 0;
  static bool snapshot_attempted = false;
  static bool successful_snapshot_attempted = false;
  static const char * successful_capture_option =
    std::getenv("MPPI_SUCCESSFUL_BATCH_AFTER_STAMP");
  static const auto successful_minimum_stamp = successful_capture_option ?
    diagnostic::minimumCaptureStamp(successful_capture_option) : std::nullopt;
  static const char * gear_capture = std::getenv("MPPI_SNAPSHOT_ON_GEAR_CHANGE");
  static const char * singleton_capture = std::getenv("MPPI_SNAPSHOT_ON_FIRST_SINGLETON");
  static const char * final_capture = std::getenv("MPPI_SNAPSHOT_ON_FINAL_SINGLETON");
  static const char * capture_gear = std::getenv("MPPI_SNAPSHOT_CUSP_DIRECTION");
  static const char * failed_batch_capture = std::getenv("MPPI_SNAPSHOT_ON_FIRST_FAILED_BATCH");
  const bool capture_failed_batch = failed_batch_capture &&
    std::string(failed_batch_capture) == "1";
  static const char * stationary_failed_capture =
    std::getenv("MPPI_FAILED_BATCH_STATIONARY_ONLY");
  const bool stationary_failed_only = stationary_failed_capture &&
    std::string(stationary_failed_capture) == "1";
  static const auto failed_capture_minimum_stamp = diagnostic::minimumCaptureStamp(
    std::getenv("MPPI_FAILED_BATCH_AFTER_STAMP"));
  const bool failed_capture_eligible = diagnostic::failedBatchCaptureEligible(
    rclcpp::Time(state_.pose.header.stamp).seconds(), state_.speed.linear.x,
    state_.speed.angular.z, stationary_failed_only, failed_capture_minimum_stamp);
  static const char * recent_capture_option = std::getenv("MPPI_FAILED_BATCH_RECENT_RING");
  static diagnostic::FailedBatchRing recent_failed_batches;
  const double capture_stamp = rclcpp::Time(state_.pose.header.stamp).seconds();
  const bool recent_capture_enabled = recent_capture_option &&
    std::string(recent_capture_option) == "1";
  static int previous_capture_gear = 0;
  bool capture_cycle = false;
  const char * capture_reason = "stationary_endpoint";
  if (snapshot_path && *snapshot_path && !snapshot_attempted) {
      const size_t real = std::min(
        critics_data_.real_path_point_count.value_or(0), path_.x.size());
      const double distance = real ? std::hypot(
        path_.x(real - 1) - state_.pose.pose.position.x,
        path_.y(real - 1) - state_.pose.pose.position.y) : 0.0;
      const bool stationary = std::abs(state_.speed.linear.x) < 0.002 &&
        distance > 0.05 && distance < 0.20;
      // Inspect the carried command once per cycle. Internal optimization can
      // produce larger exploratory controls which smoothing later cancels.
      stationary_cycles = stationary ? stationary_cycles + 1 : 0;
      capture_cycle = stationary_cycles >= 10;
      if (gear_capture && std::string(gear_capture) == "1") {
        capture_cycle = previous_capture_gear != 0 && active_cusp_direction_ != 0 &&
          previous_capture_gear != active_cusp_direction_;
        if (active_cusp_direction_ != 0) {previous_capture_gear = active_cusp_direction_;}
        capture_reason = "first_gear_change";
      }
      // Optional endpoint diagnostic takes precedence over the gear-change
      // trigger. Capture the actual one-point prefix left after pruning, while
      // its cusp is still active, rather than an earlier transition. Ghost
      // padding is deliberately excluded by the real-point metadata above.
      if (singleton_capture && std::string(singleton_capture) == "1") {
        const bool requested_gear = !capture_gear ||
          (std::string(capture_gear) == "1" && active_cusp_direction_ == 1) ||
          (std::string(capture_gear) == "-1" && active_cusp_direction_ == -1);
        capture_cycle = real == 1 && active_cusp_direction_ != 0 && requested_gear;
        capture_reason = "first_active_cusp_singleton";
      }
      if (final_capture && std::string(final_capture) == "1") {
        capture_cycle = real == 1 && active_cusp_direction_ == 0;
        capture_reason = "first_final_singleton";
      }
      if (capture_failed_batch) {capture_cycle = false;}
  }
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    bool capture = capture_cycle && i + 1 == settings_.iteration_count;
    bool capture_successful = snapshot_path && *snapshot_path &&
      !successful_snapshot_attempted && i + 1 == settings_.iteration_count &&
      diagnostic::successfulBatchCaptureEligible(capture_stamp,
      critics_data_.real_path_point_count.value_or(0), active_cusp_direction_,
      successful_minimum_stamp);
    capture = capture || capture_successful;
    bool capture_recent = false;
    unsigned int recent_slot = 0;
    const bool recent_eligible = capture_failed_batch && failed_capture_eligible &&
      recent_failed_batches.eligible(recent_capture_enabled, capture_stamp);
    if (capture && !capture_successful) {snapshot_attempted = true;}
    std::ostringstream snapshot;
    xt::xtensor<float, 2> unbounded_cvx, unbounded_cwz;
    bool unbounded_available = false;
    if (capture || (capture_failed_batch && failed_capture_eligible && snapshot_path && *snapshot_path &&
      (!snapshot_attempted || recent_eligible)))
    {
      // A second read of a fixed noise batch does not consume RNG draws.
      // Do not claim exact pre-prediction inputs for asynchronous regeneration:
      // that thread can replace the batch between this read and generation.
      auto node = parent_.lock();
      if (node && !node->get_parameter(name_ + ".regenerate_noises").as_bool()) {
        noise_generator_.setNoisedControls(state_, control_sequence_);
        unbounded_cvx = state_.cvx;
        unbounded_cwz = state_.cwz;
        unbounded_available = true;
      }
    }
    generateNoisedTrajectories();

    // Each iteration regenerates rollouts around the updated control sequence.
    // Indices are reused, so a previous candidate's collision cost cannot be
    // carried onto its new trajectory. Score the current batch from zero.
    costs_.fill(0.0f);

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

    critic_manager_.evalTrajectoriesScores(critics_data_);

    if (capture_successful) {
      if (critics_data_.fail_flag) {
        capture_successful = false;
        capture = false;  // The independent failed-batch policy below still runs.
      } else {
        successful_snapshot_attempted = true;
        capture_reason = "targeted_successful_critic_batch";
      }
    }

    if (capture_failed_batch && failed_capture_eligible && snapshot_path && *snapshot_path &&
      (!snapshot_attempted || recent_eligible) && critics_data_.fail_flag)
    {
      capture = true;
      capture_recent = snapshot_attempted;
      if (recent_eligible) {recent_slot = recent_failed_batches.take(capture_stamp);}
      snapshot_attempted = true;
      capture_reason = capture_recent ? "recent_failed_critic_batch" : "first_failed_critic_batch";
    }

    if (capture) {
      try {
        snapshot << std::setprecision(17) << "{\"scope\":\"One native optimizer iteration before smoothing; optional diagnostic capture may delay this tick\","
                 << "\"capture_reason\":\"" << capture_reason << "\","
                 << "\"failed_batch_stationary_filter\":" <<
          (stationary_failed_only ? "true" : "false") << ","
                 << "\"failed_batch_minimum_stamp\":";
        if (failed_capture_minimum_stamp &&
          diagnostic::isFiniteCaptureValue(*failed_capture_minimum_stamp))
        {
          snapshot << *failed_capture_minimum_stamp;
        } else {
          snapshot << "null";
        }
        snapshot << ','
                 << "\"stamp\":" << state_.pose.header.stamp.sec + state_.pose.header.stamp.nanosec * 1e-9
                 << ",\"iteration\":" << i << ",\"real_path_point_count\":"
                 << critics_data_.real_path_point_count.value_or(0)
                 << ",\"active_cusp_direction\":" << active_cusp_direction_
          << ",\"effective_cusp_tolerances\":" << serializedCuspTolerances()
                 << ",\"path_direction\":" << path_direction_
                 << ",\"furthest\":" << critics_data_.furthest_reached_path_point.value_or(0)
                 << ",\"temperature\":" << settings_.temperature
                 << ",\"gamma\":" << settings_.gamma
                 << ",\"vx_std\":" << settings_.sampling_std.vx
                 << ",\"wz_std\":" << settings_.sampling_std.wz
                 << ",\"model_dt\":" << settings_.model_dt
                 << ",\"fail_flag\":" << (critics_data_.fail_flag ? "true" : "false")
                 << ",\"goal\":[" << goal_.position.x << ',' << goal_.position.y << ','
                 << tf2::getYaw(goal_.orientation) << ']'
                 << ",\"robot\":[" << state_.pose.pose.position.x << ','
                 << state_.pose.pose.position.y << ',' << tf2::getYaw(state_.pose.pose.orientation)
                 << "],\"speed\":[" << state_.speed.linear.x << ',' << state_.speed.angular.z << ']';
        const auto ackermann = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
        const auto feedback = [&](const char * name, const std::optional<float> & value) {
            snapshot << ",\"" << name << "\":";
            if (value) {snapshot << *value;} else {snapshot << "null";}
          };
        feedback("initial_steering", ackermann ? ackermann->initialSteering() : std::nullopt);
        feedback("initial_conditioned_steering",
          ackermann ? ackermann->initialConditionedSteering() : std::nullopt);
        const auto field = [&](const char * name, const auto & tensor) {
            snapshot << ",\"" << name << "\":"; snapshotTensor(snapshot, tensor);
          };
        field("control_vx_before", control_sequence_.vx);
        field("control_wz_before", control_sequence_.wz);
        field("sampled_cvx", state_.cvx); field("sampled_cwz", state_.cwz);
        snapshot << ",\"unbounded_available\":" << (unbounded_available ? "true" : "false");
        if (unbounded_available) {
          field("unbounded_cvx", unbounded_cvx);
          field("unbounded_cwz", unbounded_cwz);
        }
        field("predicted_vx", state_.vx); field("predicted_wz", state_.wz);
        field("trajectory_x", generated_trajectories_.x);
        field("trajectory_y", generated_trajectories_.y);
        field("trajectory_yaw", generated_trajectories_.yaws);
        field("path_x", path_.x); field("path_y", path_.y); field("path_yaw", path_.yaws);
        field("critic_costs", costs_);
        snapshot << ",\"history\":[";
        for (size_t k = 0; k < control_history_.size(); ++k) {
          if (k) {snapshot << ',';}
          snapshot << '[' << control_history_[k].vx << ',' << control_history_[k].wz << ']';
        }
        snapshot << ']';
        std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
        snapshot << ",\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
                 << ",\"height\":" << costmap_->getSizeInCellsY()
                 << ",\"resolution\":" << costmap_->getResolution()
                 << ",\"origin\":[" << costmap_->getOriginX() << ',' << costmap_->getOriginY()
                 << "],\"data\":[";
        const size_t cells = costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY();
        for (size_t k = 0; k < cells; ++k) {
          if (k) {snapshot << ',';}
          snapshot << static_cast<unsigned int>(costmap_->getCharMap()[k]);
        }
        snapshot << "]}";
        if (capture_successful) {
          std::ofstream file(std::string(snapshot_path) + ".successful_batch.pre_update.json");
          if (!file) {throw std::runtime_error("Cannot open successful batch pre-update snapshot");}
          file << snapshot.str() << ",\"control_update_completed\":false}\n";
          file.close();
          if (!file) {throw std::runtime_error("Cannot finish successful batch pre-update snapshot");}
        }
        if (capture_failed_batch && !capture_successful) {
          // Preserve the failed batch before the weighted update can throw or
          // reset it. This sidecar is independent of any earlier gear snapshot.
          const auto suffix = capture_recent ?
            ".failed_batch_recent_" + std::to_string(recent_slot) + ".json" : ".failed_batch.json";
          std::ofstream file(std::string(snapshot_path) + suffix);
          if (!file) {throw std::runtime_error("Cannot open failed batch snapshot");}
          file << snapshot.str() << ",\"control_update_completed\":false}\n";
          file.close();
          if (!file) {throw std::runtime_error("Cannot finish failed batch snapshot");}
          // Recent slots preserve the pre-update failed batch only; leave the
          // original one-shot post-update snapshot untouched.
          if (capture_recent) {capture = false;}
        }
      } catch (const std::exception & error) {
        capture = false;
        RCLCPP_WARN(logger_, "Optional optimizer snapshot skipped: %s", error.what());
      }
    }

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

    // An all-rejected critic batch is not a distribution to optimize. Keep the
    // carried sequence intact for fallback validation, and retain the rejected
    // costs instead of running later unscored iterations under a sticky flag.
    // evalControl owns the existing bounded retry / warm-start fallback policy.
    if (critics_data_.fail_flag) {return;}
    updateControlSequence(diag);
    if (capture) {
      try {
        snapshot << ",\"regularized_costs\":"; snapshotTensor(snapshot, costs_);
        snapshot << ",\"control_vx_after\":"; snapshotTensor(snapshot, control_sequence_.vx);
        snapshot << ",\"control_wz_after\":"; snapshotTensor(snapshot, control_sequence_.wz);
        snapshot << "}\n";
        const std::string output_path = capture_successful ?
          std::string(snapshot_path) + ".successful_batch.json" : snapshot_path;
        std::ofstream file(output_path, std::ios::out | std::ios::trunc);
        if (!file) {throw std::runtime_error("Cannot open optimizer snapshot output");}
        file << snapshot.str(); file.close();
        if (!file) {throw std::runtime_error("Cannot finish optimizer snapshot output");}
        RCLCPP_INFO(logger_, "Optional optimizer snapshot written to %s", output_path.c_str());
      } catch (const std::exception & error) {
        RCLCPP_WARN(logger_, "Optional optimizer snapshot failed: %s", error.what());
      }
    }
  }
}

bool Optimizer::fallback(bool fail, const models::ControlSequence * warm_start)
{
  if (fail && long_entry_required_) {
    enforceLongEntryCertificate(false);
    critics_data_.fail_flag=false;fallback(false);return false;
  }
  if (fail && tryExtendedShortCuspContinuation(false)) {
    critics_data_.fail_flag = false;
    fallback(false);
    return false;
  }
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  if (fail && preserve_short_cusp_feasibility_ && seed_short_cusp_on_handoff_ &&
    active_cusp_direction_ != 0 && real > 0 && real <= 4 && warm_start &&
    warm_start->vx.size() == settings_.time_steps &&
    warm_start->wz.size() == settings_.time_steps &&
    std::hypot(path_.x(real - 1) - state_.pose.pose.position.x,
    path_.y(real - 1) - state_.pose.pose.position.y) <= 0.12f)
  {
    // A failed noisy batch does not invalidate a carried feasible sequence.
    // Validate before reset() destroys it and restarts the seed's crawl delay.
    // This is only the existing strict short-cusp policy, not a long-path or
    // final-goal fallback. Current feedback, gear, constraints and the entire
    // footprint forecast still have to pass; no sampled cost is fabricated.
    const auto rejected = control_sequence_;
    control_sequence_ = *warm_start;
    try {
      applyControlSequenceConstraints();
      if (shortCuspSequenceFeasible()) {
        critics_data_.fail_flag = false;
        fallback(false);  // A validated continuation resets the retry counter.
        RCLCPP_INFO(logger_, "[CUSP_WARM_FALLBACK] retained validated continuation after sampled failure");
        return false;
      }
    } catch (...) {
      control_sequence_ = rejected;
      throw;
    }
    control_sequence_ = rejected;
  }
  return fallback(fail);
}

bool Optimizer::fallback(bool fail)
{
  static size_t counter = 0;

  if (!fail) {
    counter = 0;
    return false;
  }

  // A sampled-batch retry resets controls, not the active path's handoff gates.
  const auto gates = effective_cusp_tolerances_;
  const bool attempted_completion = cusp_completion_attempted_;
  const bool gentle_required = gentle_short_cusp_required_;
  const auto completion = gentle_required ? cusp_completion_ : std::nullopt;
  const auto completion_shifts = cusp_completion_shifts_;
  reset(false /*Don't reset zone-based speed limits after fallback*/);
  effective_cusp_tolerances_ = gates;
  cusp_completion_attempted_ = attempted_completion;
  gentle_short_cusp_required_ = gentle_required;
  if (gentle_required) {
    cusp_completion_ = completion;
    cusp_completion_shifts_ = completion_shifts;
  }

  if (++counter > settings_.retry_attempt_limit) {
    counter = 0;
    throw nav2_core::NoValidControl("Optimizer fail to compute path");
  }

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
  if ((long_entry_required_ || long_entry_seed_requested_) &&
    plan.header.frame_id != robot_pose.header.frame_id)
  {
    long_entry_.reset();long_entry_required_=true;
    long_entry_seed_requested_=false;short_cusp_seed_requested_=false;
    throw nav2_core::FailedToMakeProgress("Long-entry path frame changed; stopped replan required");
  }
  path_ = utils::toTensor(plan);
  costs_.fill(0.0f);
  goal_ = goal;

  critics_data_.fail_flag = false;
  critics_data_.goal_checker = goal_checker;
  critics_data_.motion_model = motion_model_;
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
  if (short_cusp_seed_requested_) {
    short_cusp_seed_requested_ = false;
    const bool actual_handoff=long_entry_seed_requested_;
    long_entry_seed_requested_=false;
    const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
    if (actual_handoff && initialize_long_cusp_entry_ && seed_short_cusp_on_handoff_ &&
      active_cusp_direction_==1 && real>4 && settings_.shift_control_sequence &&
      std::hypot(path_.x(real-1)-robot_pose.pose.position.x,
      path_.y(real-1)-robot_pose.pose.position.y)>0.8)
    {
      // Preserve an already executable ordinary-horizon mean. The longer
      // certificate is a fallback only for the diagnosed horizon mismatch.
      if (!initializeShortCuspMean() && !initializeLongEntryCertificate()) {
        throw nav2_core::FailedToMakeProgress("No bounded long-entry certificate; stopped replan required");
      }
    } else {initializeShortCuspMean();}
  }
  if (long_entry_required_) {
    // Physical alignment retires the future-tail obligation, never basic state,
    // lifecycle or feedback validity. Validate those before either branch.
    if (!refreshLongEntryBinding() || !longEntryStateValid()) {
      long_entry_.reset();
      throw nav2_core::FailedToMakeProgress("Invalid long-entry state; stopped replan required");
    }
    if (longEntryMeasuredAligned()) {long_entry_.reset();long_entry_required_=false;}
    else {enforceLongEntryCertificate(false);}
  }
}

bool Optimizer::pathBindingValid() const
{
  if(!path_binding_||!path_binding_->canonical_plan||!costmap_ros_) {return false;}
  const auto & b=*path_binding_;const auto & t=b.canonical_to_local;
  const auto finite=diagnostic::isFiniteCaptureValue;
  if(b.generation==0||b.segment==0||b.canonical_plan->header.frame_id.empty()||
    t.child_frame_id!=b.canonical_plan->header.frame_id||t.header.frame_id.empty()||
    t.header.frame_id!=state_.pose.header.frame_id||t.header.frame_id!=costmap_ros_->getGlobalFrameID()||
    rclcpp::Time(t.header.stamp)!=rclcpp::Time(state_.pose.header.stamp)||
    b.segment_begin>=b.segment_end||b.segment_end>b.canonical_plan->poses.size()) {return false;}
  const auto & q=t.transform.rotation;const auto & v=t.transform.translation;
  for(double value:{q.x,q.y,q.z,q.w,v.x,v.y,v.z}) {if(!finite(value)) {return false;}}
  if(std::abs(q.x)>1e-6||std::abs(q.y)>1e-6||std::abs(v.z)>1e-6||
    std::abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1)>1e-6) {return false;}
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  if(b.real_ids.size()!=real) {return false;}
  size_t prior=0,local_index=0;bool first=true;
  for(size_t id:b.real_ids) {
    if(id<b.segment_begin||id>=b.segment_end||(!first&&id<=prior)) {return false;}
    const auto & p=b.canonical_plan->poses[id].pose;
    for(double value:{p.position.x,p.position.y,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w})
    {if(!finite(value)) {return false;}}
    const auto & r=p.orientation;
    if(std::abs(r.x)>1e-6||std::abs(r.y)>1e-6||
      std::abs(r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w-1)>1e-6) {return false;}
    geometry_msgs::msg::PoseStamped canonical,local;
    canonical.header=b.canonical_plan->header;canonical.header.stamp=state_.pose.header.stamp;
    canonical.pose=p;tf2::doTransform(canonical,local,t);
    if(std::hypot(local.pose.position.x-path_.x(local_index),local.pose.position.y-path_.y(local_index))>1e-6||
      std::abs(angles::shortest_angular_distance(tf2::getYaw(local.pose.orientation),path_.yaws(local_index)))>1e-6)
    {return false;}
    ++local_index;prior=id;first=false;
  }
  return true;
}

bool Optimizer::longEntryIdentityValid() const
{
  if(!long_entry_) {return false;}
  const auto & c=*long_entry_;
  if(c.canonical_plan) {
    return pathBindingValid()&&path_binding_->canonical_plan==c.canonical_plan&&
      path_binding_->generation==c.plan_generation&&path_binding_->segment==c.segment_generation&&
      c.source_id>=path_binding_->segment_begin&&c.source_id<path_binding_->segment_end;
  }
  if(path_binding_required_) {return false;}
  size_t matches=0;
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  for(size_t k=0;k<real;++k) {
    if(std::hypot(path_.x(k)-c.target.position.x,path_.y(k)-c.target.position.y)<=1e-6&&
      std::abs(angles::shortest_angular_distance(path_.yaws(k),tf2::getYaw(c.target.orientation)))<=1e-6)
    {++matches;}
  }
  return matches==1;
}

bool Optimizer::refreshLongEntryBinding()
{
  if(!long_entry_) {return false;}
  auto & c=*long_entry_;
  if(!c.canonical_plan) {return !path_binding_required_;}
  if(!longEntryIdentityValid()) {return false;}
  const auto & t=path_binding_->canonical_to_local;
  geometry_msgs::msg::PoseStamped source,target,anchor;
  source.header.frame_id=c.canonical_plan->header.frame_id;source.header.stamp=state_.pose.header.stamp;
  source.pose=c.canonical_plan->poses[c.source_id].pose;
  tf2::doTransform(source,target,t);
  source.pose=c.canonical_anchor;tf2::doTransform(source,anchor,t);
  c.target=target.pose;c.anchor_x=anchor.pose.position.x;c.anchor_y=anchor.pose.position.y;
  return true;
}

bool Optimizer::longEntryStateValid() const
{
  const auto finite=diagnostic::isFiniteCaptureValue;
  const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if(!long_entry_||!effective_cusp_tolerances_||!costmap_||!costmap_ros_||!model||
    !settings_.shift_control_sequence||active_cusp_direction_!=1||path_direction_!=1||
    !model->predictsSteeringConditioning()||!model->initialSteering()||
    !finite(*model->initialSteering())||
    (model->requiresConditionedSteering()&&(!model->initialConditionedSteering()||
    !finite(*model->initialConditionedSteering())))) {return false;}
  const auto & c=*long_entry_;
  const auto gates=cuspTolerances();
  if(settings_.time_steps<2||settings_.time_steps>81||
    !finite(settings_.model_dt)||settings_.model_dt<=0||settings_.model_dt>0.15f||
    c.epoch!=long_entry_epoch_||c.frame.empty()||c.frame!=state_.pose.header.frame_id||
    c.frame!=costmap_ros_->getGlobalFrameID()||
    !finite(gates.first)||!finite(gates.second)||gates.first<=0||gates.second<=0||
    !finite(c.gates.first)||!finite(c.gates.second)||c.gates.first<=0||c.gates.second<=0||
    !finite(state_.pose.pose.position.x)||!finite(state_.pose.pose.position.y)||
    !finite(tf2::getYaw(state_.pose.pose.orientation))||!finite(state_.speed.linear.x)||
    !finite(state_.speed.angular.z)||!finite(c.anchor_x)||!finite(c.anchor_y)||
    !finite(c.edge)||c.edge<=0||!finite(c.target.position.x)||!finite(c.target.position.y)||
    !finite(tf2::getYaw(c.target.orientation))) {return false;}
  const size_t n=2*settings_.time_steps-1;
  const int64_t now=rclcpp::Time(state_.pose.header.stamp).nanoseconds();
  if(c.shifts>=n-1||now<c.stamp_ns||now<c.last_stamp_ns||
    static_cast<double>(now-c.stamp_ns)*1e-9>n*settings_.model_dt) {return false;}
  if(!longEntryIdentityValid()) {return false;}
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  for(size_t k=0;k<real;++k) {
    if(!finite(path_.x(k))||!finite(path_.y(k))||!finite(path_.yaws(k))) {return false;}
  }
  const auto footprint=costmap_ros_->getRobotFootprint();
  if(footprint.size()<3||costmap_ros_->getUseRadius()) {return false;}
  for(const auto & point:footprint) {if(!finite(point.x)||!finite(point.y)) {return false;}}
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const double cost=checker.footprintCostAtPose(state_.pose.pose.position.x,
    state_.pose.pose.position.y,tf2::getYaw(state_.pose.pose.orientation),footprint);
  return finite(cost)&&cost>=0&&cost<nav2_costmap_2d::LETHAL_OBSTACLE;
}

bool Optimizer::longEntryMeasuredAligned() const
{
  if (!long_entry_ || !effective_cusp_tolerances_ || active_cusp_direction_!=1) {return false;}
  const auto & c=*long_entry_;
  const auto finite=diagnostic::isFiniteCaptureValue;
  const double x=state_.pose.pose.position.x,y=state_.pose.pose.position.y;
  const double yaw=tf2::getYaw(state_.pose.pose.orientation);
  const auto gates=cuspTolerances();
  if (!finite(x)||!finite(y)||!finite(yaw)||!finite(gates.first)||!finite(gates.second)||
    state_.pose.header.frame_id!=c.frame || c.frame!=costmap_ros_->getGlobalFrameID()) {return false;}
  return longEntryIdentityValid() && std::hypot(x-c.anchor_x,y-c.anchor_y)+1e-6>=c.edge &&
    std::hypot(x-c.target.position.x,y-c.target.position.y)<=std::min(c.gates.first,gates.first) &&
    std::abs(angles::shortest_angular_distance(yaw,tf2::getYaw(c.target.orientation)))<=
    std::min(c.gates.second,gates.second);
}

bool Optimizer::validateLongEntrySequence(
  models::ControlSequence & sequence,size_t & first_hit,bool select_target)
{
  first_hit=sequence.vx.size();
  const auto finite=diagnostic::isFiniteCaptureValue;
  const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if(!long_entry_||!effective_cusp_tolerances_||!costmap_||!costmap_ros_||!model||
    !settings_.shift_control_sequence||active_cusp_direction_!=1||path_direction_!=1||
    !model->predictsSteeringConditioning()||!model->initialSteering()||
    !finite(*model->initialSteering())||
    (model->requiresConditionedSteering() && (!model->initialConditionedSteering()||
    !finite(*model->initialConditionedSteering())))) {return false;}
  auto & c=*long_entry_;
  const size_t n=sequence.vx.size();
  const auto gates=cuspTolerances();
  const int64_t now=rclcpp::Time(state_.pose.header.stamp).nanoseconds();
  if(settings_.time_steps<2||settings_.time_steps>81||n!=2*settings_.time_steps-1||
    n>161||sequence.wz.size()!=n||sequence.vy.size()!=n||c.shifts>=n-1||
    c.epoch!=long_entry_epoch_||c.frame.empty()||c.frame!=state_.pose.header.frame_id||
    c.frame!=costmap_ros_->getGlobalFrameID()||now<c.stamp_ns||now<c.last_stamp_ns||
    !finite(settings_.model_dt)||settings_.model_dt<=0||settings_.model_dt>0.15f||
    static_cast<double>(now-c.stamp_ns)*1e-9>n*settings_.model_dt||
    !finite(gates.first)||!finite(gates.second)||gates.first<=0||gates.second<=0||
    !finite(state_.pose.pose.position.x)||!finite(state_.pose.pose.position.y)||
    !finite(tf2::getYaw(state_.pose.pose.orientation))||!finite(state_.speed.linear.x)||
    !finite(state_.speed.angular.z)||!finite(c.edge)||c.edge<=0||
    !finite(c.anchor_x)||!finite(c.anchor_y)) {return false;}
  const auto & limits=settings_.constraints;
  for(double value : {static_cast<double>(limits.vx_max),static_cast<double>(limits.vx_min),
    static_cast<double>(limits.wz),static_cast<double>(limits.ax_max),
    static_cast<double>(limits.ax_min),static_cast<double>(limits.az_max)})
  {if(!finite(value)) {return false;}}
  if(limits.vx_max<=0||limits.ax_max<=0||limits.ax_min>=0||limits.az_max<=0||limits.wz<=0) {return false;}
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  if(real<2) {return false;}
  for(size_t k=0;k<real;++k) {
    if(!finite(path_.x(k))||!finite(path_.y(k))||!finite(path_.yaws(k))) {return false;}
  }
  if(!select_target && !longEntryIdentityValid()) {return false;}
  for(size_t j=0;j<n;++j) {
    if(!finite(sequence.vx(j))||!finite(sequence.wz(j))||sequence.vx(j)<0) {return false;}
  }
  const auto saved=control_sequence_;
  control_sequence_=sequence;
  try {applyControlSequenceConstraints();sequence=control_sequence_;}
  catch(...) {control_sequence_=saved;throw;}
  control_sequence_=saved;
  models::State predicted;predicted.reset(1,n);
  predicted.pose=state_.pose;predicted.speed=state_.speed;
  // Extraction emits index1. Forecast precisely that stream, not control0.
  for(size_t j=0;j<n;++j) {
    predicted.cvx(0,j)=j+1<n ? sequence.vx(j+1) : 0.0f;
    predicted.cwz(0,j)=j+1<n ? sequence.wz(j+1) : 0.0f;
  }
  updateStateVelocities(predicted);
  for(size_t j=0;j+1<n;++j) {
    if(std::abs(predicted.cvx(0,j)-sequence.vx(j+1))>1e-6f||
      std::abs(predicted.cwz(0,j)-sequence.wz(j+1))>1e-6f) {return false;}
  }
  models::Trajectories trajectory;trajectory.reset(1,n);
  integrateStateVelocities(trajectory,predicted);
  const auto footprint=costmap_ros_->getRobotFootprint();
  if(footprint.size()<3||costmap_ros_->getUseRadius()) {return false;}
  for(const auto & point:footprint) {
    if(!finite(point.x)||!finite(point.y)) {return false;}
  }
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto clear=[&](double x,double y,double yaw) {
      if(!finite(x)||!finite(y)||!finite(yaw)) {return false;}
      const double cost=checker.footprintCostAtPose(x,y,yaw,footprint);
      return finite(cost)&&cost>=0&&cost<nav2_costmap_2d::LETHAL_OBSTACLE;
    };
  if(!clear(state_.pose.pose.position.x,state_.pose.pose.position.y,
    tf2::getYaw(state_.pose.pose.orientation))) {return false;}
  for(size_t j=0;j<n;++j) {
    if(!clear(trajectory.x(0,j),trajectory.y(0,j),trajectory.yaws(0,j))) {return false;}
  }
  const double xy=std::min(c.gates.first,gates.first),yaw_gate=std::min(c.gates.second,gates.second);
  size_t chosen=real;
  for(size_t j=0;j<n;++j) {
    const double x=trajectory.x(0,j),y=trajectory.y(0,j),yaw=trajectory.yaws(0,j);
    if(std::hypot(x-c.anchor_x,y-c.anchor_y)+1e-6<c.edge) {continue;}
    const size_t begin=select_target ? 1 : 0,end=select_target ? std::min<size_t>(real,9) : 1;
    for(size_t k=begin;k<end;++k) {
      const double tx=select_target ? path_.x(k) : c.target.position.x;
      const double ty=select_target ? path_.y(k) : c.target.position.y;
      const double ta=select_target ? path_.yaws(k) : tf2::getYaw(c.target.orientation);
      if(std::hypot(x-tx,y-ty)<=xy &&
        std::abs(angles::shortest_angular_distance(yaw,ta))<=yaw_gate)
      {first_hit=j;chosen=k;break;}
    }
    if(first_hit<n) {break;}
  }
  if(first_hit==n) {return false;}
  if(select_target) {
    c.target.position.x=path_.x(chosen);c.target.position.y=path_.y(chosen);
    c.target.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),path_.yaws(chosen)));
    if(c.canonical_plan) {
      if(!pathBindingValid()||chosen>=path_binding_->real_ids.size()) {return false;}
      c.source_id=path_binding_->real_ids[chosen];
      if(!refreshLongEntryBinding()) {return false;}
    }
    // Fixed-target validation checks the exact canonical source identity.
    return validateLongEntrySequence(sequence,first_hit,false);
  }
  return true;
}

bool Optimizer::initializeLongEntryCertificate()
{
  long_entry_required_=true;
  const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if(!model||settings_.time_steps<2||settings_.time_steps>81||!effective_cusp_tolerances_) {return false;}
  const float radius=model->getMinTurningRadius();
  if(!diagnostic::isFiniteCaptureValue(radius)||radius<=0) {return false;}
  LongEntryCertificate certificate;
  certificate.frame=state_.pose.header.frame_id;certificate.gates=cuspTolerances();
  certificate.anchor_x=state_.pose.pose.position.x;certificate.anchor_y=state_.pose.pose.position.y;
  certificate.stamp_ns=rclcpp::Time(state_.pose.header.stamp).nanoseconds();
  certificate.last_stamp_ns=certificate.stamp_ns;
  certificate.epoch=long_entry_epoch_;
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  for(size_t k=1;k<std::min<size_t>(real,9);++k) {
    certificate.edge=std::hypot(path_.x(k)-path_.x(0),path_.y(k)-path_.y(0));
    if(certificate.edge>1e-6) {break;}
  }
  if(path_binding_required_) {
    if(!pathBindingValid()) {return false;}
    certificate.canonical_plan=path_binding_->canonical_plan;
    certificate.plan_generation=path_binding_->generation;
    certificate.segment_generation=path_binding_->segment;
    geometry_msgs::msg::PoseStamped local_anchor,canonical_anchor;
    local_anchor.header=state_.pose.header;local_anchor.pose=state_.pose.pose;
    tf2::Transform transform;tf2::fromMsg(path_binding_->canonical_to_local.transform,transform);
    geometry_msgs::msg::TransformStamped inverse;
    inverse.header.frame_id=path_binding_->canonical_plan->header.frame_id;
    inverse.child_frame_id=state_.pose.header.frame_id;inverse.header.stamp=state_.pose.header.stamp;
    inverse.transform=tf2::toMsg(transform.inverse());
    tf2::doTransform(local_anchor,canonical_anchor,inverse);
    certificate.canonical_anchor=canonical_anchor.pose;
  }
  long_entry_=certificate;
  for(float crawl : {0.003f,0.005f,0.01f,0.02f}) {
    for(float delay : {0.0f,0.5f,1.0f,1.5f}) {
      for(float cruise : {0.02f,0.04f,0.08f,0.12f,0.16f}) {
        for(float fraction : {-1.0f,-0.5f,0.0f,0.5f,1.0f}) {
          models::ControlSequence candidate;candidate.reset(2*settings_.time_steps-1);
          for(size_t j=0;j<candidate.vx.size();++j) {
            candidate.vx(j)=std::min(j*settings_.model_dt<delay ? crawl : cruise,
              settings_.constraints.vx_max);
            candidate.wz(j)=std::clamp(candidate.vx(j)*fraction/radius,
              -settings_.constraints.wz,settings_.constraints.wz);
          }
          size_t hit=0;
          if(validateLongEntrySequence(candidate,hit,true)) {
            long_entry_->controls=candidate;
            return true;
          }
        }
      }
    }
  }
  long_entry_.reset();return false;
}

void Optimizer::enforceLongEntryCertificate(bool allow_optimized)
{
  if(!long_entry_required_) {return;}
  const auto reject=[&]() {
      long_entry_.reset();
      throw nav2_core::FailedToMakeProgress("Long-entry certificate invalid or expired; stopped replan required");
    };
  if(!long_entry_) {reject();}
  auto retained=long_entry_->controls;
  size_t retained_hit=0;
  if(!validateLongEntrySequence(retained,retained_hit)) {reject();}
  auto chosen=retained;
  if(allow_optimized && control_sequence_.vx.size()==settings_.time_steps &&
    control_sequence_.wz.size()==settings_.time_steps)
  {
    auto composed=retained;
    for(size_t j=0;j<settings_.time_steps;++j) {
      composed.vx(j)=control_sequence_.vx(j);composed.wz(j)=control_sequence_.wz(j);
    }
    size_t hit=0;
    if(validateLongEntrySequence(composed,hit)&&hit<=retained_hit) {chosen=composed;}
  }
  long_entry_->controls=chosen;
  long_entry_->last_stamp_ns=rclcpp::Time(state_.pose.header.stamp).nanoseconds();
  for(size_t j=0;j<settings_.time_steps;++j) {
    control_sequence_.vx(j)=chosen.vx(j);control_sequence_.wz(j)=chosen.wz(j);
  }
}

bool Optimizer::initializeShortCuspMean()
{
  // This only initializes the Gaussian mean. It neither emits a command nor
  // consumes a cusp. Normal noisy optimization, filtering and constraints follow.
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if (!seed_short_cusp_on_handoff_ || !model || real == 0 ||
    (real > 4 && !initialize_long_cusp_entry_) ||
    active_cusp_direction_ == 0 || settings_.time_steps < 2 || settings_.batch_size == 0)
  {
    return false;
  }
  const float x = path_.x(real - 1), y = path_.y(real - 1);
  const float yaw = path_.yaws(real - 1);
  const float yaw_error = angles::shortest_angular_distance(
    tf2::getYaw(state_.pose.pose.orientation), yaw);
  const float distance = std::hypot(x - state_.pose.pose.position.x, y - state_.pose.pose.position.y);
  // A sparsely sampled curved entry must not fall between the short-distance
  // gate and the long-entry point-count gate. Geometry, not interpolation
  // density, determines which bounded initializer family can cover it.
  const bool long_entry = initialize_long_cusp_entry_ && (real > 4 || distance > 0.12f);
  // Long entry is explicitly optional and bounded. Do not reseed every tick
  // and erase receding-horizon memory.
  if (long_entry && std::isfinite(distance) && distance > 0.8f) {
    return initializeLongCuspPrefixMean();
  }
  if (distance > (long_entry ? 0.8f : 0.12f) ||
    !std::isfinite(distance) || !std::isfinite(yaw_error)) {return false;}
  const auto gates = cuspTolerances();
  const float xy_gate = gates.first;
  const float yaw_gate = gates.second;
  if (xy_gate <= 0 || yaw_gate <= 0 || !std::isfinite(xy_gate) || !std::isfinite(yaw_gate) ||
    std::abs(yaw_error) <= yaw_gate)
  {
    return false;
  }
  const float radius = model->getMinTurningRadius();
  if (!(radius > 0) || !std::isfinite(radius)) {return false;}
  const float gear = active_cusp_direction_ > 0 ? 1.0f : -1.0f;
  const float yaw_sign = std::copysign(1.0f, yaw_error);
  std::vector<float> budgets = {0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.08f, 0.10f, 0.12f};
  // A fixed centimetre grid can step over the entire feasible stopping set.
  // Add budgets anchored to this actual endpoint and its unchanged XY gate.
  for (float extra : {0.0f, 0.5f * xy_gate, xy_gate}) {
    const float budget = std::clamp(distance + extra, 0.005f, 0.12f);
    if (std::none_of(budgets.begin(), budgets.end(), [budget](float b) {
        return std::abs(b - budget) < 1e-6f;
      }))
    {
      budgets.push_back(budget);
    }
  }
  std::vector<models::ControlSequence> proposals;
  float long_travel = 0.0f;
  if (long_entry) {
    float length = 0.0f;
    for (size_t k = 1; k < real; ++k) {
      length += std::hypot(path_.x(k) - path_.x(k - 1), path_.y(k) - path_.y(k - 1));
    }
    const float turn = std::abs(angles::shortest_angular_distance(path_.yaws(0), yaw));
    if (turn < 0.1f) {return false;}
    const float travel = length / turn * std::abs(yaw_error);
    long_travel = travel;
    budgets.clear();
    for (float extra : {-0.06f, -0.03f, 0.0f, 0.03f, 0.06f}) {
      if (travel + extra > 0.0f) {budgets.push_back(travel + extra);}
    }
  }
  const std::vector<float> crawls = long_entry ?
    std::vector<float>{0.015f, 0.03f} : std::vector<float>{0.001f, 0.005f, 0.015f};
  const std::vector<float> cruises = long_entry ?
    std::vector<float>{0.18f, 0.24f, 0.30f} : std::vector<float>{0.04f, 0.08f, 0.12f};
  for (float delay : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f}) {
    for (float crawl : crawls) {
      if (!long_entry && delay == 0 && crawl != 0.001f) {continue;}
      for (float cruise : cruises) {
        for (float curvature : {1.8f, 2.0f, 2.2f, 1.0f / radius}) {
          if (curvature > 1.0f / radius + 1e-6f) {continue;}
          for (float budget : budgets) {
            models::ControlSequence candidate; candidate.reset(settings_.time_steps);
            float remaining = budget;
            for (unsigned int j = 0; j < settings_.time_steps; ++j) {
              const float requested = j * settings_.model_dt < delay ? crawl : cruise;
              const float travel = std::min({requested, remaining / settings_.model_dt,
                  gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                  settings_.constraints.wz / curvature});
              candidate.vx(j) = gear * std::max(0.0f, travel);
              candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) * curvature;
              remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
            }
            proposals.push_back(std::move(candidate));
          }
        }
      }
    }
  }
  if (long_entry) {
    // Opposite steering carried through a gear switch can require a slower
    // crawl and cruise than the original long-entry family offers. Add only
    // twelve endpoint-anchored candidates; preserve measured steering, gear,
    // normal constraints, footprint checks and the unchanged strict gates.
    for (float delay : {2.0f, 2.5f}) {
      for (float cruise : {0.08f, 0.15f}) {
        for (float extra : {-0.02f, -0.01f, 0.0f}) {
          models::ControlSequence candidate; candidate.reset(settings_.time_steps);
          float remaining = std::max(0.0f, long_travel + extra);
          for (unsigned int j = 0; j < settings_.time_steps; ++j) {
            const float requested = j * settings_.model_dt < delay ? 0.005f : cruise;
            const float travel = std::min({requested, remaining / settings_.model_dt,
                gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                settings_.constraints.wz * radius});
            candidate.vx(j) = gear * std::max(0.0f, travel);
            candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) / radius;
            remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
          }
          proposals.push_back(std::move(candidate));
        }
      }
    }
  }
  float best_cost = std::numeric_limits<float>::infinity();
  size_t best = proposals.size();
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  const auto started = std::chrono::steady_clock::now();
  size_t gentle_first = std::numeric_limits<size_t>::max();
  const auto screen = [&](size_t first) {
    for (size_t start = first; start < proposals.size(); start += settings_.batch_size) {
      const size_t count = std::min(static_cast<size_t>(settings_.batch_size), proposals.size() - start);
      for (size_t row = 0; row < settings_.batch_size; ++row) {
        const auto & candidate = proposals[start + std::min(row, count - 1)];
        for (size_t j = 0; j < settings_.time_steps; ++j) {
          state_.cvx(row, j) = candidate.vx(j);
          state_.cwz(row, j) = candidate.wz(j);
        }
      }
      updateStateVelocities(state_);
      integrateStateVelocities(generated_trajectories_, state_);
      costs_.fill(0); critics_data_.furthest_reached_path_point.reset();
      critics_data_.path_pts_valid.reset(); critics_data_.fail_flag = false;
      // Finite seed screening is not a Gaussian MPPI update. Keep its summary
      // in CUSP_SEED diagnostics, not indistinguishable controller critic stats.
      critic_manager_.evalTrajectoriesScores(critics_data_, false);
      for (size_t row = 0; row < count; ++row) {
        const size_t last = settings_.time_steps - 1;
        const float radial = std::hypot(generated_trajectories_.x(row, last) - x,
            generated_trajectories_.y(row, last) - y);
        const float angular = std::abs(angles::shortest_angular_distance(
            generated_trajectories_.yaws(row, last), yaw));
        if (!std::isfinite(costs_(row)) || costs_(row) >= best_cost ||
          radial > xy_gate || angular > yaw_gate)
        {
          continue;
        }
        bool collision = false;
        for (size_t j = 0; j < settings_.time_steps; ++j) {
          const auto cost = checker.footprintCostAtPose(generated_trajectories_.x(row, j),
              generated_trajectories_.y(row, j), generated_trajectories_.yaws(row, j), footprint);
          if (cost == nav2_costmap_2d::LETHAL_OBSTACLE || cost == nav2_costmap_2d::NO_INFORMATION) {
            collision = true; break;
          }
        }
        if (!collision && (gentle_short_cusp_required_ || start + row >= gentle_first)) {
          // The supplemental family must survive constraints and the actual
          // index-1 emission stream, not just the conventional mean forecast.
          const auto previous = control_sequence_;
          control_sequence_ = proposals[start + row];
          try {applyControlSequenceConstraints();}
          catch (...) {control_sequence_ = previous; throw;}
          const auto constrained = control_sequence_;
          control_sequence_ = previous;
          if (!cuspSequenceFeasible(constrained)) {continue;}
          if (!actualCuspStreamFeasible(constrained)) {continue;}
          proposals[start + row] = constrained;
        }
        if (!collision) {best_cost = costs_(row); best = start + row;}
      }
    }
  };
  screen(0);
  if (!long_entry && best == proposals.size()) {
    // Coarse crawl spacing can miss a steering-settling continuation between
    // its existing speeds. Refine those intervals only after the original
    // family fails, preserving every previously selected seed. These are
    // bounded same-gear proposals, not a relaxed endpoint or collision gate.
    const size_t first = proposals.size();
    for (float delay : {0.5f, 1.0f, 1.5f, 2.0f}) {
      for (size_t i = 1; i < crawls.size(); ++i) {
        const float crawl = 0.5f * (crawls[i - 1] + crawls[i]);
        for (float cruise : cruises) {
          for (float curvature : {1.8f, 2.0f, 2.2f, 1.0f / radius}) {
            if (curvature > 1.0f / radius + 1e-6f) {continue;}
            for (float budget : budgets) {
              models::ControlSequence candidate; candidate.reset(settings_.time_steps);
              float remaining = budget;
              for (unsigned int j = 0; j < settings_.time_steps; ++j) {
                const float requested = j * settings_.model_dt < delay ? crawl : cruise;
                const float travel = std::min({requested, remaining / settings_.model_dt,
                    gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                    settings_.constraints.wz / curvature});
                candidate.vx(j) = gear * std::max(0.0f, travel);
                candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) * curvature;
                remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
              }
              proposals.push_back(std::move(candidate));
            }
          }
        }
      }
    }
    best = proposals.size();
    screen(first);
  }
  if (!long_entry && best == proposals.size()) {
    // The 12 cm entry-domain limit is not a travel-length limit. A valid
    // curved continuation can travel farther while ending inside the same
    // unchanged cusp ball. Search a small endpoint-relative extension only
    // after both original families fail; every forecast still needs the exact
    // endpoint, gear and footprint checks. Keep this distinct from long entry.
    const size_t first = proposals.size();
    // Millimetre travel spacing resolves a narrow feasible stopping interval;
    // cap at 51 budgets even if a future caller provides a wider XY gate.
    for (size_t step = 0; step <= 50; ++step) {
      const float extra = 0.5f * xy_gate + 0.001f * step;
      if (extra > xy_gate + 1e-6f) {break;}
      const float budget = distance + extra;
      if (budget <= 0.12f || budget > 0.17f) {continue;}
      for (float delay : {0.0f, 0.5f, 1.0f}) {
        for (float cruise : cruises) {
          for (float curvature : {1.8f, 2.0f, 2.2f, 1.0f / radius}) {
            if (curvature > 1.0f / radius + 1e-6f) {continue;}
            models::ControlSequence candidate; candidate.reset(settings_.time_steps);
            float remaining = budget;
            for (unsigned int j = 0; j < settings_.time_steps; ++j) {
              const float requested = j * settings_.model_dt < delay ? 0.003f : cruise;
              const float travel = std::min({requested, remaining / settings_.model_dt,
                  gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                  settings_.constraints.wz / curvature});
              candidate.vx(j) = gear * std::max(0.0f, travel);
              candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) * curvature;
              remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
            }
            proposals.push_back(std::move(candidate));
          }
        }
      }
    }
    best = proposals.size();
    screen(first);
  }
  if (!long_entry && best == proposals.size()) {
    // Very slow crawl requests cannot settle opposite steering sufficiently
    // within the finite horizon: the physical predictor regularizes curvature
    // near zero speed. Try a bounded faster crawl only after existing families
    // fail. Preserve measured steering, endpoint gates and full footprint checks.
    const size_t first = proposals.size();
    for (float delay : {0.5f, 1.0f, 1.5f}) {
      for (float crawl : {0.03f, 0.04f, 0.05f}) {
        for (float cruise : {0.08f, 0.12f}) {
          for (size_t step = 0; step <= 5; ++step) {
            const float extra = std::min(0.01f * step, xy_gate);
            const float budget = distance + extra;
            if (budget > 0.17f) {continue;}
            models::ControlSequence candidate; candidate.reset(settings_.time_steps);
            float remaining = budget;
            for (unsigned int j = 0; j < settings_.time_steps; ++j) {
              const float requested = j * settings_.model_dt < delay ? crawl : cruise;
              const float travel = std::min({requested, remaining / settings_.model_dt,
                  gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                  settings_.constraints.wz * radius});
              candidate.vx(j) = gear * std::max(0.0f, travel);
              candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) / radius;
              remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
            }
            proposals.push_back(std::move(candidate));
          }
        }
      }
    }
    best = proposals.size();
    screen(first);
  }
  if (!long_entry && best == proposals.size()) {
    // A tight endpoint ball can contain gentle arcs while excluding every
    // near-maximum-curvature proposal. Preserve existing successful choices;
    // only supplement an exhausted short-cusp search, without changing gates.
    gentle_first = proposals.size();
    auto gentle_budgets = budgets;
    std::sort(gentle_budgets.begin(), gentle_budgets.end());
    const auto original_budgets = gentle_budgets;
    for (size_t k = 1; k < original_budgets.size(); ++k) {
      gentle_budgets.push_back(0.5f * (original_budgets[k - 1] + original_budgets[k]));
    }
    for (float curvature : {0.5f, 1.0f, 1.5f}) {
      if (curvature > 1.0f / radius + 1e-6f) {continue;}
      for (float cruise : {0.03f, 0.05f, 0.08f}) {
        for (float budget : gentle_budgets) {
          models::ControlSequence candidate; candidate.reset(settings_.time_steps);
          float remaining = budget;
          for (size_t j = 0; j < settings_.time_steps; ++j) {
            const float travel = std::min({cruise, remaining / settings_.model_dt,
                gear > 0 ? settings_.constraints.vx_max : -settings_.constraints.vx_min,
                settings_.constraints.wz / curvature});
            candidate.vx(j) = gear * std::max(0.0f, travel);
            candidate.wz(j) = yaw_sign * std::abs(candidate.vx(j)) * curvature;
            remaining = std::max(0.0f, remaining - std::abs(candidate.vx(j)) * settings_.model_dt);
          }
          proposals.push_back(std::move(candidate));
        }
      }
    }
    best = proposals.size();
    screen(gentle_first);
  }
  // Screening must not leave critic cache/failure state in the Gaussian update.
  costs_.fill(0); critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset(); critics_data_.fail_flag = false;
  bool found = best < proposals.size();
  if (found) {
    if (best >= gentle_first) {gentle_short_cusp_required_ = true;}
    const auto previous = control_sequence_;
    control_sequence_ = proposals[best];
    applyControlSequenceConstraints();
    if (long_entry && !shortCuspSequenceFeasible()) {
      control_sequence_ = previous;
      found = false;
    }
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  RCLCPP_INFO(logger_, "[CUSP_SEED] proposals=%zu found=%d cost=%.5f screen_ms=%.3f",
    proposals.size(), found, best_cost, elapsed_ms);
  return found;
}

bool Optimizer::initializeLongCuspPrefixMean()
{
  // A long outgoing path is not an endpoint-completion problem. Seed only
  // the local forward entry, keeping the full path and terminal gates intact.
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  const auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if (!initialize_long_cusp_entry_ || !model || real <= 4 || active_cusp_direction_ != 1) {
    return false;
  }
  const auto gates = cuspTolerances();
  const float radius = model->getMinTurningRadius();
  if (!(gates.first > 0) || !(gates.second > 0) || !(radius > 0)) {return false;}
  const auto finite = [](double value) {
      uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
      return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    };
  if (!finite(gates.first) || !finite(gates.second) || !finite(radius) ||
    !finite(state_.pose.pose.position.x) || !finite(state_.pose.pose.position.y) ||
    !finite(tf2::getYaw(state_.pose.pose.orientation)) || !finite(state_.speed.linear.x) ||
    !finite(state_.speed.angular.z) || !model->initialSteering() ||
    !finite(*model->initialSteering()) ||
    (model->requiresConditionedSteering() && (!model->initialConditionedSteering() ||
    !finite(*model->initialConditionedSteering())))) {return false;}
  for (size_t k = 0; k < real; ++k) {
    if (!finite(path_.x(k)) || !finite(path_.y(k)) || !finite(path_.yaws(k))) {return false;}
  }
  const auto original = control_sequence_;
  std::vector<models::ControlSequence> proposals;
  for (float crawl : {0.003f, 0.005f, 0.01f, 0.02f}) {
    for (float delay : {0.0f, 0.5f, 1.0f, 1.5f}) {
      for (float cruise : {0.02f, 0.04f, 0.08f, 0.12f, 0.16f}) {
        for (float fraction : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
          for (size_t j = 0; j < settings_.time_steps; ++j) {
            const float vx = std::min(j * settings_.model_dt < delay ? crawl : cruise,
              settings_.constraints.vx_max);
            control_sequence_.vx(j) = vx;
            control_sequence_.wz(j) = std::clamp(vx * fraction / radius,
              -settings_.constraints.wz, settings_.constraints.wz);
          }
          applyControlSequenceConstraints();
          proposals.push_back(control_sequence_);
        }
      }
    }
  }
  control_sequence_ = original;
  float edge = 0.0f;
  for (size_t k = 1; k < std::min<size_t>(real, 9); ++k) {
    edge = std::hypot(path_.x(k) - path_.x(0), path_.y(k) - path_.y(0));
    if (edge > 1e-6f) {break;}
  }
  if (!(edge > 1e-6f)) {return false;}
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  size_t best = proposals.size();
  float best_cost = std::numeric_limits<float>::infinity();
  for (size_t start = 0; start < proposals.size(); start += settings_.batch_size) {
    const size_t count = std::min(static_cast<size_t>(settings_.batch_size), proposals.size() - start);
    for (size_t row = 0; row < settings_.batch_size; ++row) {
      const auto & proposal = proposals[start + std::min(row, count - 1)];
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(row,j) = proposal.vx(j); state_.cwz(row,j) = proposal.wz(j);
      }
    }
    updateStateVelocities(state_); integrateStateVelocities(generated_trajectories_,state_);
    costs_.fill(0); critics_data_.furthest_reached_path_point.reset();
    critics_data_.path_pts_valid.reset(); critics_data_.fail_flag = false;
    critic_manager_.evalTrajectoriesScores(critics_data_,false);
    for (size_t row = 0; row < count; ++row) {
      if (!std::isfinite(costs_(row)) || costs_(row) >= best_cost) {continue;}
      const size_t last = settings_.time_steps - 1;
      const float x = generated_trajectories_.x(row,last), y = generated_trajectories_.y(row,last);
      const float yaw = generated_trajectories_.yaws(row,last);
      if (std::hypot(x-state_.pose.pose.position.x,y-state_.pose.pose.position.y) < edge) {continue;}
      bool aligned = false;
      for (size_t k = 1; k < std::min<size_t>(real,9); ++k) {
        aligned = aligned || (std::hypot(x-path_.x(k),y-path_.y(k)) <= gates.first &&
          std::abs(angles::shortest_angular_distance(yaw,path_.yaws(k))) <= gates.second);
      }
      if (!aligned) {continue;}
      bool clear = true;
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        const float px=generated_trajectories_.x(row,j), py=generated_trajectories_.y(row,j);
        const float pa=generated_trajectories_.yaws(row,j);
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pa)) {clear=false;break;}
        const double cost=checker.footprintCostAtPose(px,py,pa,footprint);
        if (cost < 0 || cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {clear=false;break;}
      }
      if (clear) {best=start+row;best_cost=costs_(row);}
    }
  }
  costs_.fill(0); critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset(); critics_data_.fail_flag=false;
  if (best == proposals.size()) {return false;}
  control_sequence_=proposals[best];
  return true;
}

bool Optimizer::controlSequenceCollisionFree() const
{
  models::State predicted;
  predicted.reset(1, settings_.time_steps);
  predicted.pose = state_.pose;
  predicted.speed = state_.speed;
  for (size_t j = 0; j < settings_.time_steps; ++j) {
    if (!std::isfinite(control_sequence_.vx(j)) ||
      !std::isfinite(control_sequence_.wz(j)) ||
      (isHolonomic() && !std::isfinite(control_sequence_.vy(j))))
    {
      return false;
    }
    predicted.cvx(0, j) = control_sequence_.vx(j);
    predicted.cwz(0, j) = control_sequence_.wz(j);
    if (isHolonomic()) {predicted.cvy(0, j) = control_sequence_.vy(j);}
  }
  updateStateVelocities(predicted);
  models::Trajectories trajectory;
  trajectory.reset(1, settings_.time_steps);
  integrateStateVelocities(trajectory, predicted);
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  for (size_t j = 0; j < settings_.time_steps; ++j) {
    if (!std::isfinite(trajectory.x(0, j)) || !std::isfinite(trajectory.y(0, j)) ||
      !std::isfinite(trajectory.yaws(0, j)))
    {
      return false;
    }
    const double cost = checker.footprintCostAtPose(
      trajectory.x(0, j), trajectory.y(0, j), trajectory.yaws(0, j), footprint);
    if (cost < 0 || cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {return false;}
  }
  return true;
}

void Optimizer::preserveFilteredCollisionFeasibility(
  const models::ControlSequence * before_filter)
{
  if (!preserve_filtered_collision_feasibility_ || controlSequenceCollisionFree()) {return;}
  // Filtering and averaging are not collision-preserving operations. Retain
  // the pre-filter mean only after validating it with the same measured state,
  // actuator predictor, final constraints and footprint grid as the output.
  const auto rejected = control_sequence_;
  if (before_filter && before_filter->vx.size() == settings_.time_steps &&
    before_filter->wz.size() == settings_.time_steps &&
    (!isHolonomic() || before_filter->vy.size() == settings_.time_steps))
  {
    control_sequence_ = *before_filter;
    applyControlSequenceConstraints();
    if (controlSequenceCollisionFree()) {
      // Retention must not bypass the independent strict endpoint/gear gate.
      enforceShortCuspMeanFeasibility();
      if (controlSequenceCollisionFree()) {return;}
    }
  }
  control_sequence_ = rejected;
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  if (retain_collision_free_long_path_sample_ && real > 4 &&
    std::hypot(path_.x(real - 1) - state_.pose.pose.position.x,
    path_.y(real - 1) - state_.pose.pose.position.y) > 0.12 &&
    std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_))
  {
    // An average of clear curves need not be clear. On long segments only,
    // validate existing ranked controls before discarding the whole tick.
    // This does not add noise, iterations, retries or another planned gear.
    int gear = 0;
    for (size_t k = 1; k < real; ++k) {
      const float dx = path_.x(k) - path_.x(k - 1);
      const float dy = path_.y(k) - path_.y(k - 1);
      if (std::hypot(dx, dy) > 1e-6f) {
        gear = dx * std::cos(path_.yaws(k - 1)) + dy * std::sin(path_.yaws(k - 1)) >= 0 ?
          1 : -1;
        break;
      }
    }
    const auto finite_cost = [](float value) {
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f800000u) != 0x7f800000u;
      };
    std::vector<size_t> ranked;
    for (size_t b = 0; b < costs_.size(); ++b) {
      if (finite_cost(costs_(b)) && costs_(b) < 1e9f) {ranked.push_back(b);}
    }
    const size_t count = std::min<size_t>(64, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(),
      [&](size_t a, size_t b) {return costs_(a) == costs_(b) ? a < b : costs_(a) < costs_(b);});
    const size_t offset = settings_.shift_control_sequence ? 1 : 0;
    const auto same_gear = [&]() {
        if (!gear || gear * control_sequence_.vx(offset) < 0) {return false;}
        bool future_motion = false;
        for (size_t j = 0; j < settings_.time_steps; ++j) {
          if (!finite_cost(control_sequence_.vx(j)) ||
            !finite_cost(control_sequence_.wz(j)) ||
            (j >= offset && gear * control_sequence_.vx(j) < -1e-6f))
          {
            return false;
          }
          future_motion = future_motion || (j >= offset && gear * control_sequence_.vx(j) > 1e-6f);
        }
        // A sampled steering-settling prefix may legitimately command zero
        // now. Require later travel in the planned gear; reject a zero horizon.
        return future_motion;
      };
    try {
      for (size_t i = 0; gear && i < count; ++i) {
        const size_t b = ranked[i];
        control_sequence_.vx = xt::view(state_.cvx, b, xt::all());
        control_sequence_.wz = xt::view(state_.cwz, b, xt::all());
        if (!same_gear()) {continue;}
        applyControlSequenceConstraints();
        if (!same_gear() || !controlSequenceCollisionFree()) {continue;}
        enforceShortCuspMeanFeasibility();
        if (same_gear() && controlSequenceCollisionFree()) {
          RCLCPP_INFO(logger_, "[FILTER_SAMPLE_GUARD] retained constrained sample %zu", b);
          return;
        }
      }
    } catch (...) {
      control_sequence_ = rejected;
      throw;
    }
    control_sequence_ = rejected;
  }
  // Optional, bounded diagnostic. Capture at most one event per simulated
  // second and 64 lowest regularized-cost samples; never imply a full-batch
  // feasibility proof from this subset. Failures still propagate unchanged.
  const char * capture_path = std::getenv("MPPI_FILTER_REJECTION_SNAPSHOT_PATH");
  static size_t filter_capture_index = 0;
  static double last_filter_capture_stamp = -std::numeric_limits<double>::infinity();
  const double stamp = rclcpp::Time(state_.pose.header.stamp).seconds();
  if (capture_path && *capture_path && filter_capture_index < 2000 &&
    (stamp < last_filter_capture_stamp || stamp - last_filter_capture_stamp >= 1.0))
  {
    last_filter_capture_stamp = stamp;
    const size_t index = filter_capture_index++;
    try {
      std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
      const size_t cells = static_cast<size_t>(costmap_->getSizeInCellsX()) *
        costmap_->getSizeInCellsY();
      if (cells > 1000000 || settings_.time_steps > 1000 || path_.x.size() > 10000) {
        throw std::runtime_error("Filter rejection exceeds bounded capture size");
      }
      std::vector<size_t> order(costs_.size());
      std::iota(order.begin(), order.end(), 0);
      for (size_t row : order) {
        if (!std::isfinite(costs_(row))) {
          throw std::runtime_error("Non-finite regularized cost in filter rejection");
        }
      }
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
          return costs_(a) < costs_(b);
        });
      order.resize(std::min<size_t>(64, order.size()));
      std::ostringstream out;
      out << std::setprecision(17)
          << "{\"scope\":\"Exact rejected filter means; never emitted\""
          << ",\"capture_index\":" << index << ",\"stamp\":" << stamp
          << ",\"frame\":" << std::quoted(state_.pose.header.frame_id)
          << ",\"real_path_point_count\":" << critics_data_.real_path_point_count.value_or(0)
          << ",\"active_cusp_direction\":" << active_cusp_direction_
          << ",\"effective_cusp_tolerances\":" << serializedCuspTolerances()
          << ",\"path_direction\":" << path_direction_ << ",\"model_dt\":" << settings_.model_dt
          << ",\"robot\":[" << state_.pose.pose.position.x << ',' << state_.pose.pose.position.y
          << ',' << tf2::getYaw(state_.pose.pose.orientation)
          << "],\"speed\":[" << state_.speed.linear.x << ',' << state_.speed.angular.z
          << "],\"goal\":[" << goal_.position.x << ',' << goal_.position.y << ','
          << tf2::getYaw(goal_.orientation) << ']';
      const auto field = [&](const char * name, const auto & tensor) {
          out << ",\"" << name << "\":"; snapshotTensor(out, tensor);
        };
      field("filtered_vx", rejected.vx); field("filtered_wz", rejected.wz);
      out << ",\"before_filter_available\":" << (before_filter ? "true" : "false");
      if (before_filter) {
        field("before_filter_vx", before_filter->vx); field("before_filter_wz", before_filter->wz);
      }
      field("path_x", path_.x); field("path_y", path_.y); field("path_yaw", path_.yaws);
      const auto ackermann = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
      const auto feedback = [&](const char * name, std::optional<float> value) {
          out << ",\"" << name << "\":";
          if (value) {out << *value;} else {out << "null";}
        };
      feedback("initial_steering", ackermann ? ackermann->initialSteering() : std::nullopt);
      feedback("initial_conditioned_steering",
        ackermann ? ackermann->initialConditionedSteering() : std::nullopt);
      out << ",\"history\":[";
      for (size_t k = 0; k < control_history_.size(); ++k) {
        if (k) {out << ',';}
        out << '[' << control_history_[k].vx << ',' << control_history_[k].wz << ']';
      }
      out << "],\"batch_size\":" << costs_.size()
          << ",\"sample_cost_kind\":\"regularized costs after weighted update\",\"samples\":[";
      for (size_t k = 0; k < order.size(); ++k) {
        const size_t row = order[k];
        if (k) {out << ',';}
        out << "{\"source_row\":" << row << ",\"regularized_cost\":" << costs_(row);
        field("vx", xt::view(state_.cvx, row, xt::all()));
        field("wz", xt::view(state_.cwz, row, xt::all()));
        out << '}';
      }
      out << "],\"footprint\":[";
      const auto footprint = costmap_ros_->getRobotFootprint();
      for (size_t k = 0; k < footprint.size(); ++k) {
        if (k) {out << ',';}
        out << '[' << footprint[k].x << ',' << footprint[k].y << ']';
      }
      out << "],\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
          << ",\"height\":" << costmap_->getSizeInCellsY()
          << ",\"resolution\":" << costmap_->getResolution()
          << ",\"origin\":[" << costmap_->getOriginX() << ',' << costmap_->getOriginY()
          << "],\"data\":[";
      for (size_t k = 0; k < cells; ++k) {
        if (k) {out << ',';}
        out << static_cast<unsigned int>(costmap_->getCharMap()[k]);
      }
      out << "]}}\n";
      const size_t slot = index == 0 ? 0 : 1 + (index - 1) % 7;
      const std::string path = std::string(capture_path) +
        (slot ? "." + std::to_string(slot) : "");
      std::ofstream file(path, std::ios::out | std::ios::trunc);
      if (!file) {throw std::runtime_error("Cannot open filter rejection snapshot");}
      file << out.str(); file.close();
      if (!file) {throw std::runtime_error("Cannot finish filter rejection snapshot");}
    } catch (const std::exception & error) {
      RCLCPP_WARN(logger_, "Optional filter rejection snapshot failed: %s", error.what());
    }
  }
  throw nav2_core::NoValidControl("Filtered mean and retained mean are not collision-free");
}

std::pair<double, double> Optimizer::cuspTolerances() const
{
  if (active_cusp_direction_ != 0 && effective_cusp_tolerances_) {
    return *effective_cusp_tolerances_;
  }
  const auto node = parent_.lock();
  return {node->get_parameter(name_ + ".inversion_xy_tolerance").as_double(),
    node->get_parameter(name_ + ".inversion_yaw_tolerance").as_double()};
}

std::string Optimizer::serializedCuspTolerances() const
{
  if (!effective_cusp_tolerances_) {return "null";}
  std::ostringstream out;
  out << std::setprecision(17) << '[' << effective_cusp_tolerances_->first << ',' <<
    effective_cusp_tolerances_->second << ']';
  return out.str();
}

bool Optimizer::shortCuspSequenceFeasible() const
{
  return control_sequence_.vx.size() == settings_.time_steps &&
         control_sequence_.wz.size() == settings_.time_steps &&
         cuspSequenceFeasible(control_sequence_);
}

bool Optimizer::actualCuspStreamFeasible(const models::ControlSequence & sequence) const
{
  if (sequence.vx.size() < 2 || sequence.wz.size() != sequence.vx.size()) {return false;}
  auto emitted = sequence;
  if (settings_.shift_control_sequence) {
    for (size_t j = 0; j + 1 < emitted.vx.size(); ++j) {
      emitted.vx(j) = sequence.vx(j + 1);
      emitted.wz(j) = sequence.wz(j + 1);
    }
    emitted.vx(emitted.vx.size() - 1) = 0.0f;
    emitted.wz(emitted.wz.size() - 1) = 0.0f;
  }
  // All shifted entries are future commands, including index0. Only the
  // separately supplied measured velocity may oppose the new gear while braking.
  if (active_cusp_direction_ == 0) {return false;}
  for (size_t j = 0; j < emitted.vx.size(); ++j) {
    if (!diagnostic::isFiniteCaptureValue(emitted.vx(j)) ||
      !diagnostic::isFiniteCaptureValue(emitted.wz(j)) ||
      active_cusp_direction_ * emitted.vx(j) < -1e-6f) {return false;}
  }
  return cuspSequenceFeasible(emitted);
}

void Optimizer::enforceGentleCuspEmission()
{
  if (!gentle_short_cusp_required_) {return;}
  const auto reject=[]() {
      throw nav2_core::FailedToMakeProgress("Gentle cusp emitted stream invalid; stopped replan required");
    };
  const auto finite=diagnostic::isFiniteCaptureValue;
  const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  const size_t real=std::min(critics_data_.real_path_point_count.value_or(0),path_.x.size());
  const auto gates=cuspTolerances();
  if (!real || !active_cusp_direction_ || !model || !model->initialSteering() ||
    !finite(*model->initialSteering()) ||
    (model->requiresConditionedSteering() && (!model->initialConditionedSteering() ||
    !finite(*model->initialConditionedSteering()))) ||
    !finite(gates.first) || !finite(gates.second) || gates.first<=0 || gates.second<=0 ||
    !finite(state_.pose.pose.position.x) || !finite(state_.pose.pose.position.y) ||
    !finite(tf2::getYaw(state_.pose.pose.orientation)) || !finite(state_.speed.linear.x) ||
    !finite(state_.speed.angular.z) || !finite(path_.x(real-1)) || !finite(path_.y(real-1)) ||
    !finite(path_.yaws(real-1))) {reject();}
  if (std::hypot(state_.pose.pose.position.x-path_.x(real-1),
      state_.pose.pose.position.y-path_.y(real-1))<=gates.first &&
    std::abs(angles::shortest_angular_distance(tf2::getYaw(state_.pose.pose.orientation),
      path_.yaws(real-1)))<=gates.second)
  {
    gentle_short_cusp_required_=false;return;
  }
  const auto valid=[&]() {
      auto sequence=cusp_completion_.value_or(control_sequence_);
      if (sequence.vx.size()<control_sequence_.vx.size() ||
        sequence.wz.size()!=sequence.vx.size() ||
        control_sequence_.wz.size()!=control_sequence_.vx.size()) {return false;}
      for (size_t j=0;j<control_sequence_.vx.size();++j) {
        sequence.vx(j)=control_sequence_.vx(j);sequence.wz(j)=control_sequence_.wz(j);
      }
      if (!actualCuspStreamFeasible(sequence)) {return false;}
      if (cusp_completion_) {cusp_completion_=sequence;}
      return true;
    };
  if (valid()) {return;}
  // Keep the obligation active while searching: even a legacy candidate must
  // now pass the emitted-stream check before replacing this continuation.
  if (!initializeShortCuspMean() || !valid()) {reject();}
}

bool Optimizer::cuspSequenceFeasible(const models::ControlSequence & sequence) const
{
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  const size_t steps = sequence.vx.size();
  if (real == 0 || steps < 2 || sequence.wz.size() != steps ||
    !std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_))
  {
    return false;
  }
  models::State predicted;
  predicted.reset(1, steps);
  predicted.pose = state_.pose;
  predicted.speed = state_.speed;
  const size_t command_offset = settings_.shift_control_sequence ? 1 : 0;
  for (size_t j = 0; j < steps; ++j) {
    if (!diagnostic::isFiniteCaptureValue(sequence.vx(j)) || !diagnostic::isFiniteCaptureValue(sequence.wz(j))) {
      return false;
    }
    // A short-cusp fallback must complete the active gear segment, not invent
    // another cusp inside its prediction horizon. Measured initial velocity
    // may still oppose the new gear while braking; it is not a future command.
    if (j >= command_offset && active_cusp_direction_ * sequence.vx(j) < -1e-6f) {
      return false;
    }
    predicted.cvx(0, j) = sequence.vx(j);
    predicted.cwz(0, j) = sequence.wz(j);
  }
  updateStateVelocities(predicted);
  models::Trajectories trajectory;
  trajectory.reset(1, steps);
  integrateStateVelocities(trajectory, predicted);
  const auto gates = cuspTolerances();
  const float xy_gate = gates.first;
  const float yaw_gate = gates.second;
  const size_t last = steps - 1;
  const float radial = std::hypot(trajectory.x(0, last) - path_.x(real - 1),
      trajectory.y(0, last) - path_.y(real - 1));
  const float angular = std::abs(angles::shortest_angular_distance(
      trajectory.yaws(0, last), path_.yaws(real - 1)));
  if (!(xy_gate > 0) || !(yaw_gate > 0) || !diagnostic::isFiniteCaptureValue(radial) ||
    !diagnostic::isFiniteCaptureValue(angular) || radial > xy_gate || angular > yaw_gate)
  {
    return false;
  }
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  for (size_t j = 0; j < steps; ++j) {
    if (!diagnostic::isFiniteCaptureValue(trajectory.x(0, j)) || !diagnostic::isFiniteCaptureValue(trajectory.y(0, j)) ||
      !diagnostic::isFiniteCaptureValue(trajectory.yaws(0, j)))
    {
      return false;
    }
    const auto cost = checker.footprintCostAtPose(
      trajectory.x(0, j), trajectory.y(0, j), trajectory.yaws(0, j), footprint);
    if (cost < 0 || cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
      cost == nav2_costmap_2d::NO_INFORMATION)
    {
      return false;
    }
  }
  return true;
}

bool Optimizer::tryExtendedShortCuspContinuation(bool initialize)
{
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  const auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if (!extend_short_cusp_completion_ || !preserve_short_cusp_feasibility_ ||
    !seed_short_cusp_on_handoff_ || !model || active_cusp_direction_ == 0 ||
    real == 0 || real > 7 || (real > 4 && !initialize_long_cusp_entry_) ||
    !settings_.shift_control_sequence ||
    !(settings_.model_dt > 0.0f) || settings_.time_steps < 2)
  {
    return false;
  }
  const float distance = std::hypot(path_.x(real - 1) - state_.pose.pose.position.x,
      path_.y(real - 1) - state_.pose.pose.position.y);
  if (!std::isfinite(distance) || distance > 0.30f) {return false;}
  const size_t steps = static_cast<size_t>(std::lround(4.05 / settings_.model_dt));
  if (steps <= settings_.time_steps || steps > 201) {return false;}
  const auto original = control_sequence_;
  const auto accept = [&](const models::ControlSequence & candidate) {
      if (candidate.vx.size() != steps || candidate.wz.size() != steps ||
        candidate.vy.size() != steps) {return false;}
      for (size_t j = 1; j < steps; ++j) {
        if (!std::isfinite(candidate.vx(j)) || !std::isfinite(candidate.wz(j)) ||
          active_cusp_direction_ * candidate.vx(j) < -1e-6f) {return false;}
      }
      control_sequence_ = candidate;
      try {
        applyControlSequenceConstraints();
        const bool feasible = cuspSequenceFeasible(control_sequence_);
        if (feasible) {
          cusp_completion_ = control_sequence_;
          control_sequence_ = original;
          for (size_t j = 0; j < settings_.time_steps; ++j) {
            control_sequence_.vx(j) = cusp_completion_->vx(j);
            control_sequence_.wz(j) = cusp_completion_->wz(j);
          }
          return true;
        }
      } catch (...) {
        control_sequence_ = original;
        throw;
      }
      control_sequence_ = original;
      return false;
    };
  if (cusp_completion_) {
    // Recheck current feedback and the entire tail against the current locked
    // grid. Shifting consumes the initial delay; it must never restart here.
    const auto carried = *cusp_completion_;
    if (cusp_completion_shifts_ < steps && accept(carried)) {return true;}
    cusp_completion_.reset();
    return false;
  }
  if (!initialize || cusp_completion_attempted_) {return false;}
  cusp_completion_attempted_ = true;
  const float yaw = angles::shortest_angular_distance(
    tf2::getYaw(state_.pose.pose.orientation), path_.yaws(real - 1));
  const float curve = std::copysign(1.0f / model->getMinTurningRadius(), yaw);
  const float xy_gate = cuspTolerances().first;
  // One finite family per actual handoff/plan. It uses measured steering and
  // the native conditioning model, not an instantaneous steering reset.
  for (float delay : {1.5f, 2.0f, 2.5f, 3.0f}) {
    for (float crawl : {0.001f, 0.003f, 0.005f}) {
      for (float cruise : {0.08f, 0.12f, 0.16f}) {
        for (float extra : {0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f}) {
          if (extra > xy_gate + 1e-6f || distance + extra > 0.35f) {continue;}
          models::ControlSequence candidate;
          candidate.reset(steps);
          float remaining = distance + extra;
          for (size_t j = 0; j < steps; ++j) {
            const float v = std::min(j * settings_.model_dt < delay ? crawl : cruise,
                remaining / settings_.model_dt);
            candidate.vx(j) = active_cusp_direction_ * v;
            candidate.wz(j) = v * curve;
            remaining = std::max(0.0f, remaining - v * settings_.model_dt);
          }
          if (accept(candidate)) {
            RCLCPP_INFO(logger_, "[CUSP_COMPLETION] retained %zu-step completion; emitted horizon %u",
              steps, settings_.time_steps);
            return true;
          }
        }
      }
    }
  }
  return false;
}

void Optimizer::enforceShortCuspMeanFeasibility()
{
  enforceShortCuspMeanFeasibility(nullptr);
}

void Optimizer::enforceShortCuspMeanFeasibility(const models::ControlSequence * warm_start)
{
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  if (!preserve_short_cusp_feasibility_ || !seed_short_cusp_on_handoff_ ||
    active_cusp_direction_ == 0 || real == 0 || (real > 4 && !initialize_long_cusp_entry_) ||
    !std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_))
  {
    return;
  }
  const float distance = std::hypot(path_.x(real - 1) - state_.pose.pose.position.x,
      path_.y(real - 1) - state_.pose.pose.position.y);
  // A short segment can enter with opposing measured steering while its
  // endpoint is still beyond the legacy 12cm seed domain. Waiting until that
  // boundary can consume the turn's remaining heading margin. The optional
  // completion also covers a seven-point, 30cm entry when long-entry handling
  // is enabled. Recorded opposing-steering entries lose feasibility before
  // pruning reaches four points. It validates
  // the entire tail with unchanged gates, gear, feedback and footprint grid.
  if (real <= 7 && (real <= 4 || initialize_long_cusp_entry_) &&
    distance > 0.12f && distance <= 0.30f &&
    extend_short_cusp_completion_ && !shortCuspSequenceFeasible())
  {
    if (tryExtendedShortCuspContinuation(false) ||
      tryExtendedShortCuspContinuation(true)) {return;}
  }
  if (real > 4 || (initialize_long_cusp_entry_ && distance > 0.12f)) {
    // A distant endpoint need not be reachable within the prediction horizon.
    // Retain a proved feasible warm sequence only; otherwise leave normal MPPI
    // untouched. Never restart the timed entry family every control tick.
    if (shortCuspSequenceFeasible() || !warm_start ||
      warm_start->vx.size() != settings_.time_steps ||
      warm_start->wz.size() != settings_.time_steps) {return;}
    const auto optimized = control_sequence_;
    control_sequence_ = *warm_start;
    try {
      applyControlSequenceConstraints();
      if (shortCuspSequenceFeasible()) {
        RCLCPP_INFO(logger_, "[CUSP_ENTRY_GUARD] retained feasible long-entry warm start");
        return;
      }
      if (repair_long_cusp_warm_start_) {
        // Receding-horizon shifting can invalidate a previously feasible entry
        // while measured steering is still catching up. Try bounded temporal
        // repairs of that same warm sequence, never a fresh entry-family search.
        // Every candidate is constrained and rechecked from current feedback.
        const size_t max_delay = std::min<size_t>(4, settings_.time_steps - 1);
        for (size_t delay = 1; delay <= max_delay; ++delay) {
          control_sequence_ = *warm_start;
          for (size_t j = 0; j < settings_.time_steps; ++j) {
            const size_t source = j > delay ? j - delay : 0;
            control_sequence_.vx(j) = warm_start->vx(source);
            control_sequence_.wz(j) = warm_start->wz(source);
          }
          applyControlSequenceConstraints();
          if (shortCuspSequenceFeasible()) {
            RCLCPP_INFO(logger_, "[CUSP_ENTRY_GUARD] repaired warm start by %zu steps", delay);
            return;
          }
        }
      }
    } catch (...) {
      control_sequence_ = optimized;
      throw;
    }
    control_sequence_ = optimized;
    return;
  }
  const float yaw_error = std::abs(angles::shortest_angular_distance(
      tf2::getYaw(state_.pose.pose.orientation), path_.yaws(real - 1)));
  const float yaw_gate = cuspTolerances().second;
  // Same bounded domain as the seed initializer, never a final-goal policy.
  if (!std::isfinite(distance) || distance > 0.12f || !std::isfinite(yaw_error) ||
    yaw_error <= yaw_gate || shortCuspSequenceFeasible())
  {
    cusp_completion_.reset();
    return;
  }
  if (tryExtendedShortCuspContinuation(false)) {return;}
  if (warm_start && warm_start->vx.size() == settings_.time_steps &&
    warm_start->wz.size() == settings_.time_steps)
  {
    const auto rejected_mean = control_sequence_;
    control_sequence_ = *warm_start;
    try {
      applyControlSequenceConstraints();
      if (shortCuspSequenceFeasible()) {
        RCLCPP_INFO(logger_, "[CUSP_MEAN_GUARD] retained feasible shifted warm start");
        return;
      }
    } catch (...) {
      control_sequence_ = rejected_mean;
      throw;
    }
    control_sequence_ = rejected_mean;
  }
  // Preserve the optimizer's actual Gaussian batch and critic telemetry. The
  // fallback search uses scratch storage, not replacement diagnostic rollouts.
  const auto saved_state = state_;
  const auto saved_trajectories = generated_trajectories_;
  const auto saved_costs = costs_;
  const auto saved_furthest = critics_data_.furthest_reached_path_point;
  const auto saved_valid = critics_data_.path_pts_valid;
  const bool saved_fail = critics_data_.fail_flag;
  const auto original_mean = control_sequence_;
  const auto restore = [&]() {
      state_ = saved_state;
      generated_trajectories_ = saved_trajectories;
      costs_ = saved_costs;
      critics_data_.furthest_reached_path_point = saved_furthest;
      critics_data_.path_pts_valid = saved_valid;
      critics_data_.fail_flag = saved_fail;
    };
  bool found = false;
  try {
    found = initializeShortCuspMean();
    applyControlSequenceConstraints();
    found = found && shortCuspSequenceFeasible();
  } catch (...) {
    control_sequence_ = original_mean;
    restore();
    throw;
  }
  restore();
  if (!found) {
    control_sequence_ = original_mean;
    if (tryExtendedShortCuspContinuation(true)) {return;}
    // Bounded failure specimens: these means were rejected, never emitted. Capture
    // after restoring scratch state, before throwing, without changing controls.
    static size_t rejection_capture_attempts = 0;
    const char * rejection_path = std::getenv("MPPI_CUSP_REJECTION_SNAPSHOT_PATH");
    if (rejection_path && *rejection_path) {
      const size_t capture_index = rejection_capture_attempts++;
      try {
        std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
        const size_t cells = static_cast<size_t>(costmap_->getSizeInCellsX()) *
          costmap_->getSizeInCellsY();
        if (cells > 1000000 || path_.x.size() > 10000 || settings_.time_steps > 10000) {
          throw std::runtime_error("Rejected cusp snapshot exceeds bounded capture size");
        }
        std::ostringstream out;
        out << std::setprecision(17)
            << "{\"scope\":\"Exact rejected short-cusp mean; never emitted\""
            << ",\"capture_index\":" << capture_index
            << ",\"frame\":" << std::quoted(state_.pose.header.frame_id)
            << ",\"stamp\":" << rclcpp::Time(state_.pose.header.stamp).seconds()
            << ",\"real_path_point_count\":" << real
            << ",\"active_cusp_direction\":" << active_cusp_direction_
          << ",\"effective_cusp_tolerances\":" << serializedCuspTolerances()
            << ",\"path_direction\":" << path_direction_
            << ",\"model_dt\":" << settings_.model_dt
            << ",\"command_offset\":" << (settings_.shift_control_sequence ? 1 : 0)
            << ",\"robot\":[" << state_.pose.pose.position.x << ','
            << state_.pose.pose.position.y << ',' << tf2::getYaw(state_.pose.pose.orientation)
            << "],\"speed\":[" << state_.speed.linear.x << ',' << state_.speed.angular.z
            << "],\"goal\":[" << goal_.position.x << ',' << goal_.position.y << ','
            << tf2::getYaw(goal_.orientation) << ']';
        const auto ackermann = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
        const auto feedback = [&](const char * name, const std::optional<float> & value) {
            out << ",\"" << name << "\":";
            if (value) {out << *value;} else {out << "null";}
          };
        feedback("initial_steering", ackermann->initialSteering());
        feedback("initial_conditioned_steering", ackermann->initialConditionedSteering());
        const auto field = [&](const char * name, const auto & tensor) {
            out << ",\"" << name << "\":"; snapshotTensor(out, tensor);
          };
        field("control_vx", control_sequence_.vx); field("control_wz", control_sequence_.wz);
        field("path_x", path_.x); field("path_y", path_.y); field("path_yaw", path_.yaws);
        out << ",\"completion_attempted\":" << (cusp_completion_attempted_ ? "true" : "false")
            << ",\"completion_available\":" << (cusp_completion_ ? "true" : "false")
            << ",\"completion_shifts\":" << cusp_completion_shifts_;
        if (cusp_completion_) {
          field("completion_vx", cusp_completion_->vx);
          field("completion_wz", cusp_completion_->wz);
        }
        out << ",\"warm_start_available\":" << (warm_start ? "true" : "false");
        if (warm_start) {
          field("warm_start_vx", warm_start->vx); field("warm_start_wz", warm_start->wz);
        }
        out << ",\"history\":[";
        for (size_t k = 0; k < control_history_.size(); ++k) {
          if (k) {out << ',';}
          out << '[' << control_history_[k].vx << ',' << control_history_[k].wz << ']';
        }
        out << "],\"footprint\":[";
        const auto footprint = costmap_ros_->getRobotFootprint();
        for (size_t k = 0; k < footprint.size(); ++k) {
          if (k) {out << ',';}
          out << '[' << footprint[k].x << ',' << footprint[k].y << ']';
        }
        out << "],\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
            << ",\"height\":" << costmap_->getSizeInCellsY()
            << ",\"resolution\":" << costmap_->getResolution()
            << ",\"origin\":[" << costmap_->getOriginX() << ',' << costmap_->getOriginY()
            << "],\"data\":[";
        for (size_t k = 0; k < cells; ++k) {
          if (k) {out << ',';}
          out << static_cast<unsigned int>(costmap_->getCharMap()[k]);
        }
        out << "]}}\n";
        // Keep the first specimen plus the seven most recent failures. Repeated
        // invalid ticks must not consume capture capacity before a later cusp.
        const size_t file_index = capture_index == 0 ? 0 : 1 + (capture_index - 1) % 7;
        const std::string capture_file = file_index == 0 ? std::string(rejection_path) :
          std::string(rejection_path) + "." + std::to_string(file_index);
        std::ofstream file(capture_file, std::ios::out | std::ios::trunc);
        if (!file) {throw std::runtime_error("Cannot open rejected cusp snapshot");}
        file << out.str(); file.close();
        if (!file) {throw std::runtime_error("Cannot finish rejected cusp snapshot");}
      } catch (const std::exception & error) {
        RCLCPP_WARN(logger_, "Optional rejected cusp snapshot failed: %s", error.what());
      }
    }
    // This finite fallback has already been exhausted from current feedback.
    // NoValidControl is retried by ControllerServer's patience loop, keeping
    // the same unusable segment active. Report inability to progress instead:
    // the server publishes zero and the existing stopped-replan BT can run.
    // This is not a claim of global unreachability or permission to consume
    // the cusp, change gear, or emit the rejected sequence.
    throw nav2_core::FailedToMakeProgress(
            "No validated short-cusp continuation; stopped replan required");
  }
  RCLCPP_INFO(logger_, "[CUSP_MEAN_GUARD] replaced infeasible filtered mean with validated arc");
}

void Optimizer::shiftControlSequence()
{
  using namespace xt::placeholders;  // NOLINT
  if (long_entry_) {
    auto & certificate=*long_entry_;
    certificate.controls.vx=xt::roll(certificate.controls.vx,-1);
    certificate.controls.wz=xt::roll(certificate.controls.wz,-1);
    certificate.controls.vx(certificate.controls.vx.size()-1)=0.0f;
    certificate.controls.wz(certificate.controls.wz.size()-1)=0.0f;
    ++certificate.shifts;
  }
  if (cusp_completion_) {
    cusp_completion_->vx = xt::roll(cusp_completion_->vx, -1);
    cusp_completion_->wz = xt::roll(cusp_completion_->wz, -1);
    cusp_completion_->vx(cusp_completion_->vx.size() - 1) = 0.0f;
    cusp_completion_->wz(cusp_completion_->wz.size() - 1) = 0.0f;
    ++cusp_completion_shifts_;
  }
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

bool Optimizer::longCuspDepartureAvailable(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
  const nav_msgs::msg::Path & outgoing, double xy_gate, double yaw_gate,
  bool terminal_endpoint) const
{
  const auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  const auto finite = [](double value) {
      uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
      return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    };
  if (!finite(xy_gate) || !finite(yaw_gate) || !finite(speed.linear.x) ||
    !finite(speed.angular.z) || !finite(pose.pose.position.x) ||
    !finite(pose.pose.position.y) || !finite(tf2::getYaw(pose.pose.orientation)))
  {
    throw nav2_core::NoValidControl("Non-finite departure state or gate");
  }
  for (const auto & target : outgoing.poses) {
    if (!finite(target.pose.position.x) || !finite(target.pose.position.y) ||
      !finite(tf2::getYaw(target.pose.orientation)))
    {
      throw nav2_core::NoValidControl("Non-finite outgoing departure prefix");
    }
  }
  if (!model || !model->predictsSteeringConditioning() || !model->initialSteering() ||
    (model->requiresConditionedSteering() && !model->initialConditionedSteering()) ||
    outgoing.poses.size() < 2 ||
    outgoing.header.frame_id != pose.header.frame_id || xy_gate <= 0 || yaw_gate <= 0)
  {
    throw nav2_core::NoValidControl("Departure validation requires measured Ackermann state and a local prefix");
  }
  // This finite forward family is a policy test, not a global reachability proof.
  // It intentionally does not call prepare/reset or alter the carried control mean.
  double edge = 0.0;
  for (size_t k = 1; k < outgoing.poses.size(); ++k) {
    edge = std::hypot(outgoing.poses[k].pose.position.x - outgoing.poses[0].pose.position.x,
      outgoing.poses[k].pose.position.y - outgoing.poses[0].pose.position.y);
    if (edge > 1e-6) {break;}
  }
  if (edge <= 1e-6) {throw nav2_core::NoValidControl("Departure prefix has no genuine edge");}
  if (settings_.time_steps < 2 || settings_.time_steps > 81 ||
    !(settings_.constraints.vx_max > 0) || !(settings_.model_dt > 0))
  {
    throw nav2_core::NoValidControl("Invalid departure prediction settings");
  }
  const size_t steps = 2 * settings_.time_steps - 1;
  if (steps < 2 || steps > 161) {
    throw nav2_core::NoValidControl("Departure prediction exceeds bounded horizon");
  }
  const float radius = model->getMinTurningRadius();
  if (radius <= 0) {throw nav2_core::NoValidControl("Invalid departure turning radius");}
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  const double initial_cost = checker.footprintCostAtPose(pose.pose.position.x,
    pose.pose.position.y, tf2::getYaw(pose.pose.orientation), footprint);
  // Diagnostic only: freeze the arguments and locked raster at the rejection,
  // rather than borrowing a preceding optimizer mean or asynchronous ROS grid.
  // Eight recent slots, at most one attempted capture per simulated second.
  const auto capture_rejection = [&](const char * reason) {
      const char * explicit_path = std::getenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH");
      const char * mean_path = std::getenv("MPPI_CUSP_MEAN_TRACE_PATH");
      if ((!explicit_path || !*explicit_path) && (!mean_path || !*mean_path)) {return;}
      static std::mutex capture_mutex;
      static diagnostic::FailedBatchRing captures;
      std::lock_guard<std::mutex> lock(capture_mutex);
      try {
        const double stamp = rclcpp::Time(pose.header.stamp).seconds();
        if (!captures.eligible(true, stamp)) {return;}
        const unsigned int slot = captures.take(stamp);
        const size_t cells = static_cast<size_t>(costmap_->getSizeInCellsX()) * costmap_->getSizeInCellsY();
        if (cells > 250000 || outgoing.poses.size() > 64) {
          throw std::runtime_error("Departure diagnostic exceeds capture bounds");
        }
        std::ostringstream out;
        out << std::setprecision(17) << "{\"scope\":\"Actual departure rejection inputs under controller costmap lock; no command emitted\""
            << ",\"stamp_ns\":" << rclcpp::Time(pose.header.stamp).nanoseconds()
            << ",\"frame\":" << std::quoted(pose.header.frame_id)
            << ",\"reason\":" << std::quoted(reason)
            << ",\"terminal_endpoint\":" << (terminal_endpoint ? "true" : "false")
            << ",\"robot\":[" << pose.pose.position.x << ',' << pose.pose.position.y << ',' << tf2::getYaw(pose.pose.orientation)
            << "],\"speed\":[" << speed.linear.x << ',' << speed.angular.z
            << "],\"initial_steering\":" << *model->initialSteering()
            << ",\"initial_conditioned_steering\":";
        if (model->initialConditionedSteering()) {out << *model->initialConditionedSteering();}
        else {out << "null";}
        out << ",\"initial_footprint_cost\":" << initial_cost
            << ",\"departure_gates\":[" << xy_gate << ',' << yaw_gate
            << "],\"time_steps\":" << settings_.time_steps << ",\"model_dt\":" << settings_.model_dt
            << ",\"constraints\":[" << settings_.constraints.vx_max << ',' << settings_.constraints.vx_min
            << ',' << settings_.constraints.vy << ',' << settings_.constraints.wz << ',' << settings_.constraints.ax_max
            << ',' << settings_.constraints.ax_min << ',' << settings_.constraints.ay_max << ',' << settings_.constraints.ay_min
            << ',' << settings_.constraints.az_max << "],\"departure_prefix\":[";
        for (size_t k = 0; k < outgoing.poses.size(); ++k) {
          if (k) {out << ',';}
          const auto & target = outgoing.poses[k].pose;
          out << '[' << target.position.x << ',' << target.position.y << ',' << tf2::getYaw(target.orientation) << ']';
        }
        out << "],\"footprint\":[";
        for (size_t k = 0; k < footprint.size(); ++k) {
          if (k) {out << ',';} out << '[' << footprint[k].x << ',' << footprint[k].y << ']';
        }
        out << "],\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
            << ",\"height\":" << costmap_->getSizeInCellsY() << ",\"resolution\":" << costmap_->getResolution()
            << ",\"origin\":[" << costmap_->getOriginX() << ',' << costmap_->getOriginY() << "],\"data\":[";
        for (size_t k = 0; k < cells; ++k) {
          if (k) {out << ',';} out << static_cast<unsigned int>(costmap_->getCharMap()[k]);
        }
        out << "]}}\n";
        const std::string destination = explicit_path && *explicit_path ? explicit_path :
          std::string(mean_path) + ".departure_rejection";
        std::ofstream file(destination + "." + std::to_string(slot) + ".json", std::ios::out | std::ios::trunc);
        if (!file) {throw std::runtime_error("Cannot open departure rejection trace");}
        file << out.str(); file.close();
        if (!file) {throw std::runtime_error("Cannot finish departure rejection trace");}

      } catch (const std::exception & error) {
        RCLCPP_WARN_THROTTLE(logger_, *parent_.lock()->get_clock(), 5000,
          "Optional departure rejection capture failed: %s", error.what());
      }
    };
  if (initial_cost < 0 || initial_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
    capture_rejection("initial_footprint"); return false;
  }
  for (float crawl : {0.003f, 0.005f, 0.01f, 0.02f}) {
    for (float delay : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f}) {
      if (!terminal_endpoint && delay > 1.5f) {continue;}
      for (float cruise : {0.02f, 0.04f, 0.08f, 0.12f, 0.16f}) {
        for (float fraction : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f}) {
          if (terminal_endpoint) {
            const double yaw_error = angles::shortest_angular_distance(
              tf2::getYaw(pose.pose.orientation), tf2::getYaw(outgoing.poses.back().pose.orientation));
            const float turn = yaw_error > 1e-6 ? 1.0f : (yaw_error < -1e-6 ? -1.0f : 0.0f);
            // Same bounded curvature family as short-cusp completion. This is
            // conservative failure, never a proof that other curvatures cannot work.
            if (fraction != turn) {continue;}
          }
          for (float extra : {0.0f, 0.01f, 0.02f}) {
          if (!terminal_endpoint && extra > 0.0f) {continue;}
          const auto & endpoint = outgoing.poses.back().pose.position;
          float remaining = std::hypot(endpoint.x - pose.pose.position.x,
            endpoint.y - pose.pose.position.y) + extra;
          models::State candidate;
          candidate.reset(1, steps);
          candidate.pose = pose;
          candidate.speed = speed;
          float last_wz = 0.0f;
          for (size_t j = 0; j < steps; ++j) {
            float vx = std::min(j * settings_.model_dt < delay ? crawl : cruise,
              settings_.constraints.vx_max);
            if (terminal_endpoint) {
              vx = std::min(vx, remaining / settings_.model_dt);
              remaining = std::max(0.0f, remaining - vx * settings_.model_dt);
            }
            float wz = std::clamp(vx * fraction / radius,
              -settings_.constraints.wz, settings_.constraints.wz);
            if (j > 0) {
              const float delta = settings_.constraints.az_max * settings_.model_dt;
              wz = std::clamp(wz, last_wz - delta, last_wz + delta);
            }
            candidate.cvx(0, j) = vx;
            candidate.cwz(0, j) = wz;
            last_wz = wz;
          }
          updateStateVelocities(candidate);
          models::Trajectories trajectory;
          trajectory.reset(1, steps);
          integrateStateVelocities(trajectory, candidate);
          bool aligned = false;
          bool clear = true;
          for (size_t j = 0; j < steps; ++j) {
            const float x = trajectory.x(0, j), y = trajectory.y(0, j);
            const float yaw = trajectory.yaws(0, j);
            const double cost = checker.footprintCostAtPose(x, y, yaw, footprint);
            if (cost < 0 || cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
              clear = false;
              break;
            }
            if (std::hypot(x - pose.pose.position.x, y - pose.pose.position.y) + 1e-6 < edge) {
              continue;
            }
            const size_t begin = terminal_endpoint ? outgoing.poses.size() - 1 : 1;
            if (terminal_endpoint && j + 1 != steps) {continue;}
            for (size_t k = begin; k < (terminal_endpoint ? outgoing.poses.size() :
              std::min<size_t>(outgoing.poses.size(), 9)); ++k) {
              const auto & target = outgoing.poses[k].pose;
              if (std::hypot(x - target.position.x, y - target.position.y) <= xy_gate &&
                std::abs(angles::shortest_angular_distance(yaw,
                tf2::getYaw(target.orientation))) <= yaw_gate)
              {
                aligned = true;
              }
            }
          }
          if (clear && aligned) {return true;}
          }
        }
      }
    }
  }
  capture_rejection("candidate_family_exhausted");
  return false;
}

bool Optimizer::retainedDepartureCommandSafe(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
  const geometry_msgs::msg::Twist & command,
  const geometry_msgs::msg::PoseStamped & cusp, double stop_margin) const
{
  const auto finite = diagnostic::isFiniteCaptureValue;
  const auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  const double yaw = tf2::getYaw(pose.pose.orientation);
  const double cusp_yaw = tf2::getYaw(cusp.pose.orientation);
  if (!costmap_ || !costmap_ros_ || !model || !model->predictsSteeringConditioning() ||
    !model->initialSteering() || !finite(*model->initialSteering()) ||
    (model->requiresConditionedSteering() &&
    (!model->initialConditionedSteering() || !finite(*model->initialConditionedSteering()))) ||
    pose.header.frame_id.empty() || pose.header.frame_id != cusp.header.frame_id ||
    pose.header.frame_id != costmap_ros_->getGlobalFrameID() ||
    !finite(pose.pose.position.x) || !finite(pose.pose.position.y) || !finite(yaw) ||
    !finite(cusp.pose.position.x) || !finite(cusp.pose.position.y) || !finite(cusp_yaw) ||
    !finite(speed.linear.x) || !finite(speed.angular.z) ||
    !finite(command.linear.x) || !finite(command.angular.z) ||
    !finite(stop_margin) || stop_margin <= 0.0 ||
    !finite(settings_.model_dt) || settings_.model_dt <= 0.0f || settings_.model_dt > 0.15f ||
    !finite(settings_.constraints.ax_max) || !finite(settings_.constraints.ax_min) ||
    settings_.constraints.ax_max <= 0.0f || settings_.constraints.ax_min >= 0.0f ||
    speed.linear.x > 0.0 || command.linear.x > 0.0)
  {
    return false;
  }
  const auto remaining = [&](double x, double y) {
      return (x - cusp.pose.position.x) * std::cos(cusp_yaw) +
             (y - cusp.pose.position.y) * std::sin(cusp_yaw);
    };
  const double initial_remaining = remaining(pose.pose.position.x, pose.pose.position.y);
  const double max_speed = std::max(std::abs(speed.linear.x), std::abs(command.linear.x));
  const double deceleration = std::min(settings_.constraints.ax_max, -settings_.constraints.ax_min);
  const double reserve = max_speed * max_speed / (2.0 * deceleration);
  if (!finite(initial_remaining) || initial_remaining <= stop_margin + reserve) {return false;}

  // Hold the extracted command for three native model steps, including the
  // measured initial velocity. This bounds the next command interval rather
  // than demanding the entire incoming mean stop before an unconsumed cusp.
  // The stopping reserve uses configured model acceleration, not a claim of
  // independently measured physical stopping distance.
  models::State predicted;
  predicted.reset(1, 3);
  predicted.pose = pose;
  predicted.speed = speed;
  for (size_t j = 0; j < 3; ++j) {
    predicted.cvx(0, j) = static_cast<float>(command.linear.x);
    predicted.cwz(0, j) = static_cast<float>(command.angular.z);
  }
  updateStateVelocities(predicted);
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const auto footprint = costmap_ros_->getRobotFootprint();
  if (footprint.size() < 3 || costmap_ros_->getUseRadius()) {return false;}
  double x = pose.pose.position.x, y = pose.pose.position.y, heading = yaw;
  const auto clear = [&](double px, double py, double angle) {
      const double cost = checker.footprintCostAtPose(px, py, angle, footprint);
      return finite(cost) && cost >= 0.0 && cost < nav2_costmap_2d::LETHAL_OBSTACLE;
    };
  if (!clear(x, y, heading)) {return false;}
  for (size_t j = 0; j < 3; ++j) {
    const float vx = predicted.vx(0, j), wz = predicted.wz(0, j);
    if (!finite(vx) || !finite(wz) || vx > 0.0f) {return false;}
    x += vx * std::cos(heading) * settings_.model_dt;
    y += vx * std::sin(heading) * settings_.model_dt;
    heading += wz * settings_.model_dt;
    if (remaining(x, y) <= stop_margin + reserve || !clear(x, y, heading)) {return false;}
  }
  return remaining(x, y) < initial_remaining - 1e-6;
}

int Optimizer::shortCuspControlDirection() const
{
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  const float measured = static_cast<float>(state_.speed.linear.x);
  const bool short_cusp = preserve_short_cusp_feasibility_ &&
    seed_short_cusp_on_handoff_ && real <= 4;
  // A direction-conditioned active segment cannot execute a later reversal
  // without a handoff. Keep the predicted requests in the same gear as the
  // imminent command, including long prefixes before the terminal cusp guard.
  if ((!short_cusp && !condition_control_on_path_direction_) ||
    (active_cusp_direction_ != 1 && active_cusp_direction_ != -1) ||
    real == 0 || !std::isfinite(measured) ||
    (active_cusp_direction_ * measured < 0.0f && !condition_control_on_path_direction_) ||
    !std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_))
  {
    return 0;
  }
  return active_cusp_direction_;
}

void Optimizer::constrainShortCuspSampleDirections()
{
  const int gear = shortCuspControlDirection();
  if (gear > 0) {
    state_.cvx = xt::maximum(state_.cvx, 0.0f);
  } else if (gear < 0) {
    state_.cvx = xt::minimum(state_.cvx, 0.0f);
  }
}

void Optimizer::generateNoisedTrajectories()
{
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  // Bound requested controls before predicting physical acceleration and steering.
  // Measured feedback is never projected. With path-direction conditioning,
  // old-gear motion still brakes through the physical predictor; its sign must
  // not reopen opposite-gear requests for the rest of the horizon.
  constrainShortCuspSampleDirections();
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
}

void Optimizer::applyControlSequenceConstraints()
{
  auto & s = settings_;

  if (isHolonomic()) {
    control_sequence_.vy = xt::clip(control_sequence_.vy, -s.constraints.vy, s.constraints.vy);
  }

  const int cusp_gear = shortCuspControlDirection();
  const float vx_min = cusp_gear > 0 ? std::max(0.0f, s.constraints.vx_min) : s.constraints.vx_min;
  const float vx_max = cusp_gear < 0 ? std::min(0.0f, s.constraints.vx_max) : s.constraints.vx_max;
  control_sequence_.vx = xt::clip(control_sequence_.vx, vx_min, vx_max);
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

  // Display the same calibrated response used when scoring sampled rollouts.
  // Output commands and their physical Ackermann limits remain unchanged.
  auto ackermann = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
  if (ackermann) {
    ackermann->applyResponse(
      xt::view(sequence, xt::all(), 0), xt::view(sequence, xt::all(), 1));
  }

  if (is_holo) {
    xt::noalias(xt::view(sequence, xt::all(), 2)) = control_sequence_.vy;
  }

  integrateStateVelocities(trajectories, sequence);
  // Optional, one-shot exact reduction of a colliding final sequence. The
  // controller calls this while holding the same costmap mutex as evalControl.
  // No command or optimizer state is changed; writing may delay this tick.
  static const char * final_capture = std::getenv("MPPI_FINAL_COLLISION_SNAPSHOT_PATH");
  static bool final_capture_attempted = false;
  if (final_capture && *final_capture && !final_capture_attempted && costmap_ros_) {
    const auto footprint = costmap_ros_->getRobotFootprint();
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
    size_t collision_index = trajectories.shape(0);
    for (size_t i = 0; i < trajectories.shape(0); ++i) {
      const double cost = checker.footprintCostAtPose(
        trajectories(i, 0), trajectories(i, 1), trajectories(i, 2), footprint);
      if (cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
        cost == nav2_costmap_2d::NO_INFORMATION)
      {
        collision_index = i; break;
      }
    }
    if (collision_index < trajectories.shape(0)) {
      final_capture_attempted = true;
      try {
        std::ostringstream out;
        out << std::setprecision(17)
            << "{\"scope\":\"Final predicted trajectory and exact locked internal grid; diagnostic write may delay this tick; no command intervention\","
            << "\"stamp_ns\":" << rclcpp::Time(state_.pose.header.stamp).nanoseconds()
            << ",\"collision_index\":" << collision_index
            << ",\"robot\":[" << state_.pose.pose.position.x << ','
            << state_.pose.pose.position.y << ',' << tf2::getYaw(state_.pose.pose.orientation)
            << "],\"footprint\":[";
        for (size_t i = 0; i < footprint.size(); ++i) {
          if (i) {out << ',';}
          out << '[' << footprint[i].x << ',' << footprint[i].y << ']';
        }
        out << "],\"final_trajectory\":"; snapshotTensor(out, trajectories);
        out << ",\"final_sequence\":"; snapshotTensor(out, sequence);
        out << ",\"last_batch_x\":"; snapshotTensor(out, generated_trajectories_.x);
        out << ",\"last_batch_y\":"; snapshotTensor(out, generated_trajectories_.y);
        out << ",\"last_batch_yaw\":"; snapshotTensor(out, generated_trajectories_.yaws);
        out << ",\"last_costs\":"; snapshotTensor(out, costs_);
        out << ",\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
            << ",\"height\":" << costmap_->getSizeInCellsY()
            << ",\"resolution\":" << costmap_->getResolution()
            << ",\"origin\":[" << costmap_->getOriginX() << ','
            << costmap_->getOriginY() << "],\"data\":[";
        const size_t count = costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY();
        for (size_t i = 0; i < count; ++i) {
          if (i) {out << ',';}
          out << static_cast<unsigned int>(costmap_->getCharMap()[i]);
        }
        out << "]}}\n";
        std::ofstream file(final_capture);
        if (!file) {throw std::runtime_error("Cannot open final collision capture");}
        file << out.str(); file.close();
        if (!file) {throw std::runtime_error("Cannot finish final collision capture");}
        RCLCPP_INFO(logger_, "Exact final collision snapshot written to %s", final_capture);
      } catch (const std::exception & e) {
        RCLCPP_WARN(logger_, "Optional final collision snapshot failed: %s", e.what());
      }
    }
  }
  return std::move(trajectories);
}

bool Optimizer::conditionControlExponents(xt::xtensor<float, 1> & exponents)
{
  if (!condition_control_on_path_direction_) {return false;}
  // Never infer gear from preview ghosts. A pruned active cusp retains its
  // original segment gear; a final singleton has no such constraint.
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  int gear = 0;
  for (size_t k = 1; k < real; ++k) {
    const float dx = path_.x(k) - path_.x(k - 1);
    const float dy = path_.y(k) - path_.y(k - 1);
    if (std::hypot(dx, dy) > 1e-6f) {
      gear = dx * std::cos(path_.yaws(k - 1)) + dy * std::sin(path_.yaws(k - 1)) >= 0 ? 1 : -1;
      break;
    }
  }
  if (!gear && real > 0) {gear = active_cusp_direction_;}
  if (!gear && real == 1 && condition_terminal_path_direction_ &&
    (path_direction_ == 1 || path_direction_ == -1))
  {
    // Pruning removes the last real edge, but not the segment's approach gear.
    // Preserve that gear only while the endpoint remains in its travel half-
    // plane. Release after longitudinal overshoot (or within 5 mm of that
    // plane), so this optional policy cannot forbid a settling correction.
    const float yaw = tf2::getYaw(state_.pose.pose.orientation);
    const float along = (path_.x(0) - state_.pose.pose.position.x) * std::cos(yaw) +
      (path_.y(0) - state_.pose.pose.position.y) * std::sin(yaw);
    if (path_direction_ * along > 0.005f) {gear = path_direction_;}
  }
  if (!gear) {return false;}
  const size_t index = settings_.shift_control_sequence ? 1 : 0;
  float minimum = std::numeric_limits<float>::max();
  size_t eligible = 0;
  const auto finite_cost = [](float value) {
      uint32_t bits;std::memcpy(&bits, &value, sizeof(bits));
      return (bits & 0x7f800000u) != 0x7f800000u;
    };
  // Check the same linear SG stencil that will form the emitted command.
  // Conditioning one index alone permits neighboring opposite-gear controls
  // to cancel progress during smoothing.
  const auto filtered_vx = [&](size_t b) {
      if (!smooth_control_sequence_ || state_.cvx.shape(1) < 21) {
        return state_.cvx(b, index);
      }
      constexpr float coefficients[9] = {-21, 14, 39, 54, 59, 54, 39, 14, -21};
      float value = 0.0f;
      for (int j = 0; j < 9; ++j) {
        const int sample = static_cast<int>(index) + j - 4;
        const float vx = sample < 0 ? control_history_[sample + 4].vx : state_.cvx(b, sample);
        value += coefficients[j] / 231.0f * vx;
      }
      return value;
    };
  const auto usable = [&](size_t b) {
      const float filtered = filtered_vx(b);
      if (!finite_cost(filtered) || !finite_cost(costs_(b))) {return false;}
      const bool new_gear = gear * state_.cvx(b, index) > 0 && gear * filtered > 0;
      // A handoff can occur while measured motion is still in the old gear.
      // Optionally retain a strictly slower old-gear emitted command as a
      // braking transition. Never permit old-gear acceleration or persistence
      // once measured speed has crossed zero. The default mask is unchanged.
      constexpr float reduction = 0.005f;
      const float measured = static_cast<float>(state_.speed.linear.x);
      const bool braking = allow_path_direction_braking_ && finite_cost(measured) &&
        gear * measured < -reduction && gear * filtered < 0 &&
        std::abs(filtered) <= std::abs(measured) - reduction;
      return new_gear || braking;
    };
  for (size_t b = 0; b < costs_.size(); ++b) {
    if (usable(b)) {
      ++eligible;minimum = std::min(minimum, costs_(b));
    }
  }
  if (eligible == 0) {
    // Preserve the rejected batch at the failure itself. A stationary snapshot
    // taken only after updateControlSequence cannot capture this exception.
    // Explicit diagnostic opt-in; never suppress the original control failure.
    const char * rejection_path = std::getenv("MPPI_STALL_SNAPSHOT_PATH");
    static bool rejection_written = false;
    if (rejection_path && *rejection_path && !rejection_written) {
      try {
        std::ostringstream out;
        out << std::setprecision(17)
            << "{\"scope\":\"Rejected native direction-conditioning batch; no control update completed\","
            << "\"stamp\":" << state_.pose.header.stamp.sec +
          state_.pose.header.stamp.nanosec * 1e-9
            << ",\"gear\":" << gear << ",\"command_index\":" << index
            << ",\"smoothing\":" << (smooth_control_sequence_ ? "true" : "false")
            << ",\"real_path_point_count\":" << real
            << ",\"active_cusp_direction\":" << active_cusp_direction_
          << ",\"effective_cusp_tolerances\":" << serializedCuspTolerances()
            << ",\"history\":[";
        for (size_t k = 0; k < control_history_.size(); ++k) {
          if (k) {out << ',';}
          out << control_history_[k].vx;
        }
        out << "],\"sampled_cvx\":"; snapshotTensor(out, state_.cvx);
        out << ",\"regularized_costs\":"; snapshotTensor(out, costs_);
        out << ",\"filtered_vx\":[";
        for (size_t b = 0; b < costs_.size(); ++b) {
          if (b) {out << ',';}
          out << filtered_vx(b);
        }
        out << "]}\n";
        std::ofstream file(std::string(rejection_path) + ".direction_rejection.json");
        if (!file) {throw std::runtime_error("Cannot open rejected batch diagnostic");}
        file << out.str();file.close();
        if (!file) {throw std::runtime_error("Cannot finish rejected batch diagnostic");}
        rejection_written = true;
      } catch (const std::exception & error) {
        RCLCPP_WARN(logger_, "Optional rejected batch diagnostic failed: %s", error.what());
      }
    }
    throw DirectionSamplingFailure();
  }
  // Renormalize against the eligible minimum; simply zeroing old exponents
  // could leave every eligible sample underflowed and yield a NaN command.
  for (size_t b = 0; b < costs_.size(); ++b) {
    exponents(b) = usable(b) ?
      std::exp(-(costs_(b) - minimum) / settings_.temperature) : 0.0f;
  }
  return true;
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
  conditionControlExponents(exponents);
  auto && softmaxes = xt::eval(exponents / xt::sum(exponents, immediate));
  auto && softmaxes_extened = xt::eval(xt::view(softmaxes, xt::all(), xt::newaxis()));

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

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp)
{
  return getControlFromSequenceAsTwist(stamp, nullptr);
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp, const models::ControlSequence * warm_start)
{
  return getControlFromSequenceAsTwist(stamp, warm_start, nullptr);
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp, const models::ControlSequence * warm_start,
  const models::ControlSequence * before_filter)
{
  // Savitzky-Golay filtering follows optimization and can invalidate its
  // bounds. In particular, signed vx can cancel while wz does not, violating
  // the Ackermann cone. Project the final sequence before extracting output.
  applyControlSequenceConstraints();
  enforceShortCuspMeanFeasibility(warm_start);
  enforceLongEntryCertificate(true);
  preserveFilteredCollisionFeasibility(before_filter);
  enforceLongEntryCertificate(true);  // Nothing may invalidate the final emitted prefix.
  enforceGentleCuspEmission();  // Recheck after smoothing and collision replacements.
  unsigned int offset = settings_.shift_control_sequence ? 1 : 0;

  auto vx = control_sequence_.vx(offset);
  auto wz = control_sequence_.wz(offset);

  if (motor_deadband_floor_ > 0.0f) {
    float target_x = 0.0f;
    float target_y = 0.0f;
    bool has_target = false;

    if (active_cusp_direction_ == 0) {
      target_x = goal_.position.x;
      target_y = goal_.position.y;
      has_target = true;
    } else {
      const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
      if (real > 0) {
        target_x = path_.x(real - 1);
        target_y = path_.y(real - 1);
        has_target = true;
      }
    }

    if (has_target) {
      const float dist_to_target = std::hypot(
        state_.pose.pose.position.x - target_x,
        state_.pose.pose.position.y - target_y);
      if (dist_to_target <= motor_deadband_approach_radius_ &&
          dist_to_target > motor_deadband_terminal_radius_)
      {
        bool aligned_to_path = true;
        if (path_.yaws.size() > 0) {
          const double robot_yaw = tf2::getYaw(state_.pose.pose.orientation);
          const double path_yaw = path_.yaws(0);
          const double heading_err = std::abs(
            angles::shortest_angular_distance(robot_yaw, path_yaw));
          const float max_heading = (active_cusp_direction_ != 0) ?
            0.5236f : motor_deadband_max_heading_error_;
          aligned_to_path = (heading_err <= max_heading);
        }

        if (aligned_to_path) {
          int effective_direction = (active_cusp_direction_ != 0) ?
            active_cusp_direction_ : path_direction_;

          // On final segment (active_cusp_direction_ == 0), align effective_direction with
          // the target's relative longitudinal displacement (d_long) to prevent stale path_direction_
          // from inverting recovery commands.
          // When the goal is abeam (beside the robot, |d_long| <= d_long_deadband), longitudinal
          // displacement is near zero: follow the optimizer's commanded direction rather than
          // flipping sign or reverting to stale path_direction_.
          if (active_cusp_direction_ == 0) {
            const double robot_yaw = tf2::getYaw(state_.pose.pose.orientation);
            const float dx = target_x - static_cast<float>(state_.pose.pose.position.x);
            const float dy = target_y - static_cast<float>(state_.pose.pose.position.y);
            const float d_long = dx * static_cast<float>(std::cos(robot_yaw)) +
                                 dy * static_cast<float>(std::sin(robot_yaw));
            constexpr float d_long_deadband = 0.08f;
            if (vx > 0.005f) {
              // Deliberate forward command: apply forward deadband floor only if target is ahead
              // or abeam; never invert forward motion into reverse.
              effective_direction = (d_long >= -d_long_deadband) ? 1 : 0;
            } else if (vx < -0.005f) {
              // Deliberate reverse command: apply reverse deadband floor only if target is behind
              // or abeam; never invert reverse motion into forward.
              effective_direction = (d_long <= d_long_deadband) ? -1 : 0;
            } else {
              // Commanded standstill or noise (|vx| <= 0.005):
              // If target is clearly ahead/behind, apply floor toward target; if abeam, preserve standstill.
              if (d_long > d_long_deadband) {
                effective_direction = 1;
              } else if (d_long < -d_long_deadband) {
                effective_direction = -1;
              } else {
                effective_direction = 0;
              }
            }
          }

          if (effective_direction > 0) {
            if (vx >= 0.0f || std::abs(vx) < motor_deadband_floor_) {
              vx = std::max(vx, motor_deadband_floor_);
            }
          } else if (effective_direction < 0) {
            if (vx <= 0.0f || std::abs(vx) < motor_deadband_floor_) {
              vx = std::min(vx, -motor_deadband_floor_);
            }
          }
        }
      }
    }
  }

  control_sequence_.vx(offset) = vx;

  // Filter history must contain the command actually sent, not the
  // unconstrained filtered value that was recorded before this projection.
  control_history_.back().vx = vx;
  control_history_.back().vy = isHolonomic() ? control_sequence_.vy(offset) : 0.0f;
  control_history_.back().wz = wz;

  // Optional bounded emission diagnostic. The single optimizer snapshot can
  // be spent on an earlier gear change; this captures the actual final mean
  // across later retries without dumping or replacing the noisy batch. It is
  // not a costmap snapshot, physical forecast, or proof of command delivery.
  const char * mean_trace = std::getenv("MPPI_CUSP_MEAN_TRACE_PATH");
  const char * all_paths = std::getenv("MPPI_EMITTED_MEAN_TRACE_ALL_PATHS");
  const bool trace_all_paths = all_paths && std::string(all_paths) == "1";
  const size_t real = std::min(critics_data_.real_path_point_count.value_or(0), path_.x.size());
  static size_t traced_means = 0;
  static int previous_traced_gear = 0;
  static size_t traced_handoff_grids = 0;
  if (mean_trace && *mean_trace && real > 0 && traced_means < 10000 &&
    (trace_all_paths || (active_cusp_direction_ != 0 && real <= 4)))
  {
    try {
      std::ostringstream out;
      out << std::setprecision(17) << "{\"stamp_ns\":" << rclcpp::Time(stamp).nanoseconds()
          << ",\"pose_stamp_ns\":" << rclcpp::Time(state_.pose.header.stamp).nanoseconds()
          << ",\"frame\":" << std::quoted(state_.pose.header.frame_id)
          << ",\"scope\":\"Final native mean before emission; no physical or collision forecast\""
          << ",\"real_path_point_count\":" << real
          << ",\"active_cusp_direction\":" << active_cusp_direction_
          << ",\"effective_cusp_tolerances\":" << serializedCuspTolerances()
          << ",\"path_direction\":" << path_direction_
          << ",\"command_offset\":" << offset
          << ",\"model_dt\":" << settings_.model_dt
          << ",\"robot\":[" << state_.pose.pose.position.x << ','
          << state_.pose.pose.position.y << ',' << tf2::getYaw(state_.pose.pose.orientation)
          << "],\"speed\":[" << state_.speed.linear.x << ',' << state_.speed.angular.z
          << "],\"command\":[" << vx << ',' << wz
          << "],\"goal\":[" << goal_.position.x << ',' << goal_.position.y << ','
          << tf2::getYaw(goal_.orientation) << ']';
      const auto ackermann = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
      const auto feedback = [&](const char * name, const std::optional<float> & value) {
          out << ",\"" << name << "\":";
          if (value) {out << *value;} else {out << "null";}
        };
      feedback("initial_steering", ackermann ? ackermann->initialSteering() : std::nullopt);
      feedback("initial_conditioned_steering",
        ackermann ? ackermann->initialConditionedSteering() : std::nullopt);
      const auto field = [&](const char * name, const auto & tensor) {
          out << ",\"" << name << "\":"; snapshotTensor(out, tensor);
        };
      field("control_vx", control_sequence_.vx); field("control_wz", control_sequence_.wz);
      field("path_x", path_.x); field("path_y", path_.y); field("path_yaw", path_.yaws);
      // Optional trace only: expose the carried tail before shiftControlSequence.
      // Logging activation alone cannot establish when that tail was discarded.
      out << ",\"completion_attempted\":" << (cusp_completion_attempted_ ? "true" : "false")
          << ",\"completion_available\":" << (cusp_completion_ ? "true" : "false")
          << ",\"completion_shifts\":" << cusp_completion_shifts_;
      if (cusp_completion_) {
        field("completion_vx", cusp_completion_->vx);
        field("completion_wz", cusp_completion_->wz);
      }
      out << ",\"history\":[";
      for (size_t k = 0; k < control_history_.size(); ++k) {
        if (k) {out << ',';}
        out << '[' << control_history_[k].vx << ',' << control_history_[k].wz << ']';
      }
      out << ']';
      const char * handoff_grids = std::getenv("MPPI_EMITTED_HANDOFF_GRID_TRACE");
      if (handoff_grids && std::string(handoff_grids) == "1" &&
        path_direction_ != 0 && previous_traced_gear != 0 &&
        path_direction_ != previous_traced_gear && traced_handoff_grids < 64)
      {
        // The controller holds the costmap mutex throughout computeVelocityCommands.
        // Capture the grid used by this first emitted new-gear mean, not a later
        // asynchronous /costmap publication. Explicit diagnostic opt-in only.
        const size_t cells = static_cast<size_t>(costmap_->getSizeInCellsX()) *
          costmap_->getSizeInCellsY();
        if (cells > 250000) {throw std::runtime_error("Handoff grid exceeds capture bound");}
        out << ",\"handoff_grid_index\":" << traced_handoff_grids
            << ",\"previous_emitted_gear\":" << previous_traced_gear
            << ",\"footprint\":[";
        const auto footprint = costmap_ros_->getRobotFootprint();
        for (size_t k = 0; k < footprint.size(); ++k) {
          if (k) {out << ',';}
          out << '[' << footprint[k].x << ',' << footprint[k].y << ']';
        }
        out << "],\"grid\":{\"width\":" << costmap_->getSizeInCellsX()
            << ",\"height\":" << costmap_->getSizeInCellsY()
            << ",\"resolution\":" << costmap_->getResolution()
            << ",\"origin\":[" << costmap_->getOriginX() << ',' << costmap_->getOriginY()
            << "],\"data\":[";
        for (size_t k = 0; k < cells; ++k) {
          if (k) {out << ',';}
          out << static_cast<unsigned int>(costmap_->getCharMap()[k]);
        }
        out << "]}";
        ++traced_handoff_grids;
      }
      out << "}\n";
      std::ofstream file(mean_trace, std::ios::out | std::ios::app);
      if (!file) {throw std::runtime_error("Cannot open cusp mean trace");}
      file << out.str(); file.close();
      if (!file) {throw std::runtime_error("Cannot finish cusp mean trace");}
      ++traced_means;
      if (path_direction_ != 0) {previous_traced_gear = path_direction_;}
    } catch (const std::exception & error) {
      RCLCPP_WARN_THROTTLE(logger_, *parent_.lock()->get_clock(), 5000,
        "Optional cusp mean trace failed: %s", error.what());
    }
  }

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

}  // namespace mppi
