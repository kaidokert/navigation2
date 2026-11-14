// Copyright (c) 2026 Kaido Kert
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

#include "nav2_mppi_controller/critics/terminal_approach_critic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "angles/angles.h"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "tf2/utils.h"

namespace mppi::critics
{

namespace
{

// Bit-level finite check immune to -ffast-math / -ffinite-math-only
template<typename T>
bool finiteValue(T value)
{
  static_assert(std::is_floating_point_v<T>, "finiteValue expects a floating-point type");
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
  } else {
    uint64_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
  }
}

constexpr float kPullthroughTaperSlope = 1.5f;           // GENERIC: Taper slope past release
constexpr double kInitialSegmentRayExtrapM = 0.10;       // GENERIC: Max back-proj on seg 0 (m)
constexpr double kMinBehindGoalLonM = -0.05;             // GENERIC: Min lon behind goal (m)
constexpr double kMaxCaptureAngleRad = 1.4835;           // GENERIC: Max capture angle (85 deg)
constexpr double kMinCaptureCurveLenM = 0.05;            // GENERIC: Min capture curve len (m)
constexpr double kMaxCurveChordRatio = 2.0;              // GENERIC: Max curve/chord len ratio
constexpr double kMinForwardCosine = 0.10;               // GENERIC: Min cos(theta0) forward
constexpr double kBoundedBiarcMaxXEndM = 0.015;          // GENERIC: Max lon offset for 2a (m)
constexpr double kBoundedBiarcXEndRatio = 0.05;          // GENERIC: Max lon ratio of |X0|
constexpr double kGoalBoxXyTargetRatio = 0.25;           // GENERIC: Endpoint XY / goal_xy_tol
constexpr double kGoalBoxYawTargetRatio = 0.75;          // GENERIC: Endpoint yaw / goal_yaw_tol
constexpr double kTurnCorridorYawAllowanceRatio = 0.35;  // GENERIC: Extra yaw per rad path turn
constexpr double kMaxCollisionCheckStepM = 0.025;        // GENERIC: Max collision step (m)
constexpr double kMaxGhostRunwayScanM = 1.00;            // GENERIC: Max runway scan len (m)
constexpr double kGoalMatchPosTolM = 0.05;               // GENERIC: Max pos diff for goal (m)
constexpr double kGoalMatchYawTolRad = 0.10;             // GENERIC: Max yaw diff for goal (rad)
constexpr double kGlobalGoalPosTol = 1e-3;               // GENERIC: Global-plan goal pos tol (m)
constexpr double kGlobalGoalYawTol = 1e-3;               // GENERIC: Global-plan goal yaw tol (rad)
constexpr double kCollinearStubLatTolM = 1e-4;           // GENERIC: Collinear stub lat tol (m)
constexpr double kCollinearStubYawTolRad = 1e-3;         // GENERIC: Collinear stub yaw tol (rad)
constexpr double kCollinearStubSpanMarginM = 1e-3;       // GENERIC: Collinear stub span margin (m)
constexpr double kLatchRadiusHysteresisM = 0.05;         // GENERIC: Latch hysteresis margin (m)
constexpr double kReverseNoseThresholdM = -0.02;         // GENERIC: Reverse nose threshold (m)
constexpr double kReleaseZoneGoalDistM = 0.05;           // GENERIC: Settle endpoint dist (m)
constexpr float kCurveYawErrorScaleSq = 0.36f;           // GENERIC: (0.6 m/rad)^2 yaw scale
constexpr float kFollowTerminalWeightRatio = 0.65f;      // GENERIC: Terminal geodesic share
constexpr float kFollowMeanWeightRatio = 0.35f;          // GENERIC: Mean geodesic share
constexpr float kSettleAngularSpeedScale = 0.25f;        // GENERIC: Settle wz weight scale (m)
constexpr double kTurnSagittaDivisor = 8.0;              // GENERIC: Arc chord sagitta L^2/(8R)

}  // namespace

TerminalApproachCritic::~TerminalApproachCritic()
{
  if (callback_guard_) {
    *callback_guard_ = false;
  }
}

void TerminalApproachCritic::initialize()
{
  callback_guard_ = std::make_shared<bool>(true);
  auto getParam = parameters_handler_->getParamGetter(name_);

  // GENERIC: Outer approach-circle entry radius around the terminal goal (m)
  getParam(approach_radius_, "approach_radius", 0.65f);
  // GENERIC: Inner stock-critic cutoff radius around the terminal goal (m)
  getParam(cutoff_radius_, "cutoff_radius", 0.35f);
  // GENERIC: Minimum goal distance for initial capture curve synthesis (m)
  getParam(min_approach_distance_, "min_approach_distance", 0.16f);
  // GENERIC: Short/post-cusp final approach distance below which viable poses pass through (m)
  getParam(short_approach_distance_, "short_approach_distance", 0.28f);
  // ROBOT: Minimum Ackermann turning radius L / tan(delta_max) = 0.214 / tan(0.3644) (m)
  getParam(min_turning_radius_, "min_turning_radius", 0.5605f);
  // GENERIC: Minimum cross-track offset at circle entry to latch capture curve (m)
  getParam(activation_drift_, "activation_drift", 0.020f);
  // GENERIC: Minimum heading error at circle entry to latch capture curve (rad)
  getParam(activation_yaw_drift_, "activation_yaw_drift", 0.080f);
  // GENERIC: Maximum reference-path-to-goal turn angle at circle entry for takeover (rad)
  getParam(max_entry_path_turn_angle_, "max_entry_path_turn_angle", 0.50f);
  // GENERIC: Fallback XY goal tolerance when GoalChecker pointer is null (m)
  getParam(goal_xy_tolerance_, "goal_xy_tolerance", 0.10f);
  // GENERIC: Fallback yaw goal tolerance when GoalChecker pointer is null (rad)
  getParam(goal_yaw_tolerance_, "goal_yaw_tolerance", 0.25f);
  // GENERIC: Fraction of xy_goal_tolerance within which reference-path tracking is viable
  getParam(viable_path_tracking_ratio_, "viable_path_tracking_ratio", 0.95f);
  // GENERIC: Maximum bounded-R_min lateral residual allowed at goal (m)
  getParam(max_residual_y_, "max_residual_y", 0.025f);
  // GENERIC: Maximum short-range heading residual allowed at goal (rad)
  getParam(max_residual_yaw_, "max_residual_yaw", 0.19f);
  // GENERIC: Maximum post-latch cross-track drift before escalation/redraw (m)
  getParam(max_tracking_drift_, "max_tracking_drift", 0.045f);
  // GENERIC: Goal position shift threshold for new-goal reset vs odom drift (m)
  getParam(goal_reset_pos_tol_, "goal_reset_pos_tol", 0.25f);
  // GENERIC: Goal yaw shift threshold for new-goal reset vs odom drift (rad)
  getParam(goal_reset_yaw_tol_, "goal_reset_yaw_tol", 0.35f);
  // GENERIC: Maximum mid-approach redraws allowed (0 in unit tests and synthetic gyms)
  getParam(max_redraws_, "max_redraws", 0);
  // GENERIC: Escalation policy when infeasible ("abort" | "replan_signal" | "fallback")
  getParam(infeasible_policy_, "infeasible_policy", std::string("abort"));
  // GENERIC: Remaining along-track distance window for smooth pull-through taper (m)
  getParam(pullthrough_window_, "pullthrough_window", 0.25f);
  // GENERIC: Maximum pull-through extension length past goal (m)
  getParam(pullthrough_length_, "pullthrough_length", 0.10f);
  // GENERIC: Remaining along-track distance at or below which pull-through is 0.0 m (m)
  getParam(pullthrough_release_, "pullthrough_release", 0.015f);
  // GENERIC: Arc-length sample spacing along capture curve (m)
  getParam(sample_spacing_, "sample_spacing", 0.025f);
  // GENERIC: Curve cross-track & tangent alignment weight along capture curve
  getParam(align_weight_, "align_weight", 45.0f);
  // GENERIC: Geodesic arc-length progress weight along capture curve
  getParam(follow_weight_, "follow_weight", 55.0f);
  // GENERIC: Release-zone velocity settling weight
  getParam(settle_weight_, "settle_weight", 12.0f);
  // GENERIC: Rollout point stride for integrated-distance alignment scoring
  getParam(trajectory_point_step_, "trajectory_point_step", 2);
  // GENERIC: Cost power exponent
  getParam(power_, "cost_power", 1);

  if (enabled_ && infeasible_policy_ == "replan_signal") {
    ensureReplanPublisher(false);
  }

  if (parameters_handler_ != nullptr) {
    std::weak_ptr<bool> weak_guard = callback_guard_;
    parameters_handler_->addPostCallback(
      [this, weak_guard]() {
        auto guard = weak_guard.lock();
        if (!guard || !*guard) {
          return;
        }
        stock_thresholds_checked_ = false;
        warned_stock_threshold_mismatch_ = false;
        if (hasParameterChangedSinceLatch()) {
          const bool consume_latch =
          latch_consumed_ || latched_ || approach_escalated_ ||
          replan_pending_ || replan_allowance_consumed_;
          if (consume_latch) {
            if (replan_signal_sent_) {
              if (is_active_ && replan_pub_ != nullptr && replan_pub_->is_activated()) {
                std_msgs::msg::Bool sig;
                sig.data = false;
                replan_pub_->publish(sig);
                pending_replan_clear_ = false;
              } else {
                pending_replan_clear_ = true;
              }
              replan_signal_sent_ = false;
            }
            replan_pending_ = false;
            const bool had_active_latch = latched_ && !approach_escalated_;
            clearActiveCurve();
            if (replan_allowance_consumed_) {
              approach_escalated_ = true;
              escalation_reason_ = "PARAM_CHANGED_AFTER_REPLAN_CONSUMED";
              if (last_abort_msg_.empty()) {
                last_abort_msg_ =
                "[TERMINAL_APPROACH_ABORT] reason=PARAM_CHANGED_AFTER_REPLAN_CONSUMED";
              }
              RCLCPP_INFO(
                logger_,
                "[TERMINAL_APPROACH_ESCALATE] policy=%s "
                "reason=PARAM_CHANGED_AFTER_REPLAN_CONSUMED "
                "goal_dist=0.0000 cross_track=0.0000 yaw_err=0.0000 redraws=%d/%d replans=%d",
                infeasible_policy_.c_str(), redraw_count_, max_redraws_, replan_count_);
            } else if (max_redraws_ == 0 || redraw_count_ >= max_redraws_ || !had_active_latch) {
              approach_escalated_ = true;
              if (escalation_reason_.empty()) {
                escalation_reason_ = "PARAM_CHANGE_WHILE_LATCHED";
              }
              if (had_active_latch) {
                RCLCPP_INFO(
                  logger_,
                  "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=PARAM_CHANGE_WHILE_LATCHED "
                  "goal_dist=0.0000 cross_track=0.0000 yaw_err=0.0000 redraws=%d/%d replans=%d",
                  infeasible_policy_.c_str(), redraw_count_, max_redraws_, replan_count_);
                requestReplanIfEligible(0.0);
              }
            } else {
              ++redraw_count_;
            }
            tick_evaluated_ = false;
            snapshotLatchParameters();
          } else {
            resetLatch();
          }
        }
      });
  }

  replan_allowance_consumed_ = false;
  replan_pending_ = false;
  replan_count_ = 0;
  stock_thresholds_checked_ = false;
  warned_stock_threshold_mismatch_ = false;
  syncStockCriticThresholds(false);
  resetLatch();
}

void TerminalApproachCritic::on_activate()
{
  is_active_ = true;
  replan_allowance_consumed_ = false;
  replan_pending_ = false;
  replan_count_ = 0;
  has_plan_goal_ = false;
  has_plan_gear_ = false;
  stock_thresholds_checked_ = false;
  warned_stock_threshold_mismatch_ = false;
  syncStockCriticThresholds(true);
  if (replan_pub_ != nullptr && !replan_pub_->is_activated()) {
    replan_pub_->on_activate();
  }
  resetLatch();
  if (pending_replan_clear_ && replan_pub_ != nullptr && replan_pub_->is_activated()) {
    std_msgs::msg::Bool sig;
    sig.data = false;
    replan_pub_->publish(sig);
    pending_replan_clear_ = false;
  }
}

void TerminalApproachCritic::on_deactivate()
{
  is_active_ = false;
  replan_allowance_consumed_ = false;
  replan_pending_ = false;
  replan_count_ = 0;
  if (replan_pub_ != nullptr && replan_pub_->is_activated()) {
    replan_pub_->on_deactivate();
  }
}

void TerminalApproachCritic::on_cleanup()
{
  is_active_ = false;
  pending_replan_clear_ = false;
  replan_allowance_consumed_ = false;
  replan_pending_ = false;
  replan_count_ = 0;
  if (callback_guard_) {
    *callback_guard_ = false;
  }
  if (replan_pub_ != nullptr && replan_pub_->is_activated()) {
    replan_pub_->on_deactivate();
  }
  replan_pub_.reset();
}

void TerminalApproachCritic::ensureReplanPublisher(bool activate_now)
{
  if (replan_pub_ == nullptr) {
    if (auto node = parent_.lock()) {
      const std::string topic =
        parent_name_.empty() ? "terminal_replan_requested" :
        parent_name_ + "/terminal_replan_requested";
      replan_pub_ = node->create_publisher<std_msgs::msg::Bool>(
        topic, rclcpp::QoS(1).transient_local());
    }
  }
  if (activate_now && is_active_ && replan_pub_ != nullptr && !replan_pub_->is_activated()) {
    replan_pub_->on_activate();
  }
}

void TerminalApproachCritic::clearActiveCurve()
{
  latched_ = false;
  latched_is_reference_path_ = false;
  latch_initial_cross_track_ = 0.0;
  latched_curve_.clear();
  cached_active_path_.clear();
}

void TerminalApproachCritic::reset()
{
  replan_allowance_consumed_ = false;
  replan_pending_ = false;
  replan_count_ = 0;
  has_plan_goal_ = false;
  has_plan_gear_ = false;
  last_plan_goal_ = geometry_msgs::msg::Pose{};
  last_plan_frame_.clear();
  last_plan_gear_ = 1.0;
  resetLatch();
}

void TerminalApproachCritic::onNewPlan(const nav_msgs::msg::Path & plan)
{
  if (plan.poses.empty()) {
    reset();
    return;
  }

  const auto & plan_goal = plan.poses.back().pose;
  const std::string & plan_frame = plan.header.frame_id;
  double plan_gear = 1.0;
  bool has_plan_gear = false;
  for (size_t i = plan.poses.size() - 1u; i-- > 0u; ) {
    const double dx = plan.poses[i + 1u].pose.position.x - plan.poses[i].pose.position.x;
    const double dy = plan.poses[i + 1u].pose.position.y - plan.poses[i].pose.position.y;
    if (std::hypot(dx, dy) > 1e-6) {
      const double th = tf2::getYaw(plan.poses[i].pose.orientation);
      const double dot = dx * std::cos(th) + dy * std::sin(th);
      plan_gear = (dot < 0.0) ? -1.0 : 1.0;
      has_plan_gear = true;
      break;
    }
  }

  bool same_goal = false;
  if (has_plan_goal_) {
    const double dpos = std::hypot(
      plan_goal.position.x - last_plan_goal_.position.x,
      plan_goal.position.y - last_plan_goal_.position.y);
    const double dyaw = std::fabs(
      angles::shortest_angular_distance(
        tf2::getYaw(last_plan_goal_.orientation),
        tf2::getYaw(plan_goal.orientation)));
    same_goal = (plan_frame == last_plan_frame_) &&
      (dpos <= kGlobalGoalPosTol) &&
      (dyaw <= kGlobalGoalYawTol);
  } else if (!latched_frame_id_.empty() && plan_frame == latched_frame_id_) {
    const double dpos = std::hypot(
      plan_goal.position.x - latched_goal_.position.x,
      plan_goal.position.y - latched_goal_.position.y);
    const double dyaw = std::fabs(
      angles::shortest_angular_distance(
        tf2::getYaw(latched_goal_.orientation),
        tf2::getYaw(plan_goal.orientation)));
    same_goal = (dpos <= kGlobalGoalPosTol) && (dyaw <= kGlobalGoalYawTol);
  }
  const double ref_gear = has_plan_gear_ ? last_plan_gear_ : latched_gear_;
  const bool same_gear = !has_plan_gear || (plan_gear * ref_gear > 0.0);

  auto savePlanMetadata = [&]() {
      last_plan_goal_ = plan_goal;
      last_plan_frame_ = plan_frame;
      has_plan_goal_ = true;
      if (has_plan_gear) {
        last_plan_gear_ = plan_gear;
        has_plan_gear_ = true;
      } else if (!same_goal) {
        has_plan_gear_ = false;
      }
    };

  if (replan_pending_) {
    if ((has_plan_goal_ || !latched_frame_id_.empty()) && !same_goal) {
      replan_allowance_consumed_ = false;
      replan_count_ = 0;
    }
    resetLatch();
    savePlanMetadata();
    return;
  }

  if (replan_allowance_consumed_) {
    if ((has_plan_goal_ || !latched_frame_id_.empty()) && !same_goal) {
      replan_allowance_consumed_ = false;
      replan_count_ = 0;
      resetLatch();
      savePlanMetadata();
      return;
    }
    if (latched_ && same_goal && same_gear) {
      tick_evaluated_ = false;
      savePlanMetadata();
      return;
    }
    clearActiveCurve();
    approach_escalated_ = true;
    tick_evaluated_ = false;
    last_abort_msg_ =
      "[TERMINAL_APPROACH_ABORT] reason=SECOND_REPLACEMENT_PLAN_REJECTED";
    savePlanMetadata();
    return;
  }

  if (approach_escalated_ && !last_abort_msg_.empty() &&
    infeasible_policy_ == "abort" && !replan_allowance_consumed_)
  {
    resetLatch();
    savePlanMetadata();
    return;
  }

  if (latch_consumed_ || latched_ || approach_escalated_) {
    if (!same_goal) {
      resetLatch();
    } else if (!same_gear) {
      const bool had_active_latch = latched_ && !approach_escalated_;
      clearActiveCurve();
      latch_consumed_ = true;
      tick_evaluated_ = false;
      if (max_redraws_ == 0 || redraw_count_ >= max_redraws_ || !had_active_latch) {
        approach_escalated_ = true;
        if (escalation_reason_.empty()) {
          escalation_reason_ = "GEAR_FLIP_ON_SAME_GOAL";
        }
        if (last_abort_msg_.empty()) {
          last_abort_msg_ =
            "[TERMINAL_APPROACH_ABORT] reason=GEAR_FLIP_ON_SAME_GOAL";
        }
        if (had_active_latch) {
          requestReplanIfEligible(0.0);
        }
      } else {
        ++redraw_count_;
      }
    } else {
      tick_evaluated_ = false;
    }
    savePlanMetadata();
    return;
  }

  resetLatch();
  savePlanMetadata();
}

void TerminalApproachCritic::resetLatch()
{
  if (replan_signal_sent_) {
    if (is_active_ && replan_pub_ != nullptr && replan_pub_->is_activated()) {
      std_msgs::msg::Bool sig;
      sig.data = false;
      replan_pub_->publish(sig);
      pending_replan_clear_ = false;
    } else {
      pending_replan_clear_ = true;
    }
  }
  latched_ = false;
  latch_consumed_ = false;
  latched_is_reference_path_ = false;
  latch_initial_cross_track_ = 0.0;
  approach_escalated_ = false;
  replan_pending_ = false;
  tick_evaluated_ = false;
  latched_goal_ = geometry_msgs::msg::Pose{};
  latch_origin_goal_ = geometry_msgs::msg::Pose{};
  latched_frame_id_.clear();
  latched_gear_ = 1.0;
  redraw_count_ = 0;
  replan_signal_sent_ = false;
  local_goal_shift_caused_drift_ = false;
  last_abort_msg_.clear();
  escalation_reason_.clear();
  latched_curve_.clear();
  cached_active_path_.clear();
  snapshotLatchParameters();
}

void TerminalApproachCritic::syncStockCriticThresholds(bool log_warning)
{
  stock_thresholds_configured_ = false;
  stock_pfc_active_ = false;
  stock_pac_active_ = false;
  effective_pfc_cutoff_ = cutoff_radius_;
  effective_pac_cutoff_ = cutoff_radius_;
  auto node = parent_.lock();
  if (!node) {
    return;
  }
  const std::string prefix = parent_name_.empty() ? std::string("") : (parent_name_ + ".");
  const std::string pfc_thresh_name = prefix + "PathFollowCritic.threshold_to_consider";
  const std::string pfc_enabled_name = prefix + "PathFollowCritic.enabled";
  const std::string pac_thresh_name = prefix + "PathAlignCritic.threshold_to_consider";
  const std::string pac_enabled_name = prefix + "PathAlignCritic.enabled";

  bool pfc_enabled = true;
  if (node->has_parameter(pfc_enabled_name)) {
    node->get_parameter(pfc_enabled_name, pfc_enabled);
  }
  if (node->has_parameter(pfc_thresh_name)) {
    double val = static_cast<double>(cutoff_radius_);
    if (node->get_parameter(pfc_thresh_name, val) && finiteValue(val)) {
      effective_pfc_cutoff_ = static_cast<float>(std::max(0.0, val));
      stock_pfc_active_ = pfc_enabled;
      stock_thresholds_configured_ = true;
    }
  }

  bool pac_enabled = true;
  if (node->has_parameter(pac_enabled_name)) {
    node->get_parameter(pac_enabled_name, pac_enabled);
  }
  if (node->has_parameter(pac_thresh_name)) {
    double val = static_cast<double>(cutoff_radius_);
    if (node->get_parameter(pac_thresh_name, val) && finiteValue(val)) {
      effective_pac_cutoff_ = static_cast<float>(std::max(0.0, val));
      stock_pac_active_ = pac_enabled;
      stock_thresholds_configured_ = true;
    }
  }

  if (log_warning && enabled_ && cutoff_radius_ > 0.0f && !warned_stock_threshold_mismatch_) {
    const bool pfc_mismatch =
      stock_pfc_active_ && (std::fabs(effective_pfc_cutoff_ - cutoff_radius_) > 1e-3f);
    const bool pac_mismatch =
      stock_pac_active_ && (std::fabs(effective_pac_cutoff_ - cutoff_radius_) > 1e-3f);
    if (pfc_mismatch || pac_mismatch) {
      warned_stock_threshold_mismatch_ = true;
      RCLCPP_WARN(
        logger_,
        "[TERMINAL_APPROACH_CONFIG_WARN] %s.cutoff_radius=%.3f but stock critic "
        "threshold_to_consider (PathFollowCritic=%.3f, PathAlignCritic=%.3f) differs; "
        "aligning effective reference-path handoff to %.3f m.",
        name_.c_str(), static_cast<double>(cutoff_radius_),
        static_cast<double>(effective_pfc_cutoff_),
        static_cast<double>(effective_pac_cutoff_),
        effectiveRefCutoff());
    }
  }
}

double TerminalApproachCritic::effectiveRefCutoff() const
{
  if (cutoff_radius_ <= 0.0f) {
    return 0.0;
  }
  if (!stock_thresholds_configured_) {
    return static_cast<double>(cutoff_radius_);
  }
  if (!stock_pfc_active_ || !stock_pac_active_) {
    return static_cast<double>(std::max(cutoff_radius_, approach_radius_));
  }
  const double stock_handoff = std::max(
    static_cast<double>(effective_pfc_cutoff_),
    static_cast<double>(effective_pac_cutoff_));
  return std::clamp(
    stock_handoff, 0.0, static_cast<double>(std::max(cutoff_radius_, approach_radius_)));
}

void TerminalApproachCritic::snapshotLatchParameters()
{
  cached_enabled_ = enabled_;
  cached_approach_radius_ = approach_radius_;
  cached_cutoff_radius_ = cutoff_radius_;
  cached_min_approach_distance_ = min_approach_distance_;
  cached_short_approach_distance_ = short_approach_distance_;
  cached_min_turning_radius_ = min_turning_radius_;
  cached_activation_drift_ = activation_drift_;
  cached_activation_yaw_drift_ = activation_yaw_drift_;
  cached_max_entry_path_turn_angle_ = max_entry_path_turn_angle_;
  cached_goal_xy_tolerance_ = goal_xy_tolerance_;
  cached_goal_yaw_tolerance_ = goal_yaw_tolerance_;
  cached_viable_path_tracking_ratio_ = viable_path_tracking_ratio_;
  cached_max_residual_y_ = max_residual_y_;
  cached_max_residual_yaw_ = max_residual_yaw_;
  cached_max_tracking_drift_ = max_tracking_drift_;
  cached_goal_reset_pos_tol_ = goal_reset_pos_tol_;
  cached_goal_reset_yaw_tol_ = goal_reset_yaw_tol_;
  cached_max_redraws_ = max_redraws_;
  cached_infeasible_policy_ = infeasible_policy_;
  cached_pullthrough_window_ = pullthrough_window_;
  cached_pullthrough_length_ = pullthrough_length_;
  cached_pullthrough_release_ = pullthrough_release_;
  cached_sample_spacing_ = sample_spacing_;
  cached_align_weight_ = align_weight_;
  cached_follow_weight_ = follow_weight_;
  cached_settle_weight_ = settle_weight_;
  cached_trajectory_point_step_ = trajectory_point_step_;
  cached_power_ = power_;
}

bool TerminalApproachCritic::hasParameterChangedSinceLatch() const
{
  return cached_enabled_ != enabled_ ||
         std::fabs(cached_approach_radius_ - approach_radius_) > 1e-6f ||
         std::fabs(cached_cutoff_radius_ - cutoff_radius_) > 1e-6f ||
         std::fabs(cached_min_approach_distance_ - min_approach_distance_) > 1e-6f ||
         std::fabs(cached_short_approach_distance_ - short_approach_distance_) > 1e-6f ||
         std::fabs(cached_min_turning_radius_ - min_turning_radius_) > 1e-6f ||
         std::fabs(cached_activation_drift_ - activation_drift_) > 1e-6f ||
         std::fabs(cached_activation_yaw_drift_ - activation_yaw_drift_) > 1e-6f ||
         std::fabs(cached_max_entry_path_turn_angle_ - max_entry_path_turn_angle_) > 1e-6f ||
         std::fabs(cached_goal_xy_tolerance_ - goal_xy_tolerance_) > 1e-6f ||
         std::fabs(cached_goal_yaw_tolerance_ - goal_yaw_tolerance_) > 1e-6f ||
         std::fabs(cached_viable_path_tracking_ratio_ - viable_path_tracking_ratio_) > 1e-6f ||
         std::fabs(cached_max_residual_y_ - max_residual_y_) > 1e-6f ||
         std::fabs(cached_max_residual_yaw_ - max_residual_yaw_) > 1e-6f ||
         std::fabs(cached_max_tracking_drift_ - max_tracking_drift_) > 1e-6f ||
         std::fabs(cached_goal_reset_pos_tol_ - goal_reset_pos_tol_) > 1e-6f ||
         std::fabs(cached_goal_reset_yaw_tol_ - goal_reset_yaw_tol_) > 1e-6f ||
         cached_max_redraws_ != max_redraws_ ||
         cached_infeasible_policy_ != infeasible_policy_ ||
         std::fabs(cached_pullthrough_window_ - pullthrough_window_) > 1e-6f ||
         std::fabs(cached_pullthrough_length_ - pullthrough_length_) > 1e-6f ||
         std::fabs(cached_pullthrough_release_ - pullthrough_release_) > 1e-6f ||
         std::fabs(cached_sample_spacing_ - sample_spacing_) > 1e-6f ||
         std::fabs(cached_align_weight_ - align_weight_) > 1e-6f ||
         std::fabs(cached_follow_weight_ - follow_weight_) > 1e-6f ||
         std::fabs(cached_settle_weight_ - settle_weight_) > 1e-6f ||
         cached_trajectory_point_step_ != trajectory_point_step_ ||
         cached_power_ != power_;
}

float TerminalApproachCritic::computePullthroughLength(float along_track_remaining) const
{
  const float max_len = std::max(0.0f, pullthrough_length_);
  const float release_dist = std::max(0.0f, pullthrough_release_);
  const float window_dist = std::max(release_dist, pullthrough_window_);
  if (max_len <= 0.0f || along_track_remaining > window_dist ||
    along_track_remaining <= release_dist)
  {
    return 0.0f;
  }
  return std::clamp(
    kPullthroughTaperSlope * (along_track_remaining - release_dist), 0.0f, max_len);
}

bool TerminalApproachCritic::isFootprintCollisionFree(
  double wx, double wy, double wyaw) const
{
  if (costmap_ == nullptr || costmap_ros_ == nullptr ||
    !finiteValue(wx) || !finiteValue(wy) || !finiteValue(wyaw))
  {
    return false;
  }

  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap_->getMutex()));
  unsigned int mx = 0u, my = 0u;
  if (!costmap_->worldToMap(wx, wy, mx, my)) {
    return false;
  }
  const unsigned char center_cost = costmap_->getCost(mx, my);
  if (center_cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
    center_cost == nav2_costmap_2d::NO_INFORMATION)
  {
    return false;
  }

  auto * layered = costmap_ros_->getLayeredCostmap();
  const auto & footprint =
    (layered != nullptr && !layered->getFootprint().empty()) ?
    layered->getFootprint() : costmap_ros_->getRobotFootprint();
  if (costmap_ros_->getUseRadius() &&
    center_cost >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
  {
    return false;
  }
  if (footprint.size() < 3u) {
    return center_cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
  }

  const double cos_th = std::cos(wyaw);
  const double sin_th = std::sin(wyaw);
  for (const auto & pt : footprint) {
    const double vx = wx + (pt.x * cos_th - pt.y * sin_th);
    const double vy = wy + (pt.x * sin_th + pt.y * cos_th);
    unsigned int vmx = 0u, vmy = 0u;
    if (!costmap_->worldToMap(vx, vy, vmx, vmy)) {
      return false;
    }
  }

  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  const double fp_cost = checker.footprintCostAtPose(wx, wy, wyaw, footprint);
  if (fp_cost < 0.0 ||
    fp_cost >= static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE) ||
    fp_cost == static_cast<double>(nav2_costmap_2d::NO_INFORMATION))
  {
    return false;
  }
  return true;
}

bool TerminalApproachCritic::isReferencePathCollisionFree(
  const models::Path & path,
  size_t start_idx,
  size_t end_idx) const
{
  const size_t n = path.x.shape(0);
  if (n == 0u || start_idx >= n) {
    return false;
  }
  const size_t last_idx = std::min(end_idx, n - 1u);
  if (start_idx > last_idx) {
    return false;
  }
  for (size_t i = start_idx; i <= last_idx; ++i) {
    const double ax = static_cast<double>(path.x(i));
    const double ay = static_cast<double>(path.y(i));
    const double ath = static_cast<double>(path.yaws(i));
    if (!isFootprintCollisionFree(ax, ay, ath)) {
      return false;
    }
    if (i < last_idx) {
      const double bx = static_cast<double>(path.x(i + 1u));
      const double by = static_cast<double>(path.y(i + 1u));
      const double bth = static_cast<double>(path.yaws(i + 1u));
      const double seg_len = std::hypot(bx - ax, by - ay);
      if (seg_len > kMaxCollisionCheckStepM) {
        const size_t sub_steps =
          static_cast<size_t>(std::ceil(seg_len / kMaxCollisionCheckStepM));
        const double dth = angles::shortest_angular_distance(ath, bth);
        for (size_t k = 1u; k < sub_steps; ++k) {
          const double alpha = static_cast<double>(k) / static_cast<double>(sub_steps);
          const double mx = ax + alpha * (bx - ax);
          const double my = ay + alpha * (by - ay);
          const double mth = angles::normalize_angle(ath + alpha * dth);
          if (!isFootprintCollisionFree(mx, my, mth)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool TerminalApproachCritic::buildReferencePathCurve(
  const models::Path & path,
  size_t start_idx,
  size_t goal_idx,
  const geometry_msgs::msg::Pose & goal_pose,
  double gear,
  std::vector<utils::Pose2D> & out_curve,
  double min_lead_distance) const
{
  out_curve.clear();
  const size_t n = path.x.shape(0);
  if (n == 0u || start_idx >= n) {
    return false;
  }
  const size_t last_idx = std::min(goal_idx, n - 1u);
  if (start_idx > last_idx) {
    return false;
  }

  const double goal_x = goal_pose.position.x;
  const double goal_y = goal_pose.position.y;
  const double goal_yaw = tf2::getYaw(goal_pose.orientation);
  if (!isFootprintCollisionFree(goal_x, goal_y, goal_yaw)) {
    return false;
  }

  const double min_res = (costmap_ != nullptr) ?
    std::max(0.01, static_cast<double>(costmap_->getResolution())) : 0.01;
  const double spacing = std::max(min_res, static_cast<double>(sample_spacing_));

  std::vector<utils::Pose2D> built;
  built.reserve((last_idx - start_idx + 2u) * 3u);

  for (size_t i = start_idx; i <= last_idx; ++i) {
    const double ax = static_cast<double>(path.x(i));
    const double ay = static_cast<double>(path.y(i));
    const double ath = static_cast<double>(path.yaws(i));
    built.push_back(
      utils::Pose2D{
        static_cast<float>(ax),
        static_cast<float>(ay),
        static_cast<float>(ath)});
    if (i < last_idx) {
      const double bx = static_cast<double>(path.x(i + 1u));
      const double by = static_cast<double>(path.y(i + 1u));
      const double bth = static_cast<double>(path.yaws(i + 1u));
      const double seg_len = std::hypot(bx - ax, by - ay);
      if (!finiteValue(seg_len)) {
        return false;
      }
      if (seg_len > spacing) {
        const size_t sub_steps = static_cast<size_t>(std::ceil(seg_len / spacing));
        const double dth = angles::shortest_angular_distance(ath, bth);
        for (size_t k = 1u; k < sub_steps; ++k) {
          const double alpha = static_cast<double>(k) / static_cast<double>(sub_steps);
          const double mx = ax + alpha * (bx - ax);
          const double my = ay + alpha * (by - ay);
          const double mth = angles::normalize_angle(ath + alpha * dth);
          built.push_back(
            utils::Pose2D{
              static_cast<float>(mx),
              static_cast<float>(my),
              static_cast<float>(mth)});
        }
      }
    }
  }

  const utils::Pose2D exact_goal{
    static_cast<float>(goal_x),
    static_cast<float>(goal_y),
    static_cast<float>(goal_yaw)};
  if (built.empty()) {
    built.push_back(exact_goal);
  } else {
    const double end_d = std::hypot(
      static_cast<double>(built.back().x) - goal_x,
      static_cast<double>(built.back().y) - goal_y);
    if (end_d <= 1e-3) {
      built.back() = exact_goal;
    } else {
      const double ax = static_cast<double>(built.back().x);
      const double ay = static_cast<double>(built.back().y);
      const double ath = static_cast<double>(built.back().theta);
      const double dth = angles::shortest_angular_distance(ath, goal_yaw);
      const size_t sub_steps =
        std::max<size_t>(1u, static_cast<size_t>(std::ceil(end_d / kMaxCollisionCheckStepM)));
      for (size_t k = 1u; k <= sub_steps; ++k) {
        const double alpha = static_cast<double>(k) / static_cast<double>(sub_steps);
        const double mx = ax + alpha * (goal_x - ax);
        const double my = ay + alpha * (goal_y - ay);
        const double mth = angles::normalize_angle(ath + alpha * dth);
        if (!isFootprintCollisionFree(mx, my, mth)) {
          return false;
        }
      }
      built.push_back(exact_goal);
    }
  }

  const double travel_yaw =
    gear < 0.0 ? angles::normalize_angle(goal_yaw + M_PI) : goal_yaw;
  const double ux = std::cos(travel_yaw);
  const double uy = std::sin(travel_yaw);

  double initial_span = 0.0;
  bool collinear_stub = !built.empty();
  double prev_d_back = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < built.size(); ++i) {
    const double dx = static_cast<double>(built[i].x) - goal_x;
    const double dy = static_cast<double>(built[i].y) - goal_y;
    const double d_back = -(dx * ux + dy * uy);
    const double lat_err = std::fabs(-dx * uy + dy * ux);
    const double yaw_err = std::fabs(
      angles::shortest_angular_distance(static_cast<double>(built[i].theta), goal_yaw));
    if (lat_err > kCollinearStubLatTolM || d_back < -kCollinearStubLatTolM ||
      yaw_err > kCollinearStubYawTolRad || d_back > prev_d_back + kCollinearStubLatTolM)
    {
      collinear_stub = false;
      break;
    }
    if (i == 0u) {
      initial_span = std::max(0.0, d_back);
    }
    prev_d_back = d_back;
  }

  if (collinear_stub &&
    (built.size() < 2u || initial_span + kCollinearStubSpanMarginM < min_lead_distance))
  {
    const double max_lead =
      std::max(
        spacing,
        std::max(kInitialSegmentRayExtrapM,
          static_cast<double>(approach_radius_) + kLatchRadiusHysteresisM));
    const double lead_len = std::clamp(min_lead_distance, spacing, max_lead);
    if (lead_len > initial_span + kCollinearStubLatTolM) {
      const size_t sample_steps =
        std::max<size_t>(1u, static_cast<size_t>(std::ceil(lead_len / spacing)));
      std::vector<utils::Pose2D> lead_pts;
      lead_pts.reserve(sample_steps + built.size());
      for (size_t k = sample_steps; k >= 1u; --k) {
        const double d_back =
          (lead_len * static_cast<double>(k)) / static_cast<double>(sample_steps);
        if (built.size() >= 2u && d_back <= initial_span + 0.5 * spacing) {
          continue;
        }
        if (d_back <= initial_span + kCollinearStubLatTolM) {
          continue;
        }
        const double lx = goal_x - d_back * ux;
        const double ly = goal_y - d_back * uy;
        lead_pts.push_back(
          utils::Pose2D{
            static_cast<float>(lx),
            static_cast<float>(ly),
            static_cast<float>(goal_yaw)});
      }
      if (!lead_pts.empty()) {
        lead_pts.insert(lead_pts.end(), built.begin(), built.end());
        for (size_t i = 0; i < lead_pts.size(); ++i) {
          const double ax = static_cast<double>(lead_pts[i].x);
          const double ay = static_cast<double>(lead_pts[i].y);
          const double ath = static_cast<double>(lead_pts[i].theta);
          if (!isFootprintCollisionFree(ax, ay, ath)) {
            return false;
          }
          if (i + 1u < lead_pts.size()) {
            const double bx = static_cast<double>(lead_pts[i + 1u].x);
            const double by = static_cast<double>(lead_pts[i + 1u].y);
            const double bth = static_cast<double>(lead_pts[i + 1u].theta);
            const double seg_len = std::hypot(bx - ax, by - ay);
            if (seg_len > kMaxCollisionCheckStepM) {
              const size_t sub_steps =
                static_cast<size_t>(std::ceil(seg_len / kMaxCollisionCheckStepM));
              const double dth = angles::shortest_angular_distance(ath, bth);
              for (size_t k = 1u; k < sub_steps; ++k) {
                const double alpha = static_cast<double>(k) / static_cast<double>(sub_steps);
                const double mx = ax + alpha * (bx - ax);
                const double my = ay + alpha * (by - ay);
                const double mth = angles::normalize_angle(ath + alpha * dth);
                if (!isFootprintCollisionFree(mx, my, mth)) {
                  return false;
                }
              }
            }
          }
        }
        built = std::move(lead_pts);
      }
    }
  }

  out_curve = std::move(built);
  return true;
}

double TerminalApproachCritic::minDistanceToCurve(
  double rx, double ry,
  const std::vector<utils::Pose2D> & curve) const
{
  if (curve.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (curve.size() == 1u) {
    return std::hypot(rx - curve.front().x, ry - curve.front().y);
  }
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i + 1 < curve.size(); ++i) {
    const double ax = curve[i].x;
    const double ay = curve[i].y;
    const double bx = curve[i + 1].x;
    const double by = curve[i + 1].y;
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len_sq = dx * dx + dy * dy;
    double t = 0.0;
    if (len_sq > 1e-10) {
      t = std::clamp(((rx - ax) * dx + (ry - ay) * dy) / len_sq, 0.0, 1.0);
    }
    const double px = ax + t * dx;
    const double py = ay + t * dy;
    best = std::min(best, std::hypot(rx - px, ry - py));
  }
  return best;
}

double TerminalApproachCritic::minDistanceToPath(
  double rx, double ry,
  const models::Path & path) const
{
  const size_t n = path.x.shape(0);
  if (n == 0u) {
    return std::numeric_limits<double>::infinity();
  }
  const double max_ray =
    std::max(kInitialSegmentRayExtrapM,
      static_cast<double>(approach_radius_) + kLatchRadiusHysteresisM);
  if (n == 1u) {
    const double gx = static_cast<double>(path.x(0));
    const double gy = static_cast<double>(path.y(0));
    const double gth = static_cast<double>(path.yaws(0));
    const double ux = std::cos(gth);
    const double uy = std::sin(gth);
    const double s = std::clamp((rx - gx) * ux + (ry - gy) * uy, -max_ray, max_ray);
    return std::hypot(rx - (gx + s * ux), ry - (gy + s * uy));
  }
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i + 1 < n; ++i) {
    const double ax = path.x(i);
    const double ay = path.y(i);
    const double bx = path.x(i + 1);
    const double by = path.y(i + 1);
    const double dx = bx - ax;
    const double dy = by - ay;
    const double len_sq = dx * dx + dy * dy;
    double t = 0.0;
    if (len_sq > 1e-10) {
      const double min_t = (i == 0u) ? (-max_ray / std::sqrt(len_sq)) : 0.0;
      t = std::clamp(((rx - ax) * dx + (ry - ay) * dy) / len_sq, min_t, 1.0);
    }
    const double px = ax + t * dx;
    const double py = ay + t * dy;
    best = std::min(best, std::hypot(rx - px, ry - py));
  }
  return best;
}

std::vector<CaptureFamilyTrace> TerminalApproachCritic::traceCaptureCandidates(
  const geometry_msgs::msg::Pose & robot_pose,
  const geometry_msgs::msg::Pose & goal_pose,
  double gear,
  double goal_xy_tol,
  double goal_yaw_tol) const
{
  std::vector<utils::Pose2D> unused_curve;
  std::vector<CaptureFamilyTrace> traces;
  solveCaptureCurve(
    robot_pose, goal_pose, gear, unused_curve, goal_xy_tol, goal_yaw_tol, &traces);
  return traces;
}

bool TerminalApproachCritic::solveCaptureCurve(
  const geometry_msgs::msg::Pose & robot_pose,
  const geometry_msgs::msg::Pose & goal_pose,
  double gear,
  std::vector<utils::Pose2D> & out_curve,
  double goal_xy_tol,
  double goal_yaw_tol,
  std::vector<CaptureFamilyTrace> * out_traces,
  std::string * out_family,
  double * out_radius) const
{
  out_curve.clear();
  if (out_traces != nullptr) {
    out_traces->clear();
  }
  if (out_family != nullptr) {
    out_family->clear();
  }
  if (out_radius != nullptr) {
    *out_radius = 0.0;
  }

  const double min_turning_r = std::max(0.10, static_cast<double>(min_turning_radius_));
  const double max_res_y = std::max(0.0, static_cast<double>(max_residual_y_));
  const double max_res_yaw = std::max(0.0, static_cast<double>(max_residual_yaw_));
  const double eff_xy_tol =
    (goal_xy_tol > 0.0) ? goal_xy_tol : std::max(0.01, static_cast<double>(goal_xy_tolerance_));
  const double eff_yaw_tol =
    (goal_yaw_tol > 0.0) ? goal_yaw_tol : std::max(0.01, static_cast<double>(goal_yaw_tolerance_));
  const double max_end_xy = std::min(max_res_y, kGoalBoxXyTargetRatio * eff_xy_tol);
  const double max_end_yaw = std::min(max_res_yaw, kGoalBoxYawTargetRatio * eff_yaw_tol);

  const double goal_nose_yaw = tf2::getYaw(goal_pose.orientation);
  const double robot_nose_yaw = tf2::getYaw(robot_pose.orientation);
  const double goal_travel_yaw =
    gear < 0.0 ? angles::normalize_angle(goal_nose_yaw + M_PI) : goal_nose_yaw;
  const double robot_travel_yaw =
    gear < 0.0 ? angles::normalize_angle(robot_nose_yaw + M_PI) : robot_nose_yaw;

  const double dx = robot_pose.position.x - goal_pose.position.x;
  const double dy = robot_pose.position.y - goal_pose.position.y;
  const double cos_g = std::cos(goal_travel_yaw);
  const double sin_g = std::sin(goal_travel_yaw);

  // Transform robot pose into Goal Travel Frame (goal at (0, 0), travel heading 0)
  const double X0 = dx * cos_g + dy * sin_g;
  const double Y0 = -dx * sin_g + dy * cos_g;
  const double theta0 = angles::shortest_angular_distance(goal_travel_yaw, robot_travel_yaw);
  const double chord_dist = std::hypot(X0, Y0);

  // Require the robot to be longitudinally behind the goal in its travel direction
  // and facing within +/- 85 deg of the goal travel heading.
  if (X0 >= kMinBehindGoalLonM || std::fabs(theta0) >= kMaxCaptureAngleRad) {
    if (out_traces != nullptr) {
      out_traces->push_back(
        {"pose_admission", false, 0.0, 0.0, 0.0, 0.0, Y0, theta0,
          (X0 >= kMinBehindGoalLonM) ? "X0 >= kMinBehindGoalLonM" : "|theta0| >= 85deg"});
    }
    return false;
  }

  struct CandidateCurve
  {
    std::string family;
    double R{0.0};
    double s{1.0};
    double theta_m{0.0};
    double L1{0.0};
    double Ls{0.0};
    double L2{0.0};
    double x_offset{0.0};
    double y_offset{0.0};
    double theta_end{0.0};
  };

  std::vector<CandidateCurve> candidates;
  const double A = 2.0 * (std::cos(theta0) - 1.0);
  const double C = X0 * X0 + Y0 * Y0;

  // Family 1a: Exact C / CC bi-arc curves (Ls = 0, x_offset = 0, y_offset = 0, R >= min_turning_r)
  for (const double s : {1.0, -1.0}) {
    const std::string fam_name = (s > 0.0) ? "1a_biarc_s=+1" : "1a_biarc_s=-1";
    const double B = 2.0 * s * (-X0 * std::sin(theta0) + Y0 * (1.0 + std::cos(theta0)));
    std::vector<double> roots;
    if (std::fabs(A) < 1e-9) {
      if (std::fabs(B) > 1e-9) {
        roots.push_back(-C / B);
      }
    } else {
      const double disc = B * B - 4.0 * A * C;
      if (disc >= 0.0) {
        const double sqrt_disc = std::sqrt(disc);
        roots.push_back((-B + sqrt_disc) / (2.0 * A));
        roots.push_back((-B - sqrt_disc) / (2.0 * A));
      }
    }

    bool accepted_any = false;
    double best_pos_r = -1.0;
    std::string reject_reason = "no_real_root";
    for (const double R : roots) {
      if (finiteValue(R) && R > best_pos_r) {
        best_pos_r = R;
      }
      if (!finiteValue(R) || R < min_turning_r) {
        if (finiteValue(R) && R > 0.0) {
          reject_reason = "R < min_turning_r";
        } else if (reject_reason == "no_real_root") {
          reject_reason = "R <= 0";
        }
        continue;
      }
      const double sin_num = R * std::sin(theta0) - s * X0;
      const double cos_num = s * Y0 + R * (1.0 + std::cos(theta0));
      const double theta_m = std::atan2(sin_num, cos_num);
      if (std::fabs(theta_m) >= kMaxCaptureAngleRad) {
        reject_reason = "|theta_m| >= 85deg";
        continue;
      }
      const double xm = -s * R * std::sin(theta_m);
      if (xm < X0 - 1e-3 || xm > 1e-3) {
        reject_reason = "xm_out_of_ longitudinal_bounds";
        continue;
      }
      const double raw_L1 = R * s * (theta_m - theta0);
      const double raw_L2 = R * s * theta_m;
      if (raw_L1 < -1e-4 || raw_L2 < -1e-4) {
        reject_reason = "negative_arc_length (raw_L1 < 0 or raw_L2 < 0)";
        continue;
      }
      const double R1 = s * R;
      const double R2 = -s * R;
      const double xm1 = X0 + R1 * (std::sin(theta_m) - std::sin(theta0));
      const double ym1 = Y0 - R1 * (std::cos(theta_m) - std::cos(theta0));
      const double xm2 = R2 * std::sin(theta_m);
      const double ym2 = R2 * (1.0 - std::cos(theta_m));
      if (std::hypot(xm1 - xm2, ym1 - ym2) > 1e-4) {
        reject_reason = "midpoint_discontinuity";
        continue;
      }
      const double L1 = std::max(0.0, raw_L1);
      const double L2 = std::max(0.0, raw_L2);
      const double total_len = L1 + L2;
      if (total_len < kMinCaptureCurveLenM || total_len > kMaxCurveChordRatio * chord_dist) {
        reject_reason = "curve_length_out_of_bounds";
        continue;
      }
      candidates.push_back({fam_name, R, s, theta_m, L1, 0.0, L2, 0.0, 0.0, 0.0});
      accepted_any = true;
      if (out_traces != nullptr) {
        out_traces->push_back({fam_name, true, R, L1, 0.0, L2, 0.0, 0.0, "accepted_geometric"});
      }
    }
    if (!accepted_any && out_traces != nullptr) {
      out_traces->push_back(
        {fam_name, false, std::max(0.0, best_pos_r), 0.0, 0.0, 0.0, 0.0, 0.0, reject_reason});
    }
  }

  // Family 1b: Exact SC (straight-then-arc) curves when heading is angled toward goal line
  {
    const std::string fam_name = "1b_sc_straight_arc";
    if (std::fabs(theta0) > 1e-3 && std::cos(theta0) > kMinForwardCosine) {
      const double s = (theta0 >= 0.0) ? 1.0 : -1.0;
      const double denom = s * (1.0 - std::cos(theta0));
      if (std::fabs(denom) > 1e-7) {
        const double R = (Y0 * std::cos(theta0) - X0 * std::sin(theta0)) / denom;
        const double Ls = (-X0 - s * R * std::sin(theta0)) / std::cos(theta0);
        const double L2 = R * std::fabs(theta0);
        const double total_len = Ls + L2;
        const bool len_oob =
          (total_len < kMinCaptureCurveLenM) || (total_len > kMaxCurveChordRatio * chord_dist);
        if (!finiteValue(R) || R < min_turning_r) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, 0.0, Ls, L2, 0.0, 0.0, "R < min_turning_r"});
          }
        } else if (Ls < 0.0 || L2 < 0.0) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, 0.0, Ls, L2, 0.0, 0.0, "Ls < 0"});
          }
        } else if (len_oob) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, 0.0, Ls, L2, 0.0, 0.0, "curve_length_out_of_bounds"});
          }
        } else {
          candidates.push_back({fam_name, R, s, theta0, 0.0, Ls, L2, 0.0, 0.0, 0.0});
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, true, R, 0.0, Ls, L2, 0.0, 0.0, "accepted_geometric"});
          }
        }
      }
    } else if (out_traces != nullptr) {
      out_traces->push_back(
        {fam_name, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "|theta0| <= 1e-3 or cos(theta0) <= 0.1"});
    }
  }

  // Family 1c: Exact CS (arc-then-straight) curves when turning immediately onto goal line
  {
    const std::string fam_name = "1c_cs_arc_straight";
    if (std::fabs(theta0) > 1e-3 && std::cos(theta0) > kMinForwardCosine) {
      const double s = (theta0 >= 0.0) ? -1.0 : 1.0;
      const double denom = s * (1.0 - std::cos(theta0));
      if (std::fabs(denom) > 1e-7) {
        const double R = Y0 / denom;
        const double L1 = R * std::fabs(theta0);
        const double Ls = -X0 + s * R * std::sin(theta0);
        const double total_len = L1 + Ls;
        const bool len_oob =
          (total_len < kMinCaptureCurveLenM) || (total_len > kMaxCurveChordRatio * chord_dist);
        if (!finiteValue(R) || R < min_turning_r) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, Ls, 0.0, 0.0, 0.0, "R < min_turning_r"});
          }
        } else if (L1 < 0.0 || Ls < 0.0) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, Ls, 0.0, 0.0, 0.0, "Ls < 0 (arc overshoots X = 0)"});
          }
        } else if (len_oob) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, Ls, 0.0, 0.0, 0.0, "curve_length_out_of_bounds"});
          }
        } else {
          candidates.push_back({fam_name, R, s, 0.0, L1, Ls, 0.0, 0.0, 0.0, 0.0});
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, true, R, L1, Ls, 0.0, 0.0, 0.0, "accepted_geometric"});
          }
        }
      }
    } else if (out_traces != nullptr) {
      out_traces->push_back(
        {fam_name, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "|theta0| <= 1e-3 or cos(theta0) <= 0.1"});
    }
  }

  // Family 2: Bounded-R_min & Goal-Checker Tolerance-Box fallback candidates
  // Evaluated unconditionally in both production and trace mode; std::sort below ranks
  // exact endpoint candidates (|theta_end| == 0, |y_offset| == 0, |x_offset| == 0) first.
  {
    const double R = min_turning_r;
    const double max_sin = std::sin(kMaxCaptureAngleRad);
    const double x_end = std::min(
      {kBoundedBiarcMaxXEndM, -kBoundedBiarcXEndRatio * X0, 0.5 * max_end_xy});

    // Family 2a: Bounded-R_min bi-arc to theta_end = 0, hypot(x_end, y_end) <= max_end_xy
    if (max_end_xy > 0.0) {
      for (const double s : {1.0, -1.0}) {
        const std::string fam_name =
          (s > 0.0) ? "2a_bounded_biarc_s=+1" : "2a_bounded_biarc_s=-1";
        const double sin_theta_m = (R * std::sin(theta0) + s * (x_end - X0)) / (2.0 * R);
        if (std::fabs(sin_theta_m) >= max_sin) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, 0.0, 0.0, 0.0, 0.0, 0.0, "|sin(theta_m)| >= sin(85deg)"});
          }
          continue;
        }
        const double theta_m = std::asin(sin_theta_m);
        const double xm = X0 + s * R * (std::sin(theta_m) - std::sin(theta0));
        const double raw_L1 = R * s * (theta_m - theta0);
        const double raw_L2 = R * s * theta_m;
        const double y_end =
          Y0 + s * R * (1.0 + std::cos(theta0) - 2.0 * std::cos(theta_m));
        if (xm < X0 - 1e-3 || xm > x_end + 1e-3) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, raw_L1, 0.0, raw_L2, y_end, 0.0,
                "xm_out_of_longitudinal_bounds"});
          }
          continue;
        }
        if (raw_L1 < -1e-4 || raw_L2 < -1e-4) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, raw_L1, 0.0, raw_L2, y_end, 0.0,
                "negative_arc_length (raw_L1 < 0 or raw_L2 < 0)"});
          }
          continue;
        }
        if (std::hypot(x_end, y_end) > max_end_xy) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, raw_L1, 0.0, raw_L2, y_end, 0.0,
                "hypot(x_end, y_end) > max_end_xy"});
          }
          continue;
        }
        const double L1 = std::max(0.0, raw_L1);
        const double L2 = std::max(0.0, raw_L2);
        const double total_len = L1 + L2;
        if (total_len < kMinCaptureCurveLenM || total_len > kMaxCurveChordRatio * chord_dist) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, 0.0, L2, y_end, 0.0, "curve_length_out_of_bounds"});
          }
          continue;
        }
        candidates.push_back({fam_name, R, s, theta_m, L1, 0.0, L2, x_end, y_end, 0.0});
        if (out_traces != nullptr) {
          out_traces->push_back(
            {fam_name, true, R, L1, 0.0, L2, y_end, 0.0, "accepted_geometric"});
        }
      }

      // Family 2b: Bounded-R_min CS (arc at R_min to theta_end = 0, then straight to X = 0)
      if (std::fabs(theta0) > 1e-3 && std::cos(theta0) > kMinForwardCosine) {
        const std::string fam_name = "2b_bounded_cs";
        const double s = (theta0 >= 0.0) ? -1.0 : 1.0;
        const double L1 = R * std::fabs(theta0);
        const double Ls = -X0 + s * R * std::sin(theta0);
        const double y_end = Y0 - s * R * (1.0 - std::cos(theta0));
        const double total_len = L1 + Ls;
        const bool len_ok =
          total_len >= kMinCaptureCurveLenM && total_len <= kMaxCurveChordRatio * chord_dist;
        if (Ls < 0.0) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, Ls, 0.0, y_end, 0.0, "Ls < 0 (arc overshoots X = 0)"});
          }
        } else if (std::fabs(y_end) > max_end_xy) {
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, false, R, L1, Ls, 0.0, y_end, 0.0, "|y_end| > max_end_xy"});
          }
        } else if (len_ok) {
          candidates.push_back({fam_name, R, s, 0.0, L1, Ls, 0.0, 0.0, y_end, 0.0});
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, true, R, L1, Ls, 0.0, y_end, 0.0, "accepted_geometric"});
          }
        }
      }
    }

    // Family 2c: Short-Range Goal-Checker Tolerance-Box single-arc / straight arrival
    // (active when chord_dist < short_approach_distance_)
    if (chord_dist < static_cast<double>(short_approach_distance_) && max_end_xy > 0.0) {
      const double box_xy_limit = max_end_xy;
      const double box_yaw_limit = max_end_yaw;

      // Straight-line arrival at X = 0 when heading is already inside box_yaw_limit
      if (std::cos(theta0) > kMinForwardCosine && std::fabs(theta0) <= box_yaw_limit) {
        const double Ls = -X0 / std::cos(theta0);
        const double y_end = Y0 + Ls * std::sin(theta0);
        if (Ls >= kMinCaptureCurveLenM && std::fabs(y_end) <= box_xy_limit) {
          candidates.push_back(
            {"2c_goal_box_straight", 1e6, 1.0, theta0, 0.0, Ls, 0.0, 0.0, y_end, theta0});
          if (out_traces != nullptr) {
            out_traces->push_back(
              {"2c_goal_box_straight", true, 1e6, 0.0, Ls, 0.0, y_end, theta0,
                "accepted_geometric"});
          }
        }
      }

      // Single-arc arrival at R = R_min reaching X = 0 inside (box_xy_limit, box_yaw_limit)
      for (const double s : {1.0, -1.0}) {
        const std::string fam_name =
          (s > 0.0) ? "2c_goal_box_arc_s=+1" : "2c_goal_box_arc_s=-1";
        const double R1 = s * R;
        const double sin_theta_end = std::sin(theta0) - X0 / R1;
        if (std::fabs(sin_theta_end) >= max_sin) {
          continue;
        }
        const double theta_end = std::asin(sin_theta_end);
        const double raw_L1 = R1 * (theta_end - theta0);
        const double y_end = Y0 - R1 * (std::cos(theta_end) - std::cos(theta0));
        if (raw_L1 < kMinCaptureCurveLenM || raw_L1 > kMaxCurveChordRatio * chord_dist) {
          continue;
        }
        if (std::fabs(theta_end) <= box_yaw_limit && std::fabs(y_end) <= box_xy_limit) {
          candidates.push_back(
            {fam_name, R, s, theta_end, raw_L1, 0.0, 0.0, 0.0, y_end, theta_end});
          if (out_traces != nullptr) {
            out_traces->push_back(
              {fam_name, true, R, raw_L1, 0.0, 0.0, y_end, theta_end, "accepted_geometric"});
          }
        }
      }
    }
  }

  if (candidates.empty()) {
    return false;
  }

  // Prefer exact endpoint curves (x_offset == 0, y_offset == 0, theta_end == 0), then smallest
  // |theta_end|, then smallest |y_offset|, then gentlest curvature (largest R).
  std::sort(
    candidates.begin(), candidates.end(),
    [](const CandidateCurve & a, const CandidateCurve & b) {
      const bool a_exact =
      (std::fabs(a.x_offset) <= 1e-6 &&
      std::fabs(a.y_offset) <= 1e-6 &&
      std::fabs(a.theta_end) <= 1e-6);
      const bool b_exact =
      (std::fabs(b.x_offset) <= 1e-6 &&
      std::fabs(b.y_offset) <= 1e-6 &&
      std::fabs(b.theta_end) <= 1e-6);
      if (a_exact != b_exact) {
        return a_exact;
      }
      if (std::fabs(std::fabs(a.theta_end) - std::fabs(b.theta_end)) > 1e-6) {
        return std::fabs(a.theta_end) < std::fabs(b.theta_end);
      }
      if (std::fabs(std::fabs(a.y_offset) - std::fabs(b.y_offset)) > 1e-6) {
        return std::fabs(a.y_offset) < std::fabs(b.y_offset);
      }
      return a.R > b.R;
    });

  auto evalGoalFramePose = [&](
    const CandidateCurve & cand, double s_arc,
    double & xg, double & yg, double & th_travel_g)
    {
      const double R1 = cand.s * cand.R;
      const double R2 = -cand.s * cand.R;
      const double x1 =
        (cand.L1 > 0.0) ? (X0 + R1 * (std::sin(cand.theta_m) - std::sin(theta0))) : X0;
      const double y1 =
        (cand.L1 > 0.0) ? (Y0 - R1 * (std::cos(cand.theta_m) - std::cos(theta0))) : Y0;
      const double x2 = x1 + cand.Ls * std::cos(cand.theta_m);
      const double y2 = y1 + cand.Ls * std::sin(cand.theta_m);

      if (s_arc <= cand.L1) {
        th_travel_g = theta0 + s_arc / R1;
        xg = X0 + R1 * (std::sin(th_travel_g) - std::sin(theta0));
        yg = Y0 - R1 * (std::cos(th_travel_g) - std::cos(theta0));
      } else if (s_arc <= cand.L1 + cand.Ls) {
        const double ss = s_arc - cand.L1;
        th_travel_g = cand.theta_m;
        xg = x1 + ss * std::cos(cand.theta_m);
        yg = y1 + ss * std::sin(cand.theta_m);
      } else {
        const double s2 = std::min(s_arc - cand.L1 - cand.Ls, cand.L2);
        th_travel_g = cand.theta_m + s2 / R2;
        xg = x2 + R2 * (std::sin(th_travel_g) - std::sin(cand.theta_m));
        yg = y2 - R2 * (std::cos(th_travel_g) - std::cos(cand.theta_m));
      }
    };

  auto toWorldPose2D = [&](double xg, double yg, double th_travel_g) -> utils::Pose2D
    {
      const double wx = goal_pose.position.x + (xg * cos_g - yg * sin_g);
      const double wy = goal_pose.position.y + (xg * sin_g + yg * cos_g);
      const double world_travel_yaw = angles::normalize_angle(goal_travel_yaw + th_travel_g);
      const double world_nose_yaw =
        gear < 0.0 ? angles::normalize_angle(world_travel_yaw + M_PI) : world_travel_yaw;
      return utils::Pose2D{
      static_cast<float>(wx),
      static_cast<float>(wy),
      static_cast<float>(world_nose_yaw)};
    };

  const double min_res = (costmap_ != nullptr) ?
    std::max(0.01, static_cast<double>(costmap_->getResolution())) : 0.01;
  const double spacing = std::max(min_res, static_cast<double>(sample_spacing_));
  const double check_spacing = std::min(kMaxCollisionCheckStepM, spacing);

  for (const auto & cand : candidates) {
    const double total_len = cand.L1 + cand.Ls + cand.L2;
    const size_t num_checks = std::max<size_t>(
      2u, static_cast<size_t>(std::ceil(total_len / check_spacing)));
    bool collision_free = true;
    for (size_t i = 0; i <= num_checks; ++i) {
      const double s_arc = (total_len * static_cast<double>(i)) / static_cast<double>(num_checks);
      double xg = 0.0, yg = 0.0, th_g = 0.0;
      evalGoalFramePose(cand, s_arc, xg, yg, th_g);
      const auto p = toWorldPose2D(xg, yg, th_g);
      if (!isFootprintCollisionFree(p.x, p.y, p.theta)) {
        collision_free = false;
        break;
      }
    }
    if (!collision_free) {
      continue;
    }

    const size_t num_pts = std::max<size_t>(
      2u, static_cast<size_t>(std::ceil(total_len / spacing)));
    std::vector<utils::Pose2D> built;
    built.reserve(num_pts + 1u);
    for (size_t i = 0; i < num_pts; ++i) {
      const double s_arc = (total_len * static_cast<double>(i)) / static_cast<double>(num_pts);
      double xg = 0.0, yg = 0.0, th_g = 0.0;
      evalGoalFramePose(cand, s_arc, xg, yg, th_g);
      built.push_back(toWorldPose2D(xg, yg, th_g));
    }
    double end_xg = 0.0, end_yg = 0.0, end_th_g = 0.0;
    evalGoalFramePose(cand, total_len, end_xg, end_yg, end_th_g);
    const utils::Pose2D end_pose =
      (std::fabs(cand.y_offset) <= 1e-6 &&
      std::fabs(cand.x_offset) <= 1e-6 &&
      std::fabs(cand.theta_end) <= 1e-6) ?
      utils::Pose2D{
      static_cast<float>(goal_pose.position.x),
      static_cast<float>(goal_pose.position.y),
      static_cast<float>(goal_nose_yaw)} :
    toWorldPose2D(end_xg, end_yg, end_th_g);
    const double end_dx = static_cast<double>(end_pose.x) - goal_pose.position.x;
    const double end_dy = static_cast<double>(end_pose.y) - goal_pose.position.y;
    const double end_dyaw = std::fabs(
      angles::shortest_angular_distance(static_cast<double>(end_pose.theta), goal_nose_yaw));
    if (std::hypot(end_dx, end_dy) > max_end_xy + 1e-5 || end_dyaw > max_end_yaw + 1e-5) {
      continue;
    }
    if (!isFootprintCollisionFree(end_pose.x, end_pose.y, end_pose.theta)) {
      continue;
    }
    built.push_back(end_pose);
    out_curve = std::move(built);
    if (out_family != nullptr) {
      *out_family = cand.family;
    }
    if (out_radius != nullptr) {
      *out_radius = cand.R;
    }
    return true;
  }

  return false;
}

void TerminalApproachCritic::handleInfeasible(
  const char * reason,
  double cross_track_m,
  double goal_dist_m,
  double yaw_err_rad)
{
  char msg_buf[256];
  std::snprintf(
    msg_buf, sizeof(msg_buf),
    "[TERMINAL_APPROACH_ABORT] reason=%s cross_track=%.4fm goal_dist=%.4fm yaw_err=%.4frad "
    "redraws=%d/%d replans=%d",
    reason, cross_track_m, goal_dist_m, yaw_err_rad, redraw_count_, max_redraws_, replan_count_);
  last_abort_msg_ = msg_buf;

  clearActiveCurve();
  approach_escalated_ = true;

  RCLCPP_INFO(
    logger_,
    "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=%s goal_dist=%.4f cross_track=%.4f "
    "yaw_err=%.4f redraws=%d/%d replans=%d",
    infeasible_policy_.c_str(), reason, goal_dist_m, cross_track_m, yaw_err_rad,
    redraw_count_, max_redraws_, replan_count_);

  const bool inside_cutoff =
    (cutoff_radius_ > 0.0f) && (goal_dist_m <= static_cast<double>(cutoff_radius_));

  if (infeasible_policy_ == "abort") {
    throw nav2_core::InvalidPath(last_abort_msg_);
  }

  if (infeasible_policy_ == "replan_signal") {
    if (!replan_allowance_consumed_ && !inside_cutoff) {
      requestReplanIfEligible(goal_dist_m);
      RCLCPP_WARN(logger_, "%s (policy=replan_signal, replan_pending=true)", msg_buf);
      return;
    }
    replan_allowance_consumed_ = true;
    replan_pending_ = false;
    RCLCPP_WARN(
      logger_,
      "%s (policy=replan_signal, attempt_failed: allowance_consumed_or_inside_cutoff)",
      msg_buf);
    throw nav2_core::InvalidPath(last_abort_msg_);
  }

  if (inside_cutoff) {
    RCLCPP_WARN(
      logger_,
      "%s (policy=fallback, attempt_failed: inside_cutoff_stock_critics_off)",
      msg_buf);
    throw nav2_core::InvalidPath(last_abort_msg_);
  }

  RCLCPP_WARN(logger_, "%s (policy=fallback)", msg_buf);
}

void TerminalApproachCritic::requestReplanIfEligible(double goal_dist_m)
{
  if (infeasible_policy_ != "replan_signal" || replan_allowance_consumed_) {
    return;
  }
  const bool inside_cutoff =
    (goal_dist_m > 0.0) && (cutoff_radius_ > 0.0f) &&
    (goal_dist_m <= static_cast<double>(cutoff_radius_));
  replan_allowance_consumed_ = true;
  if (!inside_cutoff) {
    replan_pending_ = true;
    ++replan_count_;
    ensureReplanPublisher(true);
    if (!replan_signal_sent_ && replan_pub_ != nullptr && replan_pub_->is_activated()) {
      std_msgs::msg::Bool sig;
      sig.data = true;
      replan_pub_->publish(sig);
      replan_signal_sent_ = true;
      pending_replan_clear_ = false;
    }
  } else {
    replan_pending_ = false;
  }
}

void TerminalApproachCritic::score(CriticData & data)
{
  if (hasParameterChangedSinceLatch()) {
    if (latch_consumed_ || latched_ || approach_escalated_ ||
      replan_pending_ || replan_allowance_consumed_)
    {
      if (replan_signal_sent_) {
        if (is_active_ && replan_pub_ != nullptr && replan_pub_->is_activated()) {
          std_msgs::msg::Bool sig;
          sig.data = false;
          replan_pub_->publish(sig);
          pending_replan_clear_ = false;
        } else {
          pending_replan_clear_ = true;
        }
        replan_signal_sent_ = false;
      }
      replan_pending_ = false;
      const bool had_active_latch = latched_ && !approach_escalated_;
      clearActiveCurve();
      if (replan_allowance_consumed_ || max_redraws_ == 0 ||
        redraw_count_ >= max_redraws_ || !had_active_latch)
      {
        const char * r = replan_allowance_consumed_ ?
          "PARAM_CHANGED_AFTER_REPLAN_CONSUMED" : "PARAM_CHANGE_WHILE_LATCHED";
        approach_escalated_ = true;
        if (replan_allowance_consumed_ || escalation_reason_.empty()) {
          escalation_reason_ = r;
        }
        if (replan_allowance_consumed_ && last_abort_msg_.empty()) {
          last_abort_msg_ = std::string("[TERMINAL_APPROACH_ABORT] reason=") + r;
        }
        if (replan_allowance_consumed_ || had_active_latch) {
          RCLCPP_INFO(
            logger_, "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=%s "
            "goal_dist=0.0000 cross_track=0.0000 yaw_err=0.0000 redraws=%d/%d replans=%d",
            infeasible_policy_.c_str(), r, redraw_count_, max_redraws_, replan_count_);
          if (had_active_latch) {
            requestReplanIfEligible(0.0);
          }
        }
      } else {
        ++redraw_count_;
      }
      tick_evaluated_ = false;
      snapshotLatchParameters();
    } else {
      resetLatch();
    }
  }

  if (!enabled_) {
    if (!latch_consumed_ && !latched_ && !approach_escalated_ &&
      !replan_pending_ && !replan_allowance_consumed_)
    {
      resetLatch();
    }
    return;
  }
  if (!stock_thresholds_checked_) {
    stock_thresholds_checked_ = true;
    syncStockCriticThresholds(true);
  }
  if (data.path.x.shape(0) == 0u) {
    return;
  }

  const size_t last_path_idx = data.path.x.shape(0) - 1u;
  const double goal_x = data.goal.position.x, goal_y = data.goal.position.y;
  const double goal_yaw = tf2::getYaw(data.goal.orientation);
  const double robot_x = data.state.pose.pose.position.x, robot_y = data.state.pose.pose.position.y;
  const double robot_yaw = tf2::getYaw(data.state.pose.pose.orientation);
  const double goal_dist = std::hypot(robot_x - goal_x, robot_y - goal_y);
  if (!data.reuse_obstacles_cache || !tick_evaluated_) {
    local_goal_shift_caused_drift_ = false;
  }

  // Small odom-frame localization updates (<= goal_reset_pos_tol_ / yaw_tol cumulative from
  // latch_origin_goal_) rigidly transform latched_curve_. Larger local-frame shifts without a
  // new global goal in onNewPlan() never replenish replan_allowance_consumed_ or latch_consumed_.
  if (!latched_frame_id_.empty() &&
    (latch_consumed_ || latched_ || approach_escalated_ ||
    replan_pending_ || replan_signal_sent_ || replan_allowance_consumed_))
  {
    const double step_dpos =
      std::hypot(goal_x - latched_goal_.position.x, goal_y - latched_goal_.position.y);
    const double step_dyaw =
      angles::shortest_angular_distance(tf2::getYaw(latched_goal_.orientation), goal_yaw);
    const double cum_dpos =
      std::hypot(goal_x - latch_origin_goal_.position.x, goal_y - latch_origin_goal_.position.y);
    const double cum_dyaw = std::fabs(
      angles::shortest_angular_distance(tf2::getYaw(latch_origin_goal_.orientation), goal_yaw));
    if (data.state.pose.header.frame_id != latched_frame_id_ ||
      cum_dpos > static_cast<double>(goal_reset_pos_tol_) ||
      cum_dyaw > static_cast<double>(goal_reset_yaw_tol_))
    {
      latched_goal_ = data.goal;
      latch_origin_goal_ = data.goal;
      latched_frame_id_ = data.state.pose.header.frame_id;
      if (!replan_pending_) {
        const bool had_active_latch = latched_ && !approach_escalated_;
        clearActiveCurve();
        latch_consumed_ = true;
        if (replan_allowance_consumed_ || max_redraws_ == 0 ||
          redraw_count_ >= max_redraws_ || !had_active_latch)
        {
          const char * r = replan_allowance_consumed_ ?
            "LOCAL_GOAL_SHIFT_AFTER_REPLAN_CONSUMED" : "LOCAL_GOAL_SHIFT_EXCEEDED";
          approach_escalated_ = true;
          escalation_reason_ = r;
          if (last_abort_msg_.empty()) {
            last_abort_msg_ = std::string("[TERMINAL_APPROACH_ABORT] reason=") + r;
          }
          RCLCPP_INFO(
            logger_, "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=%s "
            "goal_dist=%.4f cross_track=%.4f yaw_err=%.4f redraws=%d/%d replans=%d",
            infeasible_policy_.c_str(), r, goal_dist, cum_dpos, cum_dyaw,
            redraw_count_, max_redraws_, replan_count_);
          if (had_active_latch) {
            requestReplanIfEligible(goal_dist);
          }
        } else {
          ++redraw_count_;
        }
      }
    } else if (step_dpos > 1e-6 || std::fabs(step_dyaw) > 1e-6) {
      if (latched_) {
        const double pre_shift_drift = minDistanceToCurve(robot_x, robot_y, latched_curve_);
        const double cos_d = std::cos(step_dyaw), sin_d = std::sin(step_dyaw);
        const double gx0 = latched_goal_.position.x, gy0 = latched_goal_.position.y;
        for (auto & pt : latched_curve_) {
          const double rx = static_cast<double>(pt.x) - gx0, ry = static_cast<double>(pt.y) - gy0;
          pt.x = static_cast<float>(goal_x + (rx * cos_d - ry * sin_d));
          pt.y = static_cast<float>(goal_y + (rx * sin_d + ry * cos_d));
          pt.theta = static_cast<float>(
            angles::normalize_angle(static_cast<double>(pt.theta) + step_dyaw));
        }
        if (minDistanceToCurve(robot_x, robot_y, latched_curve_) > pre_shift_drift + 1e-4) {
          local_goal_shift_caused_drift_ = true;
        }
      }
      latched_goal_ = data.goal;
    }
  }

  // Review #2 Finding 3 & cusp protection: verify that the local reference path
  // actually includes the terminal goal pose before activating or extending.
  bool has_terminal_goal = false;
  size_t terminal_goal_idx = last_path_idx;
  double tail_arc = 0.0;
  for (size_t idx = last_path_idx + 1u; idx-- > 0u; ) {
    const double px = static_cast<double>(data.path.x(idx));
    const double py = static_cast<double>(data.path.y(idx));
    const double pyaw = static_cast<double>(data.path.yaws(idx));
    if (idx < last_path_idx) {
      tail_arc += std::hypot(
        static_cast<double>(data.path.x(idx + 1u)) - px,
        static_cast<double>(data.path.y(idx + 1u)) - py);
      if (tail_arc > kMaxGhostRunwayScanM) {
        break;
      }
    }
    if (std::hypot(px - goal_x, py - goal_y) <= kGoalMatchPosTolM &&
      std::fabs(angles::shortest_angular_distance(pyaw, goal_yaw)) <= kGoalMatchYawTolRad)
    {
      has_terminal_goal = true;
      terminal_goal_idx = idx;
      break;
    }
    if (std::fabs(angles::shortest_angular_distance(pyaw, goal_yaw)) > 1e-2) {
      break;
    }
  }

  if (!has_terminal_goal) {
    if (latched_) {
      clearActiveCurve();
      approach_escalated_ = true;
      requestReplanIfEligible(goal_dist);
    }
    if (approach_escalated_) {
      if (cutoff_radius_ > 0.0f && goal_dist <= static_cast<double>(cutoff_radius_)) {
        if (last_abort_msg_.empty()) {
          last_abort_msg_ = "[TERMINAL_APPROACH_ABORT] reason=LATCHED_GOAL_LOST_INSIDE_CUTOFF";
        }
        throw nav2_core::InvalidPath(last_abort_msg_);
      }
      return;
    }
    if (!replan_pending_ && cutoff_radius_ > 0.0f &&
      goal_dist <= static_cast<double>(cutoff_radius_) && data.path.x.shape(0) >= 2u)
    {
      if (!data.reuse_obstacles_cache || !tick_evaluated_) {
        tick_evaluated_ = true;
        const size_t n_path = data.path.x.shape(0);
        size_t start_idx = 0u;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_path; ++i) {
          const double d = std::hypot(
            robot_x - static_cast<double>(data.path.x(i)),
            robot_y - static_cast<double>(data.path.y(i)));
          if (d < best_d) {
            best_d = d;
            start_idx = i;
          }
        }
        start_idx = std::min(start_idx, n_path - 2u);
        cached_active_path_.clear();
        cached_active_path_.reserve(n_path - start_idx);
        for (size_t i = start_idx; i < n_path; ++i) {
          cached_active_path_.push_back(
            utils::Pose2D{data.path.x(i), data.path.y(i), data.path.yaws(i)});
        }
        pre_cusp_gear_ = 1.0;
        for (size_t s = start_idx; s + 1u < n_path; ++s) {
          const double seg_dx = static_cast<double>(data.path.x(s + 1u) - data.path.x(s));
          const double seg_dy = static_cast<double>(data.path.y(s + 1u) - data.path.y(s));
          if (std::hypot(seg_dx, seg_dy) > 1e-6) {
            const double seg_th = static_cast<double>(data.path.yaws(s));
            pre_cusp_gear_ =
              (seg_dx * std::cos(seg_th) + seg_dy * std::sin(seg_th) < 0.0) ? -1.0 : 1.0;
            break;
          }
        }
      }
      scoreLatchedRollouts(data, robot_x, robot_y, pre_cusp_gear_);
    }
    return;
  }

  const double radius_limit = static_cast<double>(std::max(0.05f, approach_radius_)) +
    ((latched_ || approach_escalated_ || replan_pending_) ? kLatchRadiusHysteresisM : 0.0);
  if (goal_dist > radius_limit) {
    return;
  }

  double goal_xy_tol = std::max(0.01, static_cast<double>(goal_xy_tolerance_));
  double goal_yaw_tol = std::max(0.01, static_cast<double>(goal_yaw_tolerance_));
  if (data.goal_checker != nullptr) {
    geometry_msgs::msg::Pose pose_tol;
    geometry_msgs::msg::Twist vel_tol;
    if (data.goal_checker->getTolerances(pose_tol, vel_tol)) {
      const double gx = pose_tol.position.x, gy = pose_tol.position.y;
      const double gc_xy = (gx > 0.0 && gy > 0.0) ? std::min(gx, gy) : std::max(gx, gy);
      const double gc_yaw = std::fabs(tf2::getYaw(pose_tol.orientation));
      if (finiteValue(gc_xy) && gc_xy > 0.0) {
        goal_xy_tol = gc_xy;
      }
      if (finiteValue(gc_yaw) && gc_yaw > 0.0) {
        goal_yaw_tol = gc_yaw;
      }
    }
  }

  if (!data.reuse_obstacles_cache) {
    tick_evaluated_ = false;
  }
  if (data.reuse_obstacles_cache && tick_evaluated_) {
    if (!latched_ || cached_active_path_.size() < 2u) {
      return;
    }
  } else {
    tick_evaluated_ = true;
    const double path_cross_track = minDistanceToPath(robot_x, robot_y, data.path);
    double projected_path_yaw = static_cast<double>(data.path.yaws(0));
    size_t closest_path_seg_idx = 0u;
    const size_t n_path = data.path.x.shape(0);
    const double max_ray = std::max(
      kInitialSegmentRayExtrapM, static_cast<double>(approach_radius_) + kLatchRadiusHysteresisM);
    double best_d = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < n_path; ++i) {
      const double ax = data.path.x(i), ay = data.path.y(i);
      const double bx = data.path.x(i + 1), by = data.path.y(i + 1);
      const double dx = bx - ax, dy = by - ay;
      const double len_sq = dx * dx + dy * dy;
      const double min_t = (i == 0u && len_sq > 1e-10) ? (-max_ray / std::sqrt(len_sq)) : 0.0;
      const double t = (len_sq > 1e-10) ?
        std::clamp(((robot_x - ax) * dx + (robot_y - ay) * dy) / len_sq, min_t, 1.0) : 0.0;
      const double d = std::hypot(robot_x - (ax + t * dx), robot_y - (ay + t * dy));
      if (d < best_d) {
        best_d = d;
        closest_path_seg_idx = i;
        const double th_a = static_cast<double>(data.path.yaws(i));
        const double th_b = static_cast<double>(data.path.yaws(i + 1));
        const double t_yaw = (len_sq >= 1e-4) ? std::clamp(t, 0.0, 1.0) : 0.0;
        projected_path_yaw = angles::normalize_angle(
          th_a + t_yaw * angles::shortest_angular_distance(th_a, th_b));
      }
    }

    double path_gear = 1.0;
    bool inferred_from_seg = false;
    for (size_t s = terminal_goal_idx; s-- > 0u; ) {
      const double seg_dx = static_cast<double>(data.path.x(s + 1u) - data.path.x(s));
      const double seg_dy = static_cast<double>(data.path.y(s + 1u) - data.path.y(s));
      if (std::hypot(seg_dx, seg_dy) > 1e-6) {
        const double seg_th = static_cast<double>(data.path.yaws(s));
        path_gear = (seg_dx * std::cos(seg_th) + seg_dy * std::sin(seg_th) < 0.0) ? -1.0 : 1.0;
        inferred_from_seg = true;
        break;
      }
    }
    if (!inferred_from_seg) {
      if (latched_ || approach_escalated_ || replan_pending_) {
        path_gear = latched_gear_;
      } else if (has_plan_gear_) {
        path_gear = last_plan_gear_;
      } else {
        const double d_nose =
          (goal_x - robot_x) * std::cos(goal_yaw) + (goal_y - robot_y) * std::sin(goal_yaw);
        path_gear = (d_nose < kReverseNoseThresholdM) ? -1.0 : 1.0;
      }
    } else if ((latched_ || approach_escalated_) && (path_gear * latched_gear_ < 0.0)) {
      const bool had_active_latch = latched_ && !approach_escalated_;
      const bool after_replan = replan_allowance_consumed_ && !replan_pending_;
      if (after_replan || max_redraws_ == 0 || redraw_count_ >= max_redraws_ || !had_active_latch) {
        const char * r = after_replan ?
          "GEAR_FLIP_AFTER_REPLAN_CONSUMED" : "GEAR_FLIP_WHILE_LATCHED";
        clearActiveCurve();
        latch_consumed_ = true;
        approach_escalated_ = true;
        if (escalation_reason_.empty()) {
          escalation_reason_ = r;
        }
        if (last_abort_msg_.empty()) {
          last_abort_msg_ = std::string("[TERMINAL_APPROACH_ABORT] reason=") + r;
        }
        if (had_active_latch) {
          RCLCPP_INFO(
            logger_, "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=%s "
            "goal_dist=%.4f cross_track=%.4f yaw_err=0.0000 redraws=%d/%d replans=%d",
            infeasible_policy_.c_str(), r, goal_dist, path_cross_track,
            redraw_count_, max_redraws_, replan_count_);
          requestReplanIfEligible(goal_dist);
        }
      } else {
        ++redraw_count_;
        clearActiveCurve();
      }
    }

    if (replan_pending_) {
      if (goal_dist <= static_cast<double>(cutoff_radius_)) {
        replan_pending_ = false;
        approach_escalated_ = true;
        char msg_buf[256];
        std::snprintf(
          msg_buf, sizeof(msg_buf), "[TERMINAL_APPROACH_ABORT] reason=REPLAN_NOT_READY_BY_CUTOFF "
          "goal_dist=%.4fm cutoff=%.4fm replans=%d",
          goal_dist, static_cast<double>(cutoff_radius_), replan_count_);
        last_abort_msg_ = msg_buf;
        RCLCPP_INFO(
          logger_, "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=REPLAN_NOT_READY_BY_CUTOFF "
          "goal_dist=%.4f cross_track=%.4f yaw_err=0.0000 redraws=%d/%d replans=%d",
          infeasible_policy_.c_str(), goal_dist, path_cross_track,
          redraw_count_, max_redraws_, replan_count_);
        throw nav2_core::InvalidPath(last_abort_msg_);
      }
      return;
    }

    if (approach_escalated_) {
      const bool inside_cutoff =
        (cutoff_radius_ > 0.0f) && (goal_dist <= static_cast<double>(cutoff_radius_));
      if (inside_cutoff ||
        ((infeasible_policy_ == "abort" ||
        (infeasible_policy_ == "replan_signal" && replan_allowance_consumed_)) &&
        !last_abort_msg_.empty()))
      {
        if (last_abort_msg_.empty()) {
          const char * cutoff_reason =
            escalation_reason_.empty() ? "UNLATCHED_AT_CUTOFF" : escalation_reason_.c_str();
          char msg_buf[256];
          std::snprintf(
            msg_buf, sizeof(msg_buf), "[TERMINAL_APPROACH_ABORT] reason=%s "
            "goal_dist=%.4fm cutoff=%.4fm redraws=%d/%d replans=%d",
            cutoff_reason, goal_dist, static_cast<double>(cutoff_radius_),
            redraw_count_, max_redraws_, replan_count_);
          last_abort_msg_ = msg_buf;
          RCLCPP_INFO(
            logger_, "[TERMINAL_APPROACH_ESCALATE] policy=%s reason=%s "
            "goal_dist=%.4f cross_track=%.4f yaw_err=0.0000 redraws=%d/%d replans=%d",
            infeasible_policy_.c_str(), cutoff_reason, goal_dist, path_cross_track,
            redraw_count_, max_redraws_, replan_count_);
        }
        throw nav2_core::InvalidPath(last_abort_msg_);
      }
      return;
    }

    const double gear = latched_ ? latched_gear_ : path_gear;
    const double path_yaw_err =
      std::fabs(angles::shortest_angular_distance(robot_yaw, projected_path_yaw));
    const double path_to_goal_yaw_diff =
      std::fabs(angles::shortest_angular_distance(projected_path_yaw, goal_yaw));

    if (latched_ && !verifyLatchedCurveOrRedraw(
        data, robot_x, robot_y, goal_x, goal_y, goal_yaw, goal_dist, path_yaw_err,
        goal_xy_tol, goal_yaw_tol))
    {
      return;
    }

    if (!latched_) {
      const double min_dist = static_cast<double>(std::max(0.05f, min_approach_distance_));
      const double r_min = std::max(0.10, static_cast<double>(min_turning_radius_));
      const double turn_chord =
        std::min(goal_dist, static_cast<double>(std::max(0.0f, cutoff_radius_)));
      const double turn_sagitta_allowance =
        ((turn_chord * turn_chord) / (kTurnSagittaDivisor * r_min)) *
        std::min(1.0, path_to_goal_yaw_diff);
      const double viable_cross_track_limit =
        static_cast<double>(viable_path_tracking_ratio_) * goal_xy_tol + turn_sagitta_allowance;
      const double viable_yaw_limit =
        goal_yaw_tol + kTurnCorridorYawAllowanceRatio * std::min(1.0, path_to_goal_yaw_diff);
      const double travel_goal_yaw =
        gear < 0.0 ? angles::normalize_angle(goal_yaw + M_PI) : goal_yaw;
      const double dx_g = robot_x - goal_x, dy_g = robot_y - goal_y;
      const double lon_remaining =
        -(dx_g * std::cos(travel_goal_yaw) + dy_g * std::sin(travel_goal_yaw));
      const double y0_goal = -dx_g * std::sin(travel_goal_yaw) + dy_g * std::cos(travel_goal_yaw);
      const double robot_travel_yaw =
        gear < 0.0 ? angles::normalize_angle(robot_yaw + M_PI) : robot_yaw;
      const double theta0_goal =
        angles::shortest_angular_distance(travel_goal_yaw, robot_travel_yaw);
      const double max_turn_over_lon = std::max(0.0, lon_remaining) / r_min;
      const double steer_sign = (theta0_goal >= 0.0) ? -1.0 : 1.0;
      const double theta_mid =
        theta0_goal + 0.5 * steer_sign * std::min(std::fabs(theta0_goal), max_turn_over_lon);
      const double y_end_best = y0_goal + std::max(0.0, lon_remaining) * std::sin(theta_mid);
      const bool behind_goal_ok = (lon_remaining >= 0.02) ||
        (lon_remaining >= -0.02 && std::fabs(y0_goal) <= kGoalBoxXyTargetRatio * goal_xy_tol &&
        std::fabs(theta0_goal) <= goal_yaw_tol);
      const bool sub_min_kinematically_reachable = (lon_remaining >= 0.02) &&
        (path_cross_track <= viable_cross_track_limit) &&
        (path_yaw_err <= viable_yaw_limit + max_turn_over_lon) &&
        (std::fabs(y_end_best) <= viable_cross_track_limit);
      const bool ref_path_free =
        isReferencePathCollisionFree(data.path, closest_path_seg_idx, terminal_goal_idx);
      const bool viable_free = ref_path_free &&
        (path_cross_track <= viable_cross_track_limit) && (path_yaw_err <= viable_yaw_limit);
      const bool needs_capture =
        path_cross_track >= static_cast<double>(activation_drift_) ||
        path_yaw_err >= static_cast<double>(activation_yaw_drift_);
      const bool is_short_approach = goal_dist < static_cast<double>(short_approach_distance_);
      const bool is_curved_turn_entry =
        path_to_goal_yaw_diff > static_cast<double>(max_entry_path_turn_angle_);
      const bool prefer_reference_path =
        viable_free && (is_curved_turn_entry || behind_goal_ok) &&
        (!needs_capture || is_short_approach || is_curved_turn_entry);
      const double ref_cutoff = effectiveRefCutoff();
      if (prefer_reference_path && ref_cutoff > 0.0 && goal_dist > ref_cutoff &&
        !replan_allowance_consumed_)
      {
        return;
      }

      std::vector<utils::Pose2D> curve;
      std::string selected_family;
      double selected_r = 0.0;
      bool latched_ref = false, solved_initial_latch = false;
      if (prefer_reference_path && buildReferencePathCurve(
          data.path, closest_path_seg_idx, terminal_goal_idx, data.goal, gear, curve, goal_dist))
      {
        selected_family = "reference_path";
        latched_ref = solved_initial_latch = true;
      }
      if (!solved_initial_latch &&
        ((goal_dist >= min_dist && solveCaptureCurve(
          data.state.pose.pose, data.goal, gear, curve, goal_xy_tol, goal_yaw_tol,
          nullptr, &selected_family, &selected_r)) ||
        (is_short_approach && goal_dist < min_dist && ref_path_free &&
        sub_min_kinematically_reachable && buildReferencePathCurve(
          data.path, closest_path_seg_idx, terminal_goal_idx, data.goal, gear, curve, goal_dist))))
      {
        latched_ref = selected_family.empty();
        if (latched_ref) {
          selected_family = "reference_path";
          selected_r = 0.0;
        }
        solved_initial_latch = true;
      }
      if (!solved_initial_latch) {
        latched_goal_ = latch_origin_goal_ = data.goal;
        latched_frame_id_ = data.state.pose.header.frame_id;
        latched_gear_ = gear;
        handleInfeasible("ENTRY_INFEASIBLE", path_cross_track, goal_dist, path_yaw_err);
        return;
      }

      latched_ = latch_consumed_ = true;
      latched_is_reference_path_ = latched_ref;
      approach_escalated_ = replan_signal_sent_ = false;
      latched_goal_ = latch_origin_goal_ = data.goal;
      latched_frame_id_ = data.state.pose.header.frame_id;
      latched_gear_ = gear;
      latched_curve_ = std::move(curve);
      latch_initial_cross_track_ = minDistanceToCurve(robot_x, robot_y, latched_curve_);
      snapshotLatchParameters();
      RCLCPP_INFO(
        logger_, "[TERMINAL_APPROACH_LATCH] family=%s R=%.4f goal_dist=%.4f cross_track=%.4f "
        "yaw_err=%.4f gear=%.0f",
        selected_family.c_str(), selected_r, goal_dist, path_cross_track, path_yaw_err, gear);
    }
    if (!latched_ || latched_curve_.size() < 2u) {
      return;
    }
    updateCachedActivePath(robot_x, robot_y);
  }
  const double ref_cutoff = effectiveRefCutoff();
  if (latched_ && latched_is_reference_path_ && ref_cutoff > 0.0 && goal_dist > ref_cutoff) {
    return;
  }
  scoreLatchedRollouts(data, robot_x, robot_y, latched_gear_);
}

bool TerminalApproachCritic::verifyLatchedCurveOrRedraw(
  const CriticData & data,
  double robot_x,
  double robot_y,
  double goal_x,
  double goal_y,
  double goal_yaw,
  double goal_dist,
  double path_yaw_err,
  double goal_xy_tol,
  double goal_yaw_tol)
{
  size_t closest_curve_idx = 0u;
  double best_curve_pt_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < latched_curve_.size(); ++i) {
    const double d = std::hypot(robot_x - latched_curve_[i].x, robot_y - latched_curve_[i].y);
    if (d < best_curve_pt_dist) {
      best_curve_pt_dist = d;
      closest_curve_idx = i;
    }
  }

  bool latched_blocked = false;
  for (size_t i = closest_curve_idx; i < latched_curve_.size(); ++i) {
    const auto & p = latched_curve_[i];
    if (!isFootprintCollisionFree(p.x, p.y, p.theta)) {
      latched_blocked = true;
      break;
    }
    if (i + 1u < latched_curve_.size()) {
      const auto & q = latched_curve_[i + 1u];
      const double seg_len = std::hypot(
        static_cast<double>(q.x) - static_cast<double>(p.x),
        static_cast<double>(q.y) - static_cast<double>(p.y));
      if (seg_len > kMaxCollisionCheckStepM) {
        const size_t sub_steps =
          static_cast<size_t>(std::ceil(seg_len / kMaxCollisionCheckStepM));
        const double dth = angles::shortest_angular_distance(
          static_cast<double>(p.theta), static_cast<double>(q.theta));
        for (size_t k = 1u; k < sub_steps; ++k) {
          const double alpha = static_cast<double>(k) / static_cast<double>(sub_steps);
          const double mx =
            static_cast<double>(p.x) + alpha * (static_cast<double>(q.x) - p.x);
          const double my =
            static_cast<double>(p.y) + alpha * (static_cast<double>(q.y) - p.y);
          const double mth = angles::normalize_angle(
            static_cast<double>(p.theta) + alpha * dth);
          if (!isFootprintCollisionFree(mx, my, mth)) {
            latched_blocked = true;
            break;
          }
        }
        if (latched_blocked) {
          break;
        }
      }
    }
  }

  const double live_max_end_xy = std::min(
    std::max(0.0, static_cast<double>(max_residual_y_)),
    kGoalBoxXyTargetRatio * goal_xy_tol);
  const double live_max_end_yaw = std::min(
    std::max(0.0, static_cast<double>(max_residual_yaw_)),
    kGoalBoxYawTargetRatio * goal_yaw_tol);
  const auto & end_pt = latched_curve_.back();
  const double end_err_xy = std::hypot(
    static_cast<double>(end_pt.x) - goal_x,
    static_cast<double>(end_pt.y) - goal_y);
  const double end_err_yaw = std::fabs(
    angles::shortest_angular_distance(static_cast<double>(end_pt.theta), goal_yaw));
  const bool latched_tol_violated =
    (end_err_xy > live_max_end_xy + 1e-5) || (end_err_yaw > live_max_end_yaw + 1e-5);

  const double tracking_drift = minDistanceToCurve(robot_x, robot_y, latched_curve_);
  const double ref_cutoff = effectiveRefCutoff();
  const bool passive_overlap_ref =
    latched_is_reference_path_ && (ref_cutoff > 0.0) && (goal_dist > ref_cutoff);
  if (passive_overlap_ref) {
    latch_initial_cross_track_ = tracking_drift;
  }
  const double effective_max_drift =
    latch_initial_cross_track_ + static_cast<double>(max_tracking_drift_);
  const bool drift_violated = !passive_overlap_ref && (tracking_drift > effective_max_drift);
  if (latched_blocked || latched_tol_violated || drift_violated) {
    const char * fail_reason = latched_blocked ? "LATCHED_CURVE_BLOCKED" :
      (latched_tol_violated ? "LATCHED_TOLERANCE_VIOLATED" :
      (local_goal_shift_caused_drift_ ? "LOCALIZATION_JUMP" : "DRIFT_EXCEEDED"));
    if (redraw_count_ < max_redraws_) {
      std::vector<utils::Pose2D> redrawn;
      std::string redrawn_family;
      double redrawn_r = 0.0;
      if (goal_dist >= static_cast<double>(min_approach_distance_) &&
        solveCaptureCurve(
          data.state.pose.pose, data.goal, latched_gear_, redrawn,
          goal_xy_tol, goal_yaw_tol, nullptr, &redrawn_family, &redrawn_r))
      {
        latched_curve_ = std::move(redrawn);
        latched_is_reference_path_ = false;
        latch_initial_cross_track_ = minDistanceToCurve(robot_x, robot_y, latched_curve_);
        latch_origin_goal_ = data.goal;
        ++redraw_count_;
        snapshotLatchParameters();
        RCLCPP_INFO(
          logger_,
          "[TERMINAL_APPROACH_REDRAW] redraw=%d/%d reason=%s family=%s R=%.4f "
          "goal_dist=%.4f drift=%.4f",
          redraw_count_, max_redraws_, fail_reason, redrawn_family.c_str(), redrawn_r,
          goal_dist, tracking_drift);
        return true;
      }
    }
    latched_goal_ = data.goal;
    latch_origin_goal_ = data.goal;
    latched_frame_id_ = data.state.pose.header.frame_id;
    handleInfeasible(fail_reason, tracking_drift, goal_dist, path_yaw_err);
    return false;
  }
  if (!passive_overlap_ref) {
    latch_initial_cross_track_ = std::min(latch_initial_cross_track_, tracking_drift);
  }
  return true;
}

void TerminalApproachCritic::updateCachedActivePath(double robot_x, double robot_y)
{
  if (latched_curve_.size() < 2u) {
    cached_active_path_.clear();
    return;
  }
  size_t start_idx = 0u;
  double best_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < latched_curve_.size(); ++i) {
    const double d = std::hypot(robot_x - latched_curve_[i].x, robot_y - latched_curve_[i].y);
    if (d < best_dist) {
      best_dist = d;
      start_idx = i;
    }
  }
  if (start_idx + 1u >= latched_curve_.size()) {
    start_idx = latched_curve_.size() - 2u;
  }

  cached_active_path_.assign(
    latched_curve_.begin() + static_cast<std::ptrdiff_t>(start_idx),
    latched_curve_.end());

  const auto & end_pose = latched_curve_.back();
  const double end_nose_yaw = static_cast<double>(end_pose.theta);
  const double end_travel_yaw =
    latched_gear_ < 0.0 ? angles::normalize_angle(end_nose_yaw + M_PI) : end_nose_yaw;
  const float along_track_remaining = static_cast<float>(
    -((robot_x - end_pose.x) * std::cos(end_travel_yaw) +
    (robot_y - end_pose.y) * std::sin(end_travel_yaw)));

  const float pull_len = computePullthroughLength(along_track_remaining);
  if (pull_len > 1e-4f) {
    const double spacing = std::max(0.01, static_cast<double>(sample_spacing_));
    const size_t num_pull = std::max<size_t>(
      1u, static_cast<size_t>(std::ceil(static_cast<double>(pull_len) / spacing)));
    double prev_dist = 0.0;
    for (size_t k = 1u; k <= num_pull; ++k) {
      const double dist_k =
        (static_cast<double>(pull_len) * static_cast<double>(k)) / static_cast<double>(num_pull);
      const double step_len = dist_k - prev_dist;
      const size_t sub_checks = std::max<size_t>(
        1u, static_cast<size_t>(std::ceil(step_len / kMaxCollisionCheckStepM)));
      bool step_free = true;
      for (size_t sc = 1u; sc <= sub_checks; ++sc) {
        const double d_sc = prev_dist + step_len * (static_cast<double>(sc) / sub_checks);
        const double sx = static_cast<double>(end_pose.x) + d_sc * std::cos(end_travel_yaw);
        const double sy = static_cast<double>(end_pose.y) + d_sc * std::sin(end_travel_yaw);
        if (!isFootprintCollisionFree(sx, sy, end_nose_yaw)) {
          step_free = false;
          break;
        }
      }
      if (!step_free) {
        break;
      }
      prev_dist = dist_k;
      const double px = static_cast<double>(end_pose.x) + dist_k * std::cos(end_travel_yaw);
      const double py = static_cast<double>(end_pose.y) + dist_k * std::sin(end_travel_yaw);
      cached_active_path_.push_back(
        utils::Pose2D{
          static_cast<float>(px),
          static_cast<float>(py),
          end_pose.theta});
    }
  }
}

void TerminalApproachCritic::scoreLatchedRollouts(
  CriticData & data,
  double robot_x,
  double robot_y,
  double active_gear) const
{
  const auto & active_path = cached_active_path_;
  const size_t batch_size = data.trajectories.x.shape(0);
  const size_t time_steps = data.trajectories.x.shape(1);
  if (batch_size == 0u || time_steps == 0u || active_path.size() < 2u) {
    return;
  }

  const auto & end_pose = latched_ ? latched_curve_.back() : active_path.back();
  const double end_nose_yaw = static_cast<double>(end_pose.theta);
  const double end_travel_yaw =
    active_gear < 0.0 ? angles::normalize_angle(end_nose_yaw + M_PI) : end_nose_yaw;
  const float along_track_remaining = static_cast<float>(
    -((robot_x - end_pose.x) * std::cos(end_travel_yaw) +
    (robot_y - end_pose.y) * std::sin(end_travel_yaw)));

  const size_t total_segments = active_path.size() - 1u;
  const size_t reference_point_count = active_path.size();

  std::vector<float> path_integrated_distances(reference_point_count, 0.0f);
  for (size_t i = 1u; i < reference_point_count; ++i) {
    const float dx = active_path[i].x - active_path[i - 1u].x;
    const float dy = active_path[i].y - active_path[i - 1u].y;
    path_integrated_distances[i] =
      path_integrated_distances[i - 1u] + std::sqrt(dx * dx + dy * dy);
  }
  const float total_curve_length = path_integrated_distances.back();

  struct CurveProjection
  {
    float s_proj{0.0f};
    float d_perp{0.0f};
    float d_yaw{0.0f};
  };

  auto projectOntoCurve =
    [&active_path, &path_integrated_distances, total_segments](
    float px, float py, float pyaw) -> CurveProjection
    {
      size_t best_i = 0u;
      float best_alpha = 0.0f;
      float best_d_sq = std::numeric_limits<float>::infinity();

      for (size_t i = 0u; i < total_segments; ++i) {
        const float ax = active_path[i].x, ay = active_path[i].y;
        const float bx = active_path[i + 1u].x, by = active_path[i + 1u].y;
        const float ux = bx - ax, uy = by - ay;
        const float len_sq = ux * ux + uy * uy;
        const float alpha = (len_sq > 1e-12f) ?
          std::clamp(((px - ax) * ux + (py - ay) * uy) / len_sq, 0.0f, 1.0f) : 0.0f;
        const float dx = px - (ax + alpha * ux), dy = py - (ay + alpha * uy);
        const float d_sq = dx * dx + dy * dy;
        if (d_sq < best_d_sq) {
          best_d_sq = d_sq;
          best_i = i;
          best_alpha = alpha;
        }
      }

      const float seg_len =
        path_integrated_distances[best_i + 1u] - path_integrated_distances[best_i];
      const float s_proj = path_integrated_distances[best_i] + best_alpha * seg_len;
      const float d_perp = std::sqrt(best_d_sq);
      const double a_th = static_cast<double>(active_path[best_i].theta);
      const double b_th = static_cast<double>(active_path[best_i + 1u].theta);
      const double q_th =
        a_th + static_cast<double>(best_alpha) * angles::shortest_angular_distance(a_th, b_th);
      const float d_yaw = static_cast<float>(
        std::fabs(angles::shortest_angular_distance(static_cast<double>(pyaw), q_th)));
      return CurveProjection{s_proj, d_perp, d_yaw};
    };

  const size_t step = static_cast<size_t>(std::max(1, trajectory_point_step_));
  const double dist_to_end = std::hypot(robot_x - end_pose.x, robot_y - end_pose.y);
  const double goal_dist =
    std::hypot(robot_x - data.goal.position.x, robot_y - data.goal.position.y);
  const bool in_release_zone =
    latched_ &&
    (along_track_remaining <= pullthrough_release_) && (dist_to_end <= kReleaseZoneGoalDistM);
  const bool pfc_active_at_goal_dist = (cutoff_radius_ > 0.0f) && (
    stock_thresholds_configured_ ?
    (stock_pfc_active_ && (goal_dist > static_cast<double>(effective_pfc_cutoff_))) :
    (goal_dist > static_cast<double>(cutoff_radius_)));
  const float effective_follow_weight =
    (latched_ && !in_release_zone && pfc_active_at_goal_dist) ? 0.0f : follow_weight_;
  const bool has_state_vel =
    (data.state.vx.shape(0) == batch_size) &&
    (data.state.wz.shape(0) == batch_size) &&
    (data.state.vx.shape(1) > 0u) &&
    (data.state.wz.shape(1) == data.state.vx.shape(1));
  const size_t vel_steps = has_state_vel ? data.state.vx.shape(1) : 0u;

  auto && total_cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});

  for (size_t t = 0; t < batch_size; ++t) {
    float summed_align_err = 0.0f;
    float summed_remaining_geodesic = 0.0f;
    unsigned int num_samples = 0u;

    for (size_t p = step; p < time_steps; p += step) {
      const float Tx = data.trajectories.x(t, p);
      const float Ty = data.trajectories.y(t, p);
      const float Tyaw = data.trajectories.yaws(t, p);
      const auto proj = projectOntoCurve(Tx, Ty, Tyaw);
      summed_align_err += std::sqrt(
        proj.d_perp * proj.d_perp + kCurveYawErrorScaleSq * proj.d_yaw * proj.d_yaw);
      summed_remaining_geodesic +=
        std::max(0.0f, total_curve_length - proj.s_proj) + proj.d_perp;
      ++num_samples;
    }

    const float last_x = data.trajectories.x(t, time_steps - 1u);
    const float last_y = data.trajectories.y(t, time_steps - 1u);
    const float last_yaw = data.trajectories.yaws(t, time_steps - 1u);
    const auto last_proj = projectOntoCurve(last_x, last_y, last_yaw);
    const float last_remaining_geodesic =
      std::max(0.0f, total_curve_length - last_proj.s_proj) + last_proj.d_perp;

    const float last_align_err = std::sqrt(
      last_proj.d_perp * last_proj.d_perp +
      kCurveYawErrorScaleSq * last_proj.d_yaw * last_proj.d_yaw);
    const float align_term = align_weight_ * ((num_samples > 0u) ?
      (summed_align_err / static_cast<float>(num_samples)) : last_align_err);

    const float mean_remaining_geodesic = (num_samples > 0u) ?
      (summed_remaining_geodesic / static_cast<float>(num_samples)) : last_remaining_geodesic;
    const float follow_term = effective_follow_weight * (
      kFollowTerminalWeightRatio * last_remaining_geodesic +
      kFollowMeanWeightRatio * mean_remaining_geodesic);

    float settle_term = 0.0f;
    if (in_release_zone && settle_weight_ > 0.0f && has_state_vel) {
      float speed_sum = 0.0f;
      for (size_t j = 0; j < vel_steps; ++j) {
        speed_sum +=
          std::fabs(data.state.vx(t, j)) +
          kSettleAngularSpeedScale * std::fabs(data.state.wz(t, j));
      }
      settle_term = settle_weight_ * (speed_sum / static_cast<float>(vel_steps));
    }

    total_cost[t] = align_term + follow_term + settle_term;
  }

  if (power_ > 1u) {
    data.costs += xt::pow(std::move(total_cost), power_);
  } else {
    data.costs += std::move(total_cost);
  }
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::TerminalApproachCritic,
  mppi::critics::CriticFunction)
