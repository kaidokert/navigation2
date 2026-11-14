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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__TERMINAL_APPROACH_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__TERMINAL_APPROACH_CRITIC_HPP_

#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "std_msgs/msg/bool.hpp"

#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @struct CaptureFamilyTrace
 * @brief Diagnostic trace entry recording why a candidate curve family was accepted or rejected
 */
struct CaptureFamilyTrace
{
  std::string family;
  bool accepted{false};
  double R{0.0};
  double L1{0.0};
  double Ls{0.0};
  double L2{0.0};
  double y_end{0.0};
  double theta_end{0.0};
  std::string rejection_reason;
};

/**
 * @class mppi::critics::TerminalApproachCritic
 * @brief Option A1.1 Terminal Approach Critic with Continuous Concentric-Circle Handoff:
 *   - Uses concentric circles around data.goal: outer activation circle (approach_radius,
 *     default 0.65 m) and inner stock-critic cutoff circle (cutoff_radius, default 0.35 m,
 *     matching PathAlignCritic/PathFollowCritic threshold_to_consider).
 *   - Decides in the overlap ring (cutoff_radius < d_goal <= approach_radius) while stock
 *     critics continue guiding the robot:
 *     a1) Establishes a latched terminal approach before cutoff_radius—either a 1-gear
 *         collision-free Ackermann capture curve when corrective capture is needed, or the
 *         collision-free reference path (family="reference_path") for nominal, M5/M6 curved,
 *         and M8 short post-cusp approaches—and owns arrival guidance inside cutoff_radius.
 *     a2) If infeasible, enforces parameter-controlled escalation ("abort" throws
 *         nav2_core::InvalidPath immediately with zero replans; "replan_signal" consumes a
 *         strictly bounded 1-replacement-per-attempt allowance in the overlap ring without
 *         stopping the robot, and aborts if the replacement is not ready by cutoff_radius or
 *         if the replacement approach fails).
 */
class TerminalApproachCritic : public CriticFunction
{
public:
  /**
   * @brief Destructor invalidating any registered parameter callback guard
   */
  ~TerminalApproachCritic() override;

  /**
   * @brief Default-off safety for experimental terminal approach critic
   * @return false when parameter is not explicitly set
   */
  bool defaultEnabled() const override
  {
    return false;
  }

  /**
   * @brief Initialize critic parameters and publisher
   */
  void initialize() override;

  /**
   * @brief Activate critic lifecycle resources
   */
  void on_activate() override;

  /**
   * @brief Deactivate critic lifecycle resources
   */
  void on_deactivate() override;

  /**
   * @brief Cleanup critic lifecycle resources
   */
  void on_cleanup() override;

  /**
   * @brief Evaluate terminal approach feasibility and score rollouts along latched curve
   * @param data Critic data for current MPPI evaluation
   */
  void score(CriticData & data) override;

  /**
   * @brief Reset critic state when controller resets between tasks
   */
  void reset() override;

  /**
   * @brief Handle new global plan arrival via setPlan() while preserving an active valid latch
   *        across periodic replans to the same goal and terminal gear
   * @param plan Incoming global plan in its original frame before pruning
   */
  void onNewPlan(const nav_msgs::msg::Path & plan) override;

  /**
   * @brief Reset per-tick evaluation flag at the start of each evalControl() tick
   */
  void onTickStart() override
  {
    tick_evaluated_ = false;
    local_goal_shift_caused_drift_ = false;
  }

  /**
   * @brief Reset any latched capture curve and escalation state
   */
  void resetLatch();

  /**
   * @brief Compute smooth tapered pull-through extension length past the goal
   * @param along_track_remaining Longitudinal distance remaining to the goal (m)
   * @return Pull-through extension length in meters (0.0 at or below pullthrough_release)
   */
  float computePullthroughLength(float along_track_remaining) const;

  /**
   * @brief Trace candidate curve families and their acceptance/rejection reasons for a pose
   */
  std::vector<CaptureFamilyTrace> traceCaptureCandidates(
    const geometry_msgs::msg::Pose & robot_pose,
    const geometry_msgs::msg::Pose & goal_pose,
    double gear,
    double goal_xy_tol = -1.0,
    double goal_yaw_tol = -1.0) const;

  /**
   * @brief Inspect whether a terminal approach (synthetic curve or reference path) is latched
   */
  bool isLatched() const
  {
    return latched_;
  }

  /**
   * @brief Inspect whether the active latch is following the collision-free reference path
   */
  bool isReferencePathLatched() const
  {
    return latched_ && latched_is_reference_path_;
  }

  /**
   * @brief Inspect the number of mid-approach redraws performed on the current latch
   */
  int redrawCount() const
  {
    return redraw_count_;
  }

  /**
   * @brief Inspect whether a non-stop replan signal was published on the current approach
   */
  bool replanSignalSent() const
  {
    return replan_signal_sent_;
  }

  /**
   * @brief Inspect whether a replacement plan has been requested and is awaiting delivery
   */
  bool isReplanPending() const
  {
    return replan_pending_;
  }

  /**
   * @brief Inspect whether the 1-replacement-per-attempt allowance has been consumed
   */
  bool isReplanAllowanceConsumed() const
  {
    return replan_allowance_consumed_;
  }

  /**
   * @brief Inspect the number of replacement plan requests emitted in the current attempt
   */
  int replanCount() const
  {
    return replan_count_;
  }

  /**
   * @brief Inspect whether the replan signal lifecycle publisher is currently activated
   */
  bool isReplanPublisherActivated() const
  {
    return replan_pub_ != nullptr && replan_pub_->is_activated();
  }

  /**
   * @brief Inspect the currently latched capture curve in costmap frame
   */
  const std::vector<utils::Pose2D> & latchedCurve() const
  {
    return latched_curve_;
  }

  /**
   * @brief Inspect the most recent escalation/abort diagnostic message
   */
  const std::string & lastAbortMessage() const
  {
    return last_abort_msg_;
  }

protected:
  /**
   * @brief Lazily create (and optionally activate) ~/terminal_replan_requested publisher
   */
  void ensureReplanPublisher(bool activate_now = true);

  /**
   * @brief Check if full oriented robot footprint is collision-free at costmap pose (wx, wy, wyaw)
   */
  bool isFootprintCollisionFree(double wx, double wy, double wyaw) const;

  /**
   * @brief Check if remaining reference path from start_idx to end_idx is footprint collision-free
   */
  bool isReferencePathCollisionFree(
    const models::Path & path,
    size_t start_idx,
    size_t end_idx) const;

  /**
   * @brief Build a densely sampled reference-path terminal curve ending at goal_pose
   */
  bool buildReferencePathCurve(
    const models::Path & path,
    size_t start_idx,
    size_t goal_idx,
    const geometry_msgs::msg::Pose & goal_pose,
    double gear,
    std::vector<utils::Pose2D> & out_curve,
    double min_lead_distance = 0.0) const;

  /**
   * @brief Solve a 1-gear collision-free Ackermann capture curve from robot_pose to goal_pose
   */
  bool solveCaptureCurve(
    const geometry_msgs::msg::Pose & robot_pose,
    const geometry_msgs::msg::Pose & goal_pose,
    double gear,
    std::vector<utils::Pose2D> & out_curve,
    double goal_xy_tol = -1.0,
    double goal_yaw_tol = -1.0,
    std::vector<CaptureFamilyTrace> * out_traces = nullptr,
    std::string * out_family = nullptr,
    double * out_radius = nullptr) const;

  /**
   * @brief Compute minimum point-to-segment distance from (rx, ry) to a polyline of Pose2D
   */
  double minDistanceToCurve(
    double rx, double ry,
    const std::vector<utils::Pose2D> & curve) const;

  /**
   * @brief Compute minimum point-to-segment distance from (rx, ry) to CriticData::path
   */
  double minDistanceToPath(
    double rx, double ry,
    const models::Path & path) const;

  /**
   * @brief Check whether any cached dynamic parameter changed since latch time
   */
  bool hasParameterChangedSinceLatch() const;

  /**
   * @brief Snapshot current parameter values into latch cache
   */
  void snapshotLatchParameters();

  /**
   * @brief Inspect stock PathFollowCritic / PathAlignCritic threshold_to_consider on the node
   *        and warn if lower than cutoff_radius_ when TerminalApproachCritic is enabled
   */
  void syncStockCriticThresholds(bool log_warning);

  /**
   * @brief Clear only the active latched curve without wiping escalation or redraw state
   */
  void clearActiveCurve();

  /**
   * @brief Escalate infeasibility or excessive tracking drift according to infeasible_policy_
   */
  void handleInfeasible(
    const char * reason,
    double cross_track_m,
    double goal_dist_m,
    double yaw_err_rad);

  /**
   * @brief Publish replan request when mid-approach latch invalidation occurs outside cutoff
   */
  void requestReplanIfEligible(double goal_dist_m);

  /**
   * @brief Effective cutoff radius where stock reference-path critics hand off to terminal critic
   */
  double effectiveRefCutoff() const;

  /**
   * @brief Validate the active latched curve against obstacles, tolerances, and drift
   */
  bool verifyLatchedCurveOrRedraw(
    const CriticData & data,
    double robot_x,
    double robot_y,
    double goal_x,
    double goal_y,
    double goal_yaw,
    double goal_dist,
    double path_yaw_err,
    double goal_xy_tol,
    double goal_yaw_tol);

  /**
   * @brief Prune latched_curve_ to the closest point and append tapered pull-through extension
   */
  void updateCachedActivePath(double robot_x, double robot_y);

  /**
   * @brief Score MPPI candidate rollouts along cached_active_path_
   */
  void scoreLatchedRollouts(
    CriticData & data,
    double robot_x,
    double robot_y,
    double active_gear) const;

  // --- Parameters (with ROBOT / GENERIC taxonomy) ---
  float approach_radius_{0.65f};               // GENERIC: Outer approach-circle entry radius (m)
  float cutoff_radius_{0.35f};                 // GENERIC: Inner stock-critic cutoff radius (m)
  float min_approach_distance_{0.16f};         // GENERIC: Min goal dist for curve solve (m)
  float short_approach_distance_{0.28f};       // GENERIC: Short/post-cusp pass-through dist (m)
  float min_turning_radius_{0.5605f};          // ROBOT: Min Ackermann radius L/tan(delta_max) (m)
  float activation_drift_{0.020f};             // GENERIC: Min cross-track offset at entry (m)
  float activation_yaw_drift_{0.080f};         // GENERIC: Min heading error at entry (rad)
  float max_entry_path_turn_angle_{0.50f};     // GENERIC: Max path-to-goal turn angle (rad)
  float goal_xy_tolerance_{0.10f};             // GENERIC: Fallback XY goal tolerance (m)
  float goal_yaw_tolerance_{0.25f};            // GENERIC: Fallback yaw goal tolerance (rad)
  float viable_path_tracking_ratio_{0.95f};    // GENERIC: Viable path corridor / xy_tol ratio
  float max_residual_y_{0.025f};               // GENERIC: Max bounded-R_min lateral residual (m)
  float max_residual_yaw_{0.19f};              // GENERIC: Max short-range yaw residual (rad)
  float max_tracking_drift_{0.045f};           // GENERIC: Max post-latch cross-track drift (m)
  float goal_reset_pos_tol_{0.25f};            // GENERIC: Goal pos shift for new-goal reset (m)
  float goal_reset_yaw_tol_{0.35f};            // GENERIC: Goal yaw shift for new-goal reset (rad)
  int max_redraws_{0};                         // GENERIC: Max mid-approach redraws (0 in tests)
  std::string infeasible_policy_{"abort"};     // GENERIC: "abort" | "replan_signal" | "fallback"
  float pullthrough_window_{0.25f};            // GENERIC: Along-track window for pull-through (m)
  float pullthrough_length_{0.10f};            // GENERIC: Max pull-through extension past goal (m)
  float pullthrough_release_{0.015f};          // GENERIC: Distance where pull-through = 0 (m)
  float sample_spacing_{0.025f};               // GENERIC: Capture curve sample spacing (m)
  float align_weight_{45.0f};                  // GENERIC: Curve cross-track & tangent align weight
  float follow_weight_{55.0f};                 // GENERIC: Geodesic progress weight along curve
  float settle_weight_{12.0f};                 // GENERIC: Release-zone velocity settling weight
  int trajectory_point_step_{2};               // GENERIC: Stride for rollout scoring
  unsigned int power_{1};                      // GENERIC: Cost power exponent

  // --- Latched state & dynamic-parameter cache keys ---
  bool is_active_{false};
  bool pending_replan_clear_{false};
  bool latched_{false};
  bool latch_consumed_{false};
  bool latched_is_reference_path_{false};
  bool approach_escalated_{false};
  bool replan_allowance_consumed_{false};
  bool replan_pending_{false};
  int replan_count_{0};
  bool tick_evaluated_{false};
  bool has_plan_goal_{false};
  bool has_plan_gear_{false};
  geometry_msgs::msg::Pose last_plan_goal_{};
  std::string last_plan_frame_{};
  double last_plan_gear_{1.0};
  geometry_msgs::msg::Pose latched_goal_{};
  geometry_msgs::msg::Pose latch_origin_goal_{};
  std::string latched_frame_id_{};
  double latched_gear_{1.0};
  double pre_cusp_gear_{1.0};
  double latch_initial_cross_track_{0.0};
  bool stock_thresholds_configured_{false};
  bool stock_pfc_active_{true};
  bool stock_pac_active_{true};
  float effective_pfc_cutoff_{0.35f};
  float effective_pac_cutoff_{0.35f};
  bool stock_thresholds_checked_{false};
  bool warned_stock_threshold_mismatch_{false};
  int redraw_count_{0};
  bool replan_signal_sent_{false};
  bool local_goal_shift_caused_drift_{false};
  std::string last_abort_msg_{};
  std::string escalation_reason_{};
  std::vector<utils::Pose2D> latched_curve_;
  std::vector<utils::Pose2D> cached_active_path_;
  std::shared_ptr<bool> callback_guard_;

  bool cached_enabled_{false};
  float cached_approach_radius_{0.0f};
  float cached_cutoff_radius_{0.0f};
  float cached_min_approach_distance_{0.0f};
  float cached_short_approach_distance_{0.0f};
  float cached_min_turning_radius_{0.0f};
  float cached_activation_drift_{0.0f};
  float cached_activation_yaw_drift_{0.0f};
  float cached_max_entry_path_turn_angle_{0.0f};
  float cached_goal_xy_tolerance_{0.0f};
  float cached_goal_yaw_tolerance_{0.0f};
  float cached_viable_path_tracking_ratio_{0.0f};
  float cached_max_residual_y_{0.0f};
  float cached_max_residual_yaw_{0.0f};
  float cached_max_tracking_drift_{0.0f};
  float cached_goal_reset_pos_tol_{0.0f};
  float cached_goal_reset_yaw_tol_{0.0f};
  int cached_max_redraws_{0};
  std::string cached_infeasible_policy_{};
  float cached_pullthrough_window_{0.0f};
  float cached_pullthrough_length_{0.0f};
  float cached_pullthrough_release_{0.0f};
  float cached_sample_spacing_{0.0f};
  float cached_align_weight_{0.0f};
  float cached_follow_weight_{0.0f};
  float cached_settle_weight_{0.0f};
  int cached_trajectory_point_step_{0};
  unsigned int cached_power_{1u};

  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>::SharedPtr replan_pub_;
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__TERMINAL_APPROACH_CRITIC_HPP_
