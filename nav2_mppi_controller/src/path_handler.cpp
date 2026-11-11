// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Dexory
// Copyright (c) 2023 Open Navigation LLC
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
#include <cmath>
#include <limits>
#include <optional>

#include "nav2_mppi_controller/tools/path_handler.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"

namespace mppi
{

void PathHandler::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap,
  std::shared_ptr<tf2_ros::Buffer> buffer, ParametersHandler * param_handler)
{
  name_ = name;
  costmap_ = costmap;
  tf_buffer_ = buffer;
  auto node = parent.lock();
  if (!node) {
    throw std::runtime_error("PathHandler: parent node expired during initialize()");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  parameters_handler_ = param_handler;

  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(max_robot_pose_search_dist_, "max_robot_pose_search_dist", getMaxCostmapDist());
  getParam(prune_distance_, "prune_distance", 1.5);
  getParam(transform_tolerance_, "transform_tolerance", 0.1);
  getParam(enforce_path_inversion_, "enforce_path_inversion", false);
  if (enforce_path_inversion_) {
    getParam(inversion_xy_tolerance_, "inversion_xy_tolerance", 0.2);
    getParam(inversion_yaw_tolerance_, "inversion_yaw_tolerance", 0.4);
    getParam(min_inversion_horizon_, "min_inversion_horizon", 0.15);
    getParam(ghost_max_extension_, "ghost_max_extension", 0.35);
    getParam(ghost_point_spacing_, "ghost_point_spacing", 0.05);
    getParam(ghost_min_seed_points_, "ghost_min_seed_points", 4);
    getParam(micro_cusp_length_threshold_, "micro_cusp_length_threshold", 0.25);
    getParam(micro_cusp_yaw_scale_, "micro_cusp_yaw_scale", 2.0);
    getParam(terminal_stub_max_length_, "terminal_stub_max_length", 0.08);
    getParam(terminal_stub_max_yaw_diff_, "terminal_stub_max_yaw_diff", 0.175);
    getParam(goal_checker_xy_tolerance_, "terminal_stub_goal_xy_tolerance", 0.25);
    getParam(handoff_yaw_relax_per_cycle_, "handoff_yaw_relax_per_cycle", 0.005);
    getParam(handoff_yaw_relax_cap_, "handoff_yaw_relax_cap", 0.8);
    getParam(exhausted_cusp_xy_tolerance_, "exhausted_cusp_xy_tolerance", 0.0);
    getParam(exhausted_cusp_yaw_tolerance_, "exhausted_cusp_yaw_tolerance", 0.65);
    getParam(cusp_along_track_threshold_, "cusp_along_track_threshold", 0.015);
    getParam(exhausted_along_track_threshold_, "exhausted_along_track_threshold", 0.03);
    getParam(exhausted_along_track_hold_cycles_, "exhausted_along_track_hold_cycles", 15);
    getParam(
      terminal_yaw_runway_max_tgp_length_, "terminal_yaw_runway_max_tgp_length", 0.15);
    getParam(
      terminal_yaw_runway_max_goal_dist_, "terminal_yaw_runway_max_goal_dist", 0.16);
    getParam(
      terminal_yaw_runway_min_yaw_error_, "terminal_yaw_runway_min_yaw_error", 0.18);
    // <=0 disables the escape policy (default: disabled pending oscillation latch)
    getParam(handoff_escape_after_cycles_, "handoff_escape_after_cycles", 0);
    getParam(stranded_handoff_radius_, "stranded_handoff_radius", 0.0);
    getParam(stranded_handoff_cycles_, "stranded_handoff_cycles", 20);
    stranded_handoff_cycles_ = std::max(stranded_handoff_cycles_, 1);
    handoff_yaw_relax_per_cycle_ = std::max(handoff_yaw_relax_per_cycle_, 0.0);
    handoff_yaw_relax_cap_ = std::max(
      handoff_yaw_relax_cap_, static_cast<double>(inversion_yaw_tolerance_));
    exhausted_cusp_xy_tolerance_ = std::max(exhausted_cusp_xy_tolerance_, 0.0);
    exhausted_cusp_yaw_tolerance_ = std::max(exhausted_cusp_yaw_tolerance_, 0.0);
    cusp_along_track_threshold_ = std::max(cusp_along_track_threshold_, 0.0);
    exhausted_along_track_threshold_ = std::max(exhausted_along_track_threshold_, 0.0);
    exhausted_along_track_hold_cycles_ = std::max(exhausted_along_track_hold_cycles_, 0);
    terminal_yaw_runway_max_tgp_length_ = std::max(terminal_yaw_runway_max_tgp_length_, 0.0);
    terminal_yaw_runway_max_goal_dist_ = std::max(terminal_yaw_runway_max_goal_dist_, 0.0);
    terminal_yaw_runway_min_yaw_error_ = std::max(terminal_yaw_runway_min_yaw_error_, 0.0);
    if (min_inversion_horizon_ < 0.0) {
      RCLCPP_WARN(logger_,
        "min_inversion_horizon (%.3f) is negative, disabling ghost extension",
        min_inversion_horizon_);
      min_inversion_horizon_ = 0.0;
    }
    ghost_max_extension_ = std::max(ghost_max_extension_, 0.0);
    ghost_point_spacing_ = std::max(
      ghost_point_spacing_,
      costmap_->getCostmap()->getResolution());
    ghost_min_seed_points_ = std::max(ghost_min_seed_points_, 2);
    micro_cusp_length_threshold_ = std::max(micro_cusp_length_threshold_, 0.0);
    micro_cusp_yaw_scale_ = std::max(micro_cusp_yaw_scale_, 1.0);
    terminal_stub_max_length_ = std::max(terminal_stub_max_length_, 0.0);
    terminal_stub_max_yaw_diff_ = std::max(terminal_stub_max_yaw_diff_, 0.0);
    goal_checker_xy_tolerance_ = std::max(goal_checker_xy_tolerance_, 0.0);
    inversion_locale_ = 0u;
  }
  getParam(terminal_capture_enabled_, "terminal_capture_enabled", true);
  getParam(terminal_capture_max_distance_, "terminal_capture_max_distance", 1.00);
  getParam(terminal_capture_min_distance_, "terminal_capture_min_distance", 0.22);
  getParam(terminal_capture_min_cross_track_, "terminal_capture_min_cross_track", 0.025);
  getParam(terminal_capture_replan_drift_, "terminal_capture_replan_drift", 0.020);
  getParam(terminal_capture_min_turning_r_, "terminal_capture_min_turning_r", 0.561);
  getParam(terminal_capture_point_spacing_, "terminal_capture_point_spacing", 0.03);
  getParam(terminal_capture_max_residual_y_, "terminal_capture_max_residual_y", 0.040);
  getParam(terminal_capture_pullthrough_length_, "terminal_capture_pullthrough_length", 0.10);
  getParam(
    terminal_capture_pullthrough_release_, "terminal_capture_pullthrough_release", 0.015);
  terminal_capture_max_distance_ = std::max(terminal_capture_max_distance_, 0.0);
  terminal_capture_min_distance_ = std::max(terminal_capture_min_distance_, 0.05);
  terminal_capture_min_cross_track_ = std::max(terminal_capture_min_cross_track_, 0.0);
  terminal_capture_replan_drift_ = std::max(terminal_capture_replan_drift_, 0.005);
  terminal_capture_min_turning_r_ = std::max(terminal_capture_min_turning_r_, 0.10);
  terminal_capture_point_spacing_ = std::max(
    terminal_capture_point_spacing_,
    std::max(0.01, static_cast<double>(costmap_->getCostmap()->getResolution())));
  terminal_capture_max_residual_y_ = std::max(terminal_capture_max_residual_y_, 0.0);
  terminal_capture_pullthrough_length_ = std::max(terminal_capture_pullthrough_length_, 0.0);
  terminal_capture_pullthrough_release_ = std::max(terminal_capture_pullthrough_release_, 0.0);
  has_latched_terminal_capture_ = false;
  nominal_terminal_plan_.poses.clear();
}

std::pair<nav_msgs::msg::Path, PathIterator>
PathHandler::getGlobalPlanConsideringBoundsInCostmapFrame(
  const geometry_msgs::msg::PoseStamped & global_pose)
{
  using nav2_util::geometry_utils::euclidean_distance;

  auto begin = global_plan_up_to_inversion_.poses.begin();

  // Limit the search for the closest pose up to max_robot_pose_search_dist on the path
  auto closest_pose_upper_bound =
    nav2_util::geometry_utils::first_after_integrated_distance(
    global_plan_up_to_inversion_.poses.begin(), global_plan_up_to_inversion_.poses.end(),
    max_robot_pose_search_dist_);

  // Find closest point to the robot
  auto closest_point = nav2_util::geometry_utils::min_by(
    begin, closest_pose_upper_bound,
    [&global_pose](const geometry_msgs::msg::PoseStamped & ps) {
      return euclidean_distance(global_pose, ps);
    });

  // When a post-cusp handoff has only 2 poses ([cusp_pose, goal_pose]), or on
  // the terminal segment when path inversion is enforced, retain the penultimate
  // pose when it is within prune_distance_ and inside the costmap bounds so the
  // terminal segment never collapses to a 1-pose (REAL=0.0000m) path.
  if ((two_pose_post_cusp_segment_ ||
    (enforce_path_inversion_ && inversion_locale_ == 0u)) &&
    global_plan_up_to_inversion_.poses.size() >= 2 &&
    closest_point == global_plan_up_to_inversion_.poses.end() - 1)
  {
    auto prev_point = global_plan_up_to_inversion_.poses.end() - 2;
    geometry_msgs::msg::PoseStamped prev_pose = *prev_point;
    prev_pose.header.stamp = global_pose.header.stamp;
    prev_pose.header.frame_id = global_plan_.header.frame_id;
    geometry_msgs::msg::PoseStamped prev_costmap_pose;
    unsigned int prev_mx, prev_my;
    if (euclidean_distance(prev_pose, *closest_point) <= prune_distance_ &&
      transformPose(costmap_->getGlobalFrameID(), prev_pose, prev_costmap_pose) &&
      costmap_->getCostmap()->worldToMap(
        prev_costmap_pose.pose.position.x, prev_costmap_pose.pose.position.y, prev_mx, prev_my))
    {
      closest_point = prev_point;
    }
  }

  nav_msgs::msg::Path transformed_plan;
  transformed_plan.header.frame_id = costmap_->getGlobalFrameID();
  transformed_plan.header.stamp = global_pose.header.stamp;

  auto pruned_plan_end =
    nav2_util::geometry_utils::first_after_integrated_distance(
    closest_point, global_plan_up_to_inversion_.poses.end(), prune_distance_);

  unsigned int mx, my;
  // Find the furthest relevent pose on the path to consider within costmap
  // bounds
  // Transforming it to the costmap frame in the same loop
  for (auto global_plan_pose = closest_point; global_plan_pose != pruned_plan_end;
    ++global_plan_pose)
  {
    // Transform from global plan frame to costmap frame
    geometry_msgs::msg::PoseStamped costmap_plan_pose;
    global_plan_pose->header.stamp = global_pose.header.stamp;
    global_plan_pose->header.frame_id = global_plan_.header.frame_id;
    transformPose(costmap_->getGlobalFrameID(), *global_plan_pose, costmap_plan_pose);

    // Check if pose is inside the costmap
    if (!costmap_->getCostmap()->worldToMap(
        costmap_plan_pose.pose.position.x, costmap_plan_pose.pose.position.y, mx, my))
    {
      return {transformed_plan, closest_point};
    }

    // Filling the transformed plan to return with the transformed pose
    transformed_plan.poses.push_back(costmap_plan_pose);
  }

  return {transformed_plan, closest_point};
}

geometry_msgs::msg::PoseStamped PathHandler::transformToGlobalPlanFrame(
  const geometry_msgs::msg::PoseStamped & pose)
{
  if (global_plan_up_to_inversion_.poses.empty()) {
    throw nav2_core::InvalidPath("Received plan with zero length");
  }

  geometry_msgs::msg::PoseStamped robot_pose;
  if (!transformPose(global_plan_up_to_inversion_.header.frame_id, pose, robot_pose)) {
    throw nav2_core::ControllerTFError(
            "Unable to transform robot pose into global plan's frame");
  }

  return robot_pose;
}

nav_msgs::msg::Path PathHandler::transformPath(
  const geometry_msgs::msg::PoseStamped & robot_pose)
{
  // Find relevent bounds of path to use
  geometry_msgs::msg::PoseStamped global_pose;
  try {
    global_pose = transformToGlobalPlanFrame(robot_pose);
  } catch (...) {
    stranded_cycles_ = 0u;  // a failed cycle breaks the consecutive stranded count
    throw;
  }
  auto [transformed_plan, lower_bound] = getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);

  std::optional<geometry_msgs::msg::PoseStamped> prev_pose_before_prune;
  if (lower_bound != global_plan_up_to_inversion_.poses.begin() &&
    lower_bound <= global_plan_up_to_inversion_.poses.end())
  {
    prev_pose_before_prune = *(lower_bound - 1);
  }
  prunePlan(global_plan_up_to_inversion_, lower_bound);

  bool handoff_blocked_by_along_track = false;
  double along_track_remaining = 0.0;
  if (enforce_path_inversion_ && inversion_locale_ != 0u && !terminal_stub_aligned_) {
    // Stranded handoff (#102): with the segment fully consumed and the robot beside the cusp but
    // outside the xy gate, the tolerance check can never pass and MPPI rocks on a ghost stub.
    // The cycle threshold is clamped here as well: dynamic parameter updates bypass initialize().
    bool stranded_handoff = false;
    if (stranded_handoff_radius_ > 0.0) {
      const double cusp_d = getCuspDistance(global_pose);
      // Measure the pruned segment itself, not transformed_plan: prune_distance or the costmap
      // bounds can clip the local plan to one pose while part of the approach remains.
      const bool consumed = utils::pathLength(global_plan_up_to_inversion_) < 1e-3;
      if (consumed && cusp_d > inversion_xy_tolerance_ && cusp_d <= stranded_handoff_radius_) {
        const int cycles = std::max(stranded_handoff_cycles_, 1);
        const unsigned int needed = static_cast<unsigned int>(cycles);
        if (++stranded_cycles_ >= needed) {
          stranded_handoff = true;
          RCLCPP_INFO(logger_,
            "[HANDOFF_STRANDED] cusp=%.3fm (gate %.2f, radius %.2f) after %u cycles",
            cusp_d, inversion_xy_tolerance_, stranded_handoff_radius_, stranded_cycles_);
        }
      } else {
        stranded_cycles_ = 0u;
      }
    } else {
      stranded_cycles_ = 0u;
    }
    if (stranded_handoff || isWithinInversionTolerances(
        global_pose, &handoff_blocked_by_along_track, &along_track_remaining))
    {
      stranded_cycles_ = 0u;
      // A stranded handoff short-circuits the tolerance check, which would otherwise have reset the
      // Layer 1 count; do it here so no relaxation carries into the next segment.
      handoff_blocked_cycles_ = 0u;
      const size_t post_cusp_poses =
        global_plan_.poses.size() > static_cast<size_t>(inversion_locale_) ?
        global_plan_.poses.size() - static_cast<size_t>(inversion_locale_) : 0u;
      const bool retain_cusp_pose = (post_cusp_poses == 1u && inversion_locale_ >= 1u);
      const size_t prune_idx = retain_cusp_pose ?
        static_cast<size_t>(inversion_locale_ - 1u) :
        static_cast<size_t>(inversion_locale_);
      prunePlan(global_plan_, global_plan_.poses.begin() + prune_idx);
      global_plan_up_to_inversion_ = global_plan_;
      inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
      has_cusp_original_orientation_ = false;
      if (inversion_locale_ > 0u &&
        static_cast<size_t>(inversion_locale_) <= global_plan_.poses.size())
      {
        cusp_original_orientation_ =
          global_plan_.poses[inversion_locale_ - 1u].pose.orientation;
        has_cusp_original_orientation_ = true;
      }
      two_pose_post_cusp_segment_ =
        retain_cusp_pose && (global_plan_up_to_inversion_.poses.size() == 2u);
      handoff_blocked_cycles_ = 0u;
      exhausted_along_track_cycles_ = 0u;
      has_last_exhausted_check_pose_ = false;
      has_latched_terminal_capture_ = false;
      nominal_terminal_plan_.poses.clear();
      current_segment_length_ = static_cast<double>(
        utils::pathLength(global_plan_up_to_inversion_));
      updateCurrentSegmentDirection();
      absorbTerminalGoalSnapMicroStub();
      {
        const auto & cusp_pose = global_plan_up_to_inversion_.poses.back();
        double dx = global_pose.pose.position.x - cusp_pose.pose.position.x;
        double dy = global_pose.pose.position.y - cusp_pose.pose.position.y;
        RCLCPP_INFO(logger_,
          "[HANDOFF_SEG] new segment: pts=%zu len=%.3fm inv=%u "
          "robot=(%.3f,%.3f) next_cusp=(%.3f,%.3f) dist=%.3fm",
          global_plan_up_to_inversion_.poses.size(),
          current_segment_length_, inversion_locale_,
          global_pose.pose.position.x, global_pose.pose.position.y,
          cusp_pose.pose.position.x, cusp_pose.pose.position.y,
          std::hypot(dx, dy));
      }

      // Immediate rebuild: the transformed_plan computed above is stale (from
      // pre-handoff state). Rebuild so ghost extension sees the new segment
      // direction, enabling correct reverse-heading extrapolation.
      auto [refreshed_plan, refreshed_bound] =
        getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
      transformed_plan = std::move(refreshed_plan);
      prev_pose_before_prune.reset();
      if (refreshed_bound != global_plan_up_to_inversion_.poses.begin() &&
        refreshed_bound <= global_plan_up_to_inversion_.poses.end())
      {
        prev_pose_before_prune = *(refreshed_bound - 1);
      }
      prunePlan(global_plan_up_to_inversion_, refreshed_bound);
    }
  } else {
    stranded_cycles_ = 0u;
  }

  applyTerminalCapture(global_pose, prev_pose_before_prune, transformed_plan);

  const double real_tgp_length = static_cast<double>(utils::pathLength(transformed_plan));
  double tgp_length = real_tgp_length;
  double cusp_dist = std::numeric_limits<double>::infinity();
  double ghost_heading = 0.0;

  // Layer 2 escape (notes/cusp_handoff_fix_plan.md): the handoff has been
  // blocked long enough that extrapolating the consumed segment cannot help —
  // extend along the NEXT segment's travel direction instead, so the optimizer
  // has a reachable objective that also reduces the blocking yaw error.
  double escape_heading_plan = 0.0;
  bool escape_handled = false;
  if (enforce_path_inversion_ && inversion_locale_ != 0u &&
    computeEscapeHeading(escape_heading_plan) && !transformed_plan.poses.empty())
  {
    // Rotate the plan-frame direction into the costmap frame by transforming
    // the two plan poses that define it (same convention as the bounds loop).
    geometry_msgs::msg::PoseStamped a = global_plan_.poses[inversion_locale_ - 1];
    geometry_msgs::msg::PoseStamped b = global_plan_.poses[inversion_locale_];
    a.header.stamp = global_pose.header.stamp;
    a.header.frame_id = global_plan_.header.frame_id;
    b.header.stamp = global_pose.header.stamp;
    b.header.frame_id = global_plan_.header.frame_id;
    geometry_msgs::msg::PoseStamped ta, tb;
    double escape_heading = escape_heading_plan;
    if (transformPose(costmap_->getGlobalFrameID(), a, ta) &&
      transformPose(costmap_->getGlobalFrameID(), b, tb))
    {
      escape_heading = std::atan2(
        tb.pose.position.y - ta.pose.position.y,
        tb.pose.position.x - ta.pose.position.x);
    }
    const double target_total_length = std::min(
      min_inversion_horizon_,
      real_tgp_length + std::max(ghost_point_spacing_ * 4.0, ghost_max_extension_));
    const size_t before = transformed_plan.poses.size();
    const size_t appended = appendGhostPath(
      transformed_plan, escape_heading, target_total_length, tgp_length);
    if (appended > 0) {
      // Path poses carry NOSE orientation; on a reversing post-cusp segment
      // that is travel+pi. appendGhostPath stamps the travel heading, which
      // would make PathAlignCritic(use_path_orientations) penalize exactly
      // the reverse move the escape encourages — use the post-cusp pose's
      // own orientation instead (review finding 5, 2026-09-25).
      for (size_t i = before; i < transformed_plan.poses.size(); ++i) {
        transformed_plan.poses[i].pose.orientation = tb.pose.orientation;
      }
      escape_handled = true;
      if (clock_) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
          "[HANDOFF_ESCAPE] blocked=%u appended=%zu pts along next-segment "
          "heading=%.3frad real=%.4fm total=%.4fm",
          handoff_blocked_cycles_, appended, escape_heading,
          real_tgp_length, tgp_length);
      }
    }
    // appended == 0 (escape direction blocked by cost): fall through to the
    // normal ghost branch below (review finding 8).
  }
  const bool short_unhandled_inversion =
    !escape_handled && real_tgp_length < min_inversion_horizon_ && inversion_locale_ != 0u;
  if (!escape_handled && shouldUseGhostPath(
      transformed_plan, global_pose, real_tgp_length, cusp_dist,
      ghost_heading))
  {
    // Cap ghost extension proportional to remaining segment length.
    // For micro-segments (e.g. 11cm), extending by ghost_max_extension_ (70cm)
    // overwhelms the actual maneuver — MPPI commits to a long trajectory in the
    // wrong direction.  Instead, extend by at most the real TGP length (doubling
    // the visible path) with a floor of ghost_point_spacing_ * 4 so even a fully
    // consumed segment gets a minimal gradient signal.
    const double proportional_ext = std::max(
      ghost_point_spacing_ * 4.0,
      std::min(real_tgp_length, ghost_max_extension_));
    const double target_total_length = std::min(
      min_inversion_horizon_,
      real_tgp_length + proportional_ext);
    const size_t appended = appendGhostPath(
      transformed_plan, ghost_heading, target_total_length, tgp_length);
    if (appended > 0) {
      RCLCPP_INFO(logger_,
        "[GHOST_PATH] appended %zu pts heading=%.3frad(%.0fdeg) "
        "real=%.4fm total=%.4fm cusp=%.4fm",
        appended, ghost_heading, ghost_heading * 180.0 / M_PI,
        real_tgp_length, tgp_length, cusp_dist);
    }
  } else if (short_unhandled_inversion) {
    // Log WHY ghost didn't fire when TGP is short
    cusp_dist = getCuspDistance(global_pose);
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
      "[GHOST_SKIP] tgp=%.4fm pts=%zu inv=%u cusp=%.3fm",
      real_tgp_length, transformed_plan.poses.size(), inversion_locale_, cusp_dist);
  }

  // If the along-track guard blocks handoff after pruning has reduced the local
  // plan to one pose, only add a robot-to-cusp anchor for a short, near-straight
  // setup whose singleton is the actual planner cusp. A singleton caused by a
  // local-costmap boundary may be an ordinary waypoint; do not create a chord
  // across an unseen bend.
  if (handoff_blocked_by_along_track && transformed_plan.poses.size() == 1u) {
    const bool has_active_cusp = inversion_locale_ > 0u &&
      static_cast<size_t>(inversion_locale_) <= global_plan_.poses.size();
    if (has_active_cusp) {
      const size_t cusp_index = static_cast<size_t>(inversion_locale_ - 1u);
      bool singleton_is_cusp = false;
      geometry_msgs::msg::PoseStamped cusp_costmap_pose;
      auto cusp_pose = global_plan_.poses[cusp_index];
      cusp_pose.header.stamp = global_pose.header.stamp;
      cusp_pose.header.frame_id = global_plan_.header.frame_id;
      if (transformPose(costmap_->getGlobalFrameID(), cusp_pose, cusp_costmap_pose)) {
        const auto & singleton = transformed_plan.poses.back().pose.position;
        const auto & cusp = cusp_costmap_pose.pose.position;
        singleton_is_cusp = std::hypot(singleton.x - cusp.x, singleton.y - cusp.y) <= 1e-3;
      }

      const double dx_to_cusp =
        global_plan_.poses[cusp_index].pose.position.x - global_pose.pose.position.x;
      const double dy_to_cusp =
        global_plan_.poses[cusp_index].pose.position.y - global_pose.pose.position.y;
      const double chord_length_sq = dx_to_cusp * dx_to_cusp + dy_to_cusp * dy_to_cusp;
      double max_chord_deviation = 0.0;
      bool short_path_is_straight = chord_length_sq > 1e-8 &&
        current_segment_length_ < micro_cusp_length_threshold_;
      if (short_path_is_straight) {
        size_t segment_start = cusp_index;
        double segment_length = 0.0;
        while (segment_start > 0u &&
          segment_length < current_segment_length_ - 1e-4)
        {
          const auto & current = global_plan_.poses[segment_start].pose.position;
          const auto & previous = global_plan_.poses[segment_start - 1u].pose.position;
          segment_length += std::hypot(current.x - previous.x, current.y - previous.y);
          --segment_start;
        }

        for (size_t i = segment_start; i <= cusp_index; ++i) {
          const auto & point = global_plan_.poses[i].pose.position;
          const double px = point.x - global_pose.pose.position.x;
          const double py = point.y - global_pose.pose.position.y;
          // Use distance to the infinite chord line. Clamping at either chord
          // endpoint would mistake a collinear point behind the robot for a
          // large bend, while skipping extrapolated samples can miss a real
          // off-axis setup point.
          const double projection =
            (px * dx_to_cusp + py * dy_to_cusp) / chord_length_sq;
          const double deviation_x = px - projection * dx_to_cusp;
          const double deviation_y = py - projection * dy_to_cusp;
          max_chord_deviation = std::max(
            max_chord_deviation, std::hypot(deviation_x, deviation_y));
        }
        short_path_is_straight = max_chord_deviation <= cusp_along_track_threshold_;
      }

      const auto & actual_cusp = global_plan_.poses[cusp_index].pose.position;
      const double robot_to_cusp = std::hypot(
        actual_cusp.x - global_pose.pose.position.x,
        actual_cusp.y - global_pose.pose.position.y);
      geometry_msgs::msg::PoseStamped progress_anchor;
      unsigned int anchor_mx, anchor_my;
      if (singleton_is_cusp && short_path_is_straight &&
        robot_to_cusp <= inversion_xy_tolerance_ &&
        transformPose(costmap_->getGlobalFrameID(), global_pose, progress_anchor) &&
        costmap_->getCostmap()->worldToMap(
          progress_anchor.pose.position.x, progress_anchor.pose.position.y,
          anchor_mx, anchor_my))
      {
        transformed_plan.poses.insert(transformed_plan.poses.begin(), progress_anchor);
        tgp_length = static_cast<double>(utils::pathLength(transformed_plan));
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
          "[HANDOFF_PROGRESS_TARGET] straight cusp approach anchor added "
          "along=%.3fm target=%.3fm deviation=%.3fm",
          along_track_remaining, tgp_length, max_chord_deviation);
      } else {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
          "[HANDOFF_PROGRESS_TARGET_SKIP] singleton_cusp=%d straight=%d "
          "robot_cusp=%.3fm deviation=%.3fm",
          singleton_is_cusp, short_path_is_straight, robot_to_cusp, max_chord_deviation);
      }
    } else {
      RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
        "[HANDOFF_PROGRESS_TARGET_SKIP] no active cusp index");
    }
  }

  if (transformed_plan.poses.empty()) {
    throw nav2_core::InvalidPath("Resulting plan has 0 poses in it.");
  }

  cusp_dist = std::isfinite(cusp_dist) ? cusp_dist : getCuspDistance(global_pose);
  RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
    "[PathHandler] TGP_PTS=%zu REAL=%.4fm TOTAL=%.4fm INV=%u CUSP=%.4fm",
    transformed_plan.poses.size(), real_tgp_length, tgp_length,
    inversion_locale_, cusp_dist);

  return transformed_plan;
}

bool PathHandler::shouldUseGhostPath(
  const nav_msgs::msg::Path & transformed_plan,
  const geometry_msgs::msg::PoseStamped & global_pose,
  const double tgp_length,
  double & cusp_dist,
  double & heading) const
{
  if (!enforce_path_inversion_ || min_inversion_horizon_ <= 0.0 ||
    transformed_plan.poses.empty() || tgp_length >= min_inversion_horizon_)
  {
    return false;
  }

  cusp_dist = getCuspDistance(global_pose);

  if (inversion_locale_ == 0u) {
    // Terminal segment yaw-alignment runway: once the robot has reached the
    // final poses (<= terminal_yaw_runway_max_tgp_length_) of the terminal real
    // path inside the XY goal region (cusp_dist <= terminal_yaw_runway_max_goal_dist_)
    // but still has an unaligned heading relative to the final goal pose,
    // extend a short ghost runway along the goal tangent so PathFollowCritic
    // and PathAlignCritic pull the Ackermann chassis into goal yaw tolerance
    // without cutting the approach curve early.
    if (transformed_plan.poses.size() < 2u ||
      global_plan_up_to_inversion_.poses.empty() ||
      tgp_length > terminal_yaw_runway_max_tgp_length_ ||
      cusp_dist > terminal_yaw_runway_max_goal_dist_)
    {
      return false;
    }
    const double plan_goal_yaw =
      tf2::getYaw(global_plan_up_to_inversion_.poses.back().pose.orientation);
    const double robot_yaw = tf2::getYaw(global_pose.pose.orientation);
    const double goal_yaw_err = std::fabs(
      angles::shortest_angular_distance(robot_yaw, plan_goal_yaw));
    const double min_yaw_err = has_active_goal_checker_yaw_tolerance_ ?
      std::min(
        terminal_yaw_runway_min_yaw_error_,
        active_goal_checker_yaw_tolerance_ * kTerminalRunwayGoalYawToleranceScale) :
      terminal_yaw_runway_min_yaw_error_;
    if (goal_yaw_err > min_yaw_err) {
      const double costmap_goal_yaw =
        tf2::getYaw(transformed_plan.poses.back().pose.orientation);
      heading = current_segment_is_reverse_ ?
        angles::normalize_angle(costmap_goal_yaw + M_PI) : costmap_goal_yaw;
      return true;
    }
    return false;
  }

  double heading_variance = 0.0;
  double seed_arc_length = 0.0;
  if (!estimateGhostHeading(transformed_plan, heading, heading_variance, seed_arc_length)) {
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 200,
      "[GHOST_DIAG] heading estimation failed (tgp=%.4fm pts=%zu seed_arc=%.4f)",
      tgp_length, transformed_plan.poses.size(), seed_arc_length);
    return false;
  }

  return true;
}

bool PathHandler::estimateGhostHeading(
  const nav_msgs::msg::Path & transformed_plan,
  double & heading,
  double & heading_variance,
  double & seed_arc_length) const
{
  heading = 0.0;
  heading_variance = std::numeric_limits<double>::infinity();
  seed_arc_length = 0.0;

  // Single-point fallback: use the pose orientation as heading, but detect
  // reverse segments and flip by 180°. The pose orientation is the robot's
  // nose direction; for reverse segments, travel direction is opposite.
  if (transformed_plan.poses.size() == 1) {
    double pose_yaw = tf2::getYaw(transformed_plan.poses[0].pose.orientation);

    bool is_reverse_segment = current_segment_is_reverse_;
    if (!is_reverse_segment && global_plan_up_to_inversion_.poses.size() >= 2) {
      const auto & p0 = global_plan_up_to_inversion_.poses[0].pose.position;
      const auto & p1 = global_plan_up_to_inversion_.poses[1].pose.position;
      double travel_dir = std::atan2(p1.y - p0.y, p1.x - p0.x);
      if (std::fabs(angles::shortest_angular_distance(pose_yaw, travel_dir)) > M_PI_2) {
        is_reverse_segment = true;
      }
    }
    if (is_reverse_segment) {
      pose_yaw = angles::normalize_angle(pose_yaw + M_PI);
      RCLCPP_INFO(logger_,
        "[GHOST_FLIP] Single-point reverse detected: pose=%.0fdeg -> travel=%.0fdeg",
        tf2::getYaw(transformed_plan.poses[0].pose.orientation) * 180.0 / M_PI,
        pose_yaw * 180.0 / M_PI);
    }

    heading = pose_yaw;
    heading_variance = 0.0;
    seed_arc_length = 0.0;  // Single point — no arc, but heading is valid
    return true;
  }

  // Use configured seed count when available, but accept as few as 2 points
  // for post-handoff micro-segments.
  constexpr size_t hard_floor = 2u;
  const size_t seed_points = std::min(
    static_cast<size_t>(ghost_min_seed_points_),
    std::max(hard_floor, transformed_plan.poses.size()));
  if (transformed_plan.poses.size() < hard_floor) {
    return false;
  }

  const size_t start = transformed_plan.poses.size() - seed_points;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double total_weight = 0.0;
  size_t valid_segments = 0u;

  for (size_t i = start + 1; i < transformed_plan.poses.size(); ++i) {
    const auto & prev = transformed_plan.poses[i - 1].pose.position;
    const auto & curr = transformed_plan.poses[i].pose.position;
    const double dx = curr.x - prev.x;
    const double dy = curr.y - prev.y;
    const double seg_len = std::hypot(dx, dy);
    if (seg_len < 1e-4) {
      continue;
    }

    const double seg_heading = std::atan2(dy, dx);
    seed_arc_length += seg_len;
    sum_x += std::cos(seg_heading) * seg_len;
    sum_y += std::sin(seg_heading) * seg_len;
    total_weight += seg_len;
    valid_segments++;
  }

  if (valid_segments == 0u) {
    return false;
  }

  heading = std::atan2(sum_y, sum_x);

  double weighted_variance = 0.0;
  for (size_t i = start + 1; i < transformed_plan.poses.size(); ++i) {
    const auto & prev = transformed_plan.poses[i - 1].pose.position;
    const auto & curr = transformed_plan.poses[i].pose.position;
    const double dx = curr.x - prev.x;
    const double dy = curr.y - prev.y;
    const double seg_len = std::hypot(dx, dy);
    if (seg_len < 1e-4) {
      continue;
    }

    const double seg_heading = std::atan2(dy, dx);
    const double diff = angles::shortest_angular_distance(heading, seg_heading);
    weighted_variance += diff * diff * seg_len;
  }
  heading_variance = weighted_variance / total_weight;

  return true;
}

size_t PathHandler::appendGhostPath(
  nav_msgs::msg::Path & transformed_plan,
  const double heading,
  const double target_total_length,
  double & total_length) const
{
  if (transformed_plan.poses.empty()) {
    return 0u;
  }

  if (target_total_length <= total_length || ghost_point_spacing_ <= 0.0) {
    return 0u;
  }

  const double requested_extension = std::min(
    target_total_length - total_length,
    ghost_max_extension_);
  if (requested_extension <= 1e-4) {
    return 0u;
  }

  size_t appended = 0u;
  double appended_length = 0.0;
  auto next_pose = transformed_plan.poses.back();
  // Preserve the robot nose orientation on reverse segments (where travel heading
  // is nose + pi) and whenever terminal_stub_aligned_ has stamped the final goal
  // orientation onto the cusp pose.
  const double endpoint_yaw = tf2::getYaw(next_pose.pose.orientation);
  if (!terminal_stub_aligned_ && inversion_locale_ != 0u &&
    std::fabs(angles::shortest_angular_distance(endpoint_yaw, heading)) <= M_PI_2)
  {
    next_pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(heading);
  }

  while (appended_length + 1e-4 < requested_extension) {
    const double step = std::min(
      ghost_point_spacing_,
      requested_extension - appended_length);
    next_pose.pose.position.x += std::cos(heading) * step;
    next_pose.pose.position.y += std::sin(heading) * step;

    if (!isTraversableCostmapPose(next_pose)) {
      break;
    }

    transformed_plan.poses.push_back(next_pose);
    appended++;
    appended_length += step;
  }

  total_length += appended_length;
  return appended;
}

bool PathHandler::isTraversableCostmapPose(const geometry_msgs::msg::PoseStamped & pose) const
{
  unsigned int mx, my;
  auto * costmap = costmap_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));
  if (!costmap->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
    return false;
  }

  const unsigned char cost = costmap->getCost(mx, my);
  return cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
}

double PathHandler::getCuspDistance(const geometry_msgs::msg::PoseStamped & global_pose) const
{
  if (global_plan_up_to_inversion_.poses.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const auto & cusp = global_plan_up_to_inversion_.poses.back();
  return std::hypot(
    global_pose.pose.position.x - cusp.pose.position.x,
    global_pose.pose.position.y - cusp.pose.position.y);
}

bool PathHandler::transformPose(
  const std::string & frame, const geometry_msgs::msg::PoseStamped & in_pose,
  geometry_msgs::msg::PoseStamped & out_pose) const
{
  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf_buffer_->transform(
      in_pose, out_pose, frame,
      tf2::durationFromSec(transform_tolerance_));
    out_pose.header.frame_id = frame;
    return true;
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(logger_, "Exception in transformPose: %s", ex.what());
  }
  return false;
}

double PathHandler::getMaxCostmapDist()
{
  const auto & costmap = costmap_->getCostmap();
  return static_cast<double>(std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY())) *
         costmap->getResolution() * 0.50;
}

void PathHandler::setPath(const nav_msgs::msg::Path & plan)
{
  bool had_prev_cusp = false;
  geometry_msgs::msg::Pose prev_cusp_pose{};
  if (enforce_path_inversion_ && inversion_locale_ > 0u &&
    !global_plan_up_to_inversion_.poses.empty())
  {
    had_prev_cusp = true;
    prev_cusp_pose = global_plan_up_to_inversion_.poses.back().pose;
    if (has_cusp_original_orientation_) {
      prev_cusp_pose.orientation = cusp_original_orientation_;
    }
  }

  global_plan_ = plan;
  stranded_cycles_ = 0u;
  global_plan_up_to_inversion_ = global_plan_;
  terminal_stub_aligned_ = false;
  two_pose_post_cusp_segment_ = false;
  has_cusp_original_orientation_ = false;
  has_latched_terminal_capture_ = false;
  nominal_terminal_plan_.poses.clear();
  bool same_active_cusp = false;
  if (enforce_path_inversion_) {
    inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
    if (inversion_locale_ > 0u &&
      static_cast<size_t>(inversion_locale_) <= global_plan_.poses.size())
    {
      cusp_original_orientation_ =
        global_plan_.poses[inversion_locale_ - 1u].pose.orientation;
      has_cusp_original_orientation_ = true;
      if (had_prev_cusp && !global_plan_up_to_inversion_.poses.empty()) {
        const auto & new_cusp = global_plan_up_to_inversion_.poses.back().pose;
        const double cusp_dx = new_cusp.position.x - prev_cusp_pose.position.x;
        const double cusp_dy = new_cusp.position.y - prev_cusp_pose.position.y;
        const double cusp_dyaw = std::fabs(angles::shortest_angular_distance(
          tf2::getYaw(new_cusp.orientation),
          tf2::getYaw(prev_cusp_pose.orientation)));
        same_active_cusp = (std::hypot(cusp_dx, cusp_dy) <= 0.05) && (cusp_dyaw <= 0.10);
      }
    }
    current_segment_length_ = static_cast<double>(
      utils::pathLength(global_plan_up_to_inversion_));
    updateCurrentSegmentDirection();
    absorbTerminalGoalSnapMicroStub();
  } else {
    inversion_locale_ = 0u;
    current_segment_length_ = static_cast<double>(
      utils::pathLength(global_plan_up_to_inversion_));
    updateCurrentSegmentDirection();
  }
  if (!same_active_cusp) {
    handoff_blocked_cycles_ = 0u;
    exhausted_along_track_cycles_ = 0u;
    has_last_exhausted_check_pose_ = false;
  }
}

void PathHandler::updateCurrentSegmentDirection()
{
  current_segment_is_reverse_ = false;
  const auto & poses = global_plan_up_to_inversion_.poses;
  if (poses.size() < 2u) {
    return;
  }
  for (size_t i = poses.size() - 1u; i >= 1u; --i) {
    const auto & p0 = poses[i - 1u].pose.position;
    const auto & p1 = poses[i].pose.position;
    const double dx = p1.x - p0.x;
    const double dy = p1.y - p0.y;
    if (std::hypot(dx, dy) < 1e-4) {
      continue;
    }
    const double travel_dir = std::atan2(dy, dx);
    const double pose_yaw = tf2::getYaw(poses[i - 1u].pose.orientation);
    current_segment_is_reverse_ =
      std::fabs(angles::shortest_angular_distance(pose_yaw, travel_dir)) > M_PI_2;
    return;
  }
}

void PathHandler::setGoalChecker(nav2_core::GoalChecker * goal_checker)
{
  bool has_tol = false;
  double xy_tol = 0.0;
  bool has_yaw_tol = false;
  double yaw_tol = 0.0;
  if (goal_checker != nullptr) {
    geometry_msgs::msg::Pose pose_tol;
    geometry_msgs::msg::Twist vel_tol;
    if (goal_checker->getTolerances(pose_tol, vel_tol)) {
      const double cand = std::min(pose_tol.position.x, pose_tol.position.y);
      if (std::isfinite(cand) && cand > 0.0) {
        has_tol = true;
        xy_tol = cand;
      }
      const double cand_yaw = std::fabs(tf2::getYaw(pose_tol.orientation));
      if (std::isfinite(cand_yaw) && cand_yaw > 0.0) {
        has_yaw_tol = true;
        yaw_tol = cand_yaw;
      }
    }
  }
  has_active_goal_checker_yaw_tolerance_ = has_yaw_tol;
  active_goal_checker_yaw_tolerance_ = yaw_tol;
  if (has_tol != has_active_goal_checker_xy_tolerance_ ||
    (has_tol && std::fabs(xy_tol - active_goal_checker_xy_tolerance_) > 1e-9))
  {
    has_active_goal_checker_xy_tolerance_ = has_tol;
    active_goal_checker_xy_tolerance_ = xy_tol;
    if (enforce_path_inversion_ && inversion_locale_ != 0u) {
      absorbTerminalGoalSnapMicroStub();
    }
  }
}

void PathHandler::absorbTerminalGoalSnapMicroStub()
{
  terminal_stub_aligned_ = false;
  if (!enforce_path_inversion_ || inversion_locale_ == 0u ||
    static_cast<size_t>(inversion_locale_) >= global_plan_.poses.size())
  {
    return;
  }

  if (has_cusp_original_orientation_) {
    global_plan_.poses[inversion_locale_ - 1u].pose.orientation = cusp_original_orientation_;
    if (!global_plan_up_to_inversion_.poses.empty()) {
      global_plan_up_to_inversion_.poses.back().pose.orientation = cusp_original_orientation_;
    }
  }

  nav_msgs::msg::Path post_cusp_tail;
  post_cusp_tail.header = global_plan_.header;
  post_cusp_tail.poses.assign(
    global_plan_.poses.begin() + (inversion_locale_ - 1u),
    global_plan_.poses.end());
  nav_msgs::msg::Path tail_up_to_inv = post_cusp_tail;
  const unsigned int next_inv = utils::removePosesAfterFirstInversion(tail_up_to_inv);
  const double tail_length = static_cast<double>(utils::pathLength(post_cusp_tail));
  const auto & cusp_pose = global_plan_.poses[inversion_locale_ - 1u];
  const auto & goal_pose = global_plan_.poses.back();
  const double goal_dist = std::hypot(
    goal_pose.pose.position.x - cusp_pose.pose.position.x,
    goal_pose.pose.position.y - cusp_pose.pose.position.y);
  const double goal_yaw_diff = std::fabs(angles::shortest_angular_distance(
    tf2::getYaw(cusp_pose.pose.orientation),
    tf2::getYaw(goal_pose.pose.orientation)));

  const double effective_goal_xy_tol =
    has_active_goal_checker_xy_tolerance_ ?
    active_goal_checker_xy_tolerance_ :
    goal_checker_xy_tolerance_;
  const double max_stub_len = std::min(terminal_stub_max_length_, micro_cusp_length_threshold_);
  const double max_goal_dist = std::min(
    max_stub_len,
    std::min(static_cast<double>(inversion_xy_tolerance_), effective_goal_xy_tol));
  const double max_yaw_diff = std::min(
    terminal_stub_max_yaw_diff_,
    static_cast<double>(inversion_yaw_tolerance_) * micro_cusp_yaw_scale_);

  if (next_inv == 0u &&
    tail_length <= max_stub_len &&
    tail_length < current_segment_length_ &&
    goal_dist <= max_goal_dist &&
    goal_yaw_diff <= max_yaw_diff)
  {
    // Stamp the final goal's orientation onto the cusp pose while keeping the
    // cusp active (inversion_locale_ != 0u) so ghost-path continuation remains
    // active during the approach and aligns the robot to the final goal yaw,
    // and inhibit handoff pruning into the micro-stub via terminal_stub_aligned_.
    const auto goal_orientation = goal_pose.pose.orientation;
    global_plan_.poses[inversion_locale_ - 1u].pose.orientation = goal_orientation;
    if (!global_plan_up_to_inversion_.poses.empty()) {
      global_plan_up_to_inversion_.poses.back().pose.orientation = goal_orientation;
    }
    terminal_stub_aligned_ = true;
    RCLCPP_DEBUG(logger_,
      "[TERMINAL_STUB_ALIGNED] Aligned cusp yaw to goal over %.3fm terminal stub "
      "(%zu pts, goal_dist=%.3fm, yaw_diff=%.3frad); active segment len=%.3fm",
      tail_length, post_cusp_tail.poses.size() - 1u, goal_dist, goal_yaw_diff,
      current_segment_length_);
  }
}

nav_msgs::msg::Path & PathHandler::getPath() {return global_plan_;}

void PathHandler::prunePlan(nav_msgs::msg::Path & plan, const PathIterator end)
{
  plan.poses.erase(plan.poses.begin(), end);
}

geometry_msgs::msg::PoseStamped PathHandler::getTransformedGoal(
  const builtin_interfaces::msg::Time & stamp)
{
  auto goal = global_plan_.poses.back();
  goal.header.frame_id = global_plan_.header.frame_id;
  goal.header.stamp = stamp;
  if (goal.header.frame_id.empty()) {
    throw nav2_core::ControllerTFError("Goal pose has an empty frame_id");
  }
  geometry_msgs::msg::PoseStamped transformed_goal;
  if (!transformPose(costmap_->getGlobalFrameID(), goal, transformed_goal)) {
    throw nav2_core::ControllerTFError("Unable to transform goal pose into costmap frame");
  }
  return transformed_goal;
}

bool PathHandler::isWithinInversionTolerances(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  bool * handoff_blocked_by_along_track,
  double * along_track_remaining_output) const
{
  if (handoff_blocked_by_along_track) {
    *handoff_blocked_by_along_track = false;
  }
  if (along_track_remaining_output) {
    *along_track_remaining_output = 0.0;
  }

  // Keep full path if we are within tolerance of the inversion pose
  const auto last_pose = global_plan_up_to_inversion_.poses.back();
  float distance = hypotf(
    robot_pose.pose.position.x - last_pose.pose.position.x,
    robot_pose.pose.position.y - last_pose.pose.position.y);

  float angle_distance = angles::shortest_angular_distance(
    tf2::getYaw(robot_pose.pose.orientation),
    tf2::getYaw(last_pose.pose.orientation));

  // Relax yaw tolerance for micro-cusps: segments shorter than the threshold
  // can need relaxed yaw alignment at arrival. Their length alone does not
  // make them disposable: a fresh short reverse leg may be a positioning stroke.
  // For intermediate short segments (15-30 cm), retain micro-cusp yaw
  // relaxation with <= 2 remaining poses. The along-track guard below
  // separately prevents handoff while a positioning leg remains unconsumed.
  const bool is_micro_cusp =
    (current_segment_length_ < micro_cusp_length_threshold_) &&
    (current_segment_length_ <= static_cast<double>(inversion_xy_tolerance_) ||
    global_plan_up_to_inversion_.poses.size() <= 2u);
  const double effective_yaw_tol = is_micro_cusp ?
    inversion_yaw_tolerance_ * micro_cusp_yaw_scale_ :
    inversion_yaw_tolerance_;

  // Layer 1 (cusp handoff deadlock, notes/cusp_handoff_fix_plan.md): each
  // consecutive blocked cycle (xy met, yaw not) widens the yaw gate toward a
  // cap, so a heading error the robot cannot correct in place cannot block the
  // handoff forever. Safe to relax now that PathAlignCritic with path
  // orientations supplies the corrective signal after handoff.
  // Floor at effective_yaw_tol: with a large micro_cusp_yaw_scale the scaled
  // tolerance can exceed the cap, and min(cap, ...) alone would TIGHTEN the
  // micro gate (review finding, 2026-09-25).
  const double relaxed_yaw_tol = std::max(
    effective_yaw_tol,
    std::min(
      handoff_yaw_relax_cap_,
      effective_yaw_tol +
      handoff_yaw_relax_per_cycle_ * static_cast<double>(handoff_blocked_cycles_)));

  // When the pre-cusp segment has been completely consumed by prunePlan()
  // (<= 1 pose remaining in global_plan_up_to_inversion_) and is not a
  // terminal-stub-aligned goal pose, distinguish genuine lateral cusp
  // exhaustion (robot has arrived at/abeam of the cusp or lateral offset
  // dominates over remaining along-track distance inside inversion_xy_tolerance_)
  // from premature longitudinal pruning on a sparse path (where the robot is
  // still longitudinally short of the cusp along-track).
  const double cusp_yaw = tf2::getYaw(last_pose.pose.orientation);
  const double travel_yaw = current_segment_is_reverse_ ?
    angles::normalize_angle(cusp_yaw + M_PI) : cusp_yaw;
  const double dx_to_cusp = last_pose.pose.position.x - robot_pose.pose.position.x;
  const double dy_to_cusp = last_pose.pose.position.y - robot_pose.pose.position.y;
  const double along_track_remaining =
    dx_to_cusp * std::cos(travel_yaw) + dy_to_cusp * std::sin(travel_yaw);
  const double cross_track_offset = std::fabs(
    -dx_to_cusp * std::sin(travel_yaw) + dy_to_cusp * std::cos(travel_yaw));

  const bool is_segment_exhausted =
    (global_plan_up_to_inversion_.poses.size() <= 1u) && !terminal_stub_aligned_;
  const bool is_lateral_exhausted = is_segment_exhausted &&
    (along_track_remaining <= exhausted_along_track_threshold_ ||
    (along_track_remaining <= static_cast<double>(inversion_xy_tolerance_) &&
    cross_track_offset > along_track_remaining));
  const double active_xy_tol = is_lateral_exhausted ?
    std::max(static_cast<double>(inversion_xy_tolerance_), exhausted_cusp_xy_tolerance_) :
    static_cast<double>(inversion_xy_tolerance_);
  const double active_yaw_tol = is_lateral_exhausted ?
    std::max(relaxed_yaw_tol, exhausted_cusp_yaw_tolerance_) :
    relaxed_yaw_tol;

  // Count actual motion steps (not stationary calls) when laterally offset on
  // an exhausted segment with remaining along-track distance > threshold.
  if (is_lateral_exhausted && distance <= active_xy_tol &&
    along_track_remaining > exhausted_along_track_threshold_)
  {
    const double robot_x = robot_pose.pose.position.x;
    const double robot_y = robot_pose.pose.position.y;
    const double robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
    if (!has_last_exhausted_check_pose_) {
      has_last_exhausted_check_pose_ = true;
      last_exhausted_check_x_ = robot_x;
      last_exhausted_check_y_ = robot_y;
      last_exhausted_check_yaw_ = robot_yaw;
      exhausted_along_track_cycles_ = 1u;
    } else {
      const double step_dist = std::hypot(
        robot_x - last_exhausted_check_x_, robot_y - last_exhausted_check_y_);
      const double step_yaw = std::fabs(
        angles::shortest_angular_distance(robot_yaw, last_exhausted_check_yaw_));
      if (step_dist >= 1e-3 || step_yaw >= 1e-3) {
        last_exhausted_check_x_ = robot_x;
        last_exhausted_check_y_ = robot_y;
        last_exhausted_check_yaw_ = robot_yaw;
        ++exhausted_along_track_cycles_;
      }
    }
  } else {
    exhausted_along_track_cycles_ = 0u;
    has_last_exhausted_check_pose_ = false;
  }
  // Protect unconsumed positioning strokes even when the entire segment fits
  // inside the XY arrival radius. Otherwise a replan's short reverse setup can
  // be handed off at its starting pose, before the optimizer ever sees it.
  const bool premature_along_track =
    ((!is_segment_exhausted && along_track_remaining > cusp_along_track_threshold_) ||
    (is_segment_exhausted && !is_lateral_exhausted &&
    along_track_remaining > cusp_along_track_threshold_) ||
    (is_lateral_exhausted && along_track_remaining > exhausted_along_track_threshold_ &&
    exhausted_along_track_cycles_ <=
    static_cast<unsigned int>(std::max(exhausted_along_track_hold_cycles_, 0))));
  if (handoff_blocked_by_along_track) {
    *handoff_blocked_by_along_track = premature_along_track;
  }
  if (along_track_remaining_output) {
    *along_track_remaining_output = along_track_remaining;
  }

  const bool standard_result = !premature_along_track &&
    distance <= inversion_xy_tolerance_ &&
    std::fabs(angle_distance) <= relaxed_yaw_tol;
  bool result = !premature_along_track &&
    distance <= active_xy_tol &&
    std::fabs(angle_distance) <= active_yaw_tol;

  if (premature_along_track && distance <= active_xy_tol &&
    std::fabs(angle_distance) <= active_yaw_tol && clock_)
  {
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
      "[HANDOFF_ALONG_TRACK_BLOCKED] xy=%.4fm yaw=%.3frad "
      "along=%.4fm cross=%.4fm seg_len=%.3fm exhausted=%d",
      distance, std::fabs(angle_distance), along_track_remaining,
      cross_track_offset, current_segment_length_, is_segment_exhausted);
  }

  if (!result && distance <= active_xy_tol && std::fabs(angle_distance) > active_yaw_tol) {
    ++handoff_blocked_cycles_;
    // clock_ is only set by initialize(); guard so the tolerance check stays
    // callable on a bare PathHandler (unit tests construct it without a node —
    // dereferencing the null clock here was a SIGSEGV in path_handler_test).
    if (clock_) {
      RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
        "[HANDOFF_BLOCKED] xy=%.4fm(<%.2f) yaw=%.3frad(need<%.2f%s) "
        "seg_len=%.3fm blocked=%u",
        distance, active_xy_tol,
        std::fabs(angle_distance), active_yaw_tol,
        is_micro_cusp ? " MICRO" : (is_lateral_exhausted ? " EXHAUSTED" : ""),
        current_segment_length_, handoff_blocked_cycles_);
    }
  } else if (result || distance > active_xy_tol) {
    // Unlocked, or left the xy tolerance: relaxation must not bank across
    // separate approaches.
    handoff_blocked_cycles_ = 0u;
    if (result && is_lateral_exhausted && !standard_result) {
      RCLCPP_INFO(logger_,
        "[HANDOFF_EXHAUSTED] xy=%.4fm(tol=%.2f) yaw=%.3frad(tol=%.2f) seg_len=%.3fm",
        distance, active_xy_tol, std::fabs(angle_distance), active_yaw_tol,
        current_segment_length_);
    } else if (result && is_micro_cusp) {
      RCLCPP_INFO(logger_,
        "[HANDOFF_MICRO] xy=%.4fm yaw=%.3frad(tol=%.2f) seg_len=%.3fm",
        distance, std::fabs(angle_distance), active_yaw_tol,
        current_segment_length_);
    }
  }
  return result;
}

bool PathHandler::computeEscapeHeading(double & heading) const
{
  // Layer 2: after enough consecutive blocked cycles, offer the optimizer the
  // FUTURE — the travel direction of the segment after the cusp — instead of
  // more extrapolation of the segment it has already consumed. Plan frame.
  if (handoff_escape_after_cycles_ <= 0 ||  // policy disabled
    inversion_locale_ == 0u ||
    handoff_blocked_cycles_ < static_cast<unsigned int>(handoff_escape_after_cycles_) ||
    static_cast<size_t>(inversion_locale_) >= global_plan_.poses.size())
  {
    return false;
  }
  const auto & cusp = global_plan_.poses[inversion_locale_ - 1].pose.position;
  const auto & next = global_plan_.poses[inversion_locale_].pose.position;
  const double dx = next.x - cusp.x;
  const double dy = next.y - cusp.y;
  if (std::hypot(dx, dy) < 1e-6) {
    return false;
  }
  heading = std::atan2(dy, dx);
  return true;
}

bool PathHandler::isFootprintCollisionFreeInPlanFrame(
  const geometry_msgs::msg::PoseStamped & plan_pose) const
{
  if (!costmap_ || !costmap_->getCostmap()) {
    return false;
  }
  geometry_msgs::msg::PoseStamped costmap_pose;
  if (!transformPose(costmap_->getGlobalFrameID(), plan_pose, costmap_pose)) {
    return false;
  }

  auto * costmap = costmap_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));

  const double wx = costmap_pose.pose.position.x;
  const double wy = costmap_pose.pose.position.y;
  const double wyaw = tf2::getYaw(costmap_pose.pose.orientation);
  unsigned int mx = 0u, my = 0u;
  if (!costmap->worldToMap(wx, wy, mx, my)) {
    return false;
  }

  const unsigned char center_cost = costmap->getCost(mx, my);
  if (center_cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
    return false;
  }

  std::vector<geometry_msgs::msg::Point> footprint;
  if (costmap_->getLayeredCostmap()) {
    footprint = costmap_->getLayeredCostmap()->getFootprint();
  }
  if (footprint.empty()) {
    footprint = costmap_->getRobotFootprint();
  }
  if (costmap_->getUseRadius() || footprint.size() < 3u) {
    return center_cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
  }

  const double cos_th = std::cos(wyaw);
  const double sin_th = std::sin(wyaw);
  for (const auto & pt : footprint) {
    const double vx = wx + (pt.x * cos_th - pt.y * sin_th);
    const double vy = wy + (pt.x * sin_th + pt.y * cos_th);
    unsigned int vmx = 0u, vmy = 0u;
    if (!costmap->worldToMap(vx, vy, vmx, vmy)) {
      return false;
    }
  }

  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap);
  const double fp_cost = checker.footprintCostAtPose(wx, wy, wyaw, footprint);
  if (fp_cost < 0.0 ||
    fp_cost >= static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE))
  {
    return false;
  }
  return true;
}

void PathHandler::applyTerminalCapture(
  const geometry_msgs::msg::PoseStamped & global_pose,
  std::optional<geometry_msgs::msg::PoseStamped> & prev_pose_before_prune,
  nav_msgs::msg::Path & transformed_plan)
{
  const bool capture_params_changed = has_latched_terminal_capture_ && (
    std::fabs(terminal_capture_max_distance_ - latched_capture_max_distance_) > 1e-9 ||
    std::fabs(terminal_capture_min_distance_ - latched_capture_min_distance_) > 1e-9 ||
    std::fabs(terminal_capture_min_cross_track_ - latched_capture_min_cross_track_) > 1e-9 ||
    std::fabs(terminal_capture_replan_drift_ - latched_capture_replan_drift_) > 1e-9 ||
    std::fabs(terminal_capture_min_turning_r_ - latched_capture_min_turning_r_) > 1e-9 ||
    std::fabs(terminal_capture_point_spacing_ - latched_capture_point_spacing_) > 1e-9 ||
    std::fabs(terminal_capture_max_residual_y_ - latched_capture_max_residual_y_) > 1e-9);
  if ((!terminal_capture_enabled_ || capture_params_changed) && has_latched_terminal_capture_) {
    if (!nominal_terminal_plan_.poses.empty()) {
      global_plan_up_to_inversion_ = nominal_terminal_plan_;
    }
    has_latched_terminal_capture_ = false;
    auto [restored_plan, restored_bound] =
      getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
    transformed_plan = std::move(restored_plan);
    prev_pose_before_prune.reset();
    if (restored_bound != global_plan_up_to_inversion_.poses.begin() &&
      restored_bound <= global_plan_up_to_inversion_.poses.end())
    {
      prev_pose_before_prune = *(restored_bound - 1);
    }
    prunePlan(global_plan_up_to_inversion_, restored_bound);
  }

  if (!terminal_capture_enabled_ ||
    inversion_locale_ != 0u ||
    current_segment_length_ < 0.40 ||
    global_plan_up_to_inversion_.poses.empty())
  {
    return;
  }

  const double min_dist = std::max(terminal_capture_min_distance_, 0.05);
  const double max_dist = std::max(terminal_capture_max_distance_, 0.0);
  const double min_cross_track = std::max(terminal_capture_min_cross_track_, 0.0);
  const double replan_drift = std::max(terminal_capture_replan_drift_, 0.005);

  auto minDistToPlan = [this](
    const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::PoseStamped & pose,
    const std::optional<geometry_msgs::msg::PoseStamped> & prev_pose) -> double
    {
      if (plan.poses.empty()) {
        return std::numeric_limits<double>::infinity();
      }
      const double rx = pose.pose.position.x;
      const double ry = pose.pose.position.y;
      double best = std::numeric_limits<double>::infinity();

      auto segDist = [&](
        const geometry_msgs::msg::PoseStamped & a,
        const geometry_msgs::msg::PoseStamped & b,
        double min_t_ext) -> double
        {
          const double ax = a.pose.position.x;
          const double ay = a.pose.position.y;
          const double bx = b.pose.position.x;
          const double by = b.pose.position.y;
          const double dx = bx - ax;
          const double dy = by - ay;
          const double len_sq = dx * dx + dy * dy;
          double t = 0.0;
          if (len_sq > 1e-10) {
            const double min_t = (min_t_ext < 0.0) ? (min_t_ext / std::sqrt(len_sq)) : 0.0;
            t = std::clamp(((rx - ax) * dx + (ry - ay) * dy) / len_sq, min_t, 1.0);
          }
          const double px = ax + t * dx;
          const double py = ay + t * dy;
          return std::hypot(rx - px, ry - py);
        };

      if (prev_pose.has_value()) {
        best = std::min(best, segDist(*prev_pose, plan.poses.front(), 0.0));
      }
      for (size_t i = 0; i + 1 < plan.poses.size(); ++i) {
        const double min_t_ext = (i == 0u && !prev_pose.has_value()) ? -0.10 : 0.0;
        best = std::min(best, segDist(plan.poses[i], plan.poses[i + 1], min_t_ext));
      }
      if (!std::isfinite(best)) {
        const double gear = current_segment_is_reverse_ ? -1.0 : 1.0;
        const double goal_nose_yaw = tf2::getYaw(plan.poses.front().pose.orientation);
        const double goal_travel_yaw =
          gear < 0.0 ? angles::normalize_angle(goal_nose_yaw + M_PI) : goal_nose_yaw;
        const double dx = rx - plan.poses.front().pose.position.x;
        const double dy = ry - plan.poses.front().pose.position.y;
        best = std::fabs(-dx * std::sin(goal_travel_yaw) + dy * std::cos(goal_travel_yaw));
      }
      return best;
    };

  auto goal_pose = (has_latched_terminal_capture_ && !nominal_terminal_plan_.poses.empty()) ?
    nominal_terminal_plan_.poses.back() : global_plan_up_to_inversion_.poses.back();
  goal_pose.header.frame_id = global_plan_.header.frame_id;
  goal_pose.header.stamp = global_pose.header.stamp;
  const double goal_dist = std::hypot(
    global_pose.pose.position.x - goal_pose.pose.position.x,
    global_pose.pose.position.y - goal_pose.pose.position.y);
  double cross_track_dist = minDistToPlan(
    global_plan_up_to_inversion_, global_pose, prev_pose_before_prune);

  bool skip_initial_latch = false;
  if (has_latched_terminal_capture_) {
    bool latched_blocked = false;
    std::vector<geometry_msgs::msg::PoseStamped> check_poses;
    check_poses.reserve(global_plan_up_to_inversion_.poses.size() + 1u);
    if (prev_pose_before_prune.has_value()) {
      check_poses.push_back(*prev_pose_before_prune);
    }
    for (const auto & p : global_plan_up_to_inversion_.poses) {
      check_poses.push_back(p);
    }
    for (size_t i = 0; i < check_poses.size() && !latched_blocked; ++i) {
      auto p = check_poses[i];
      p.header.frame_id = global_plan_.header.frame_id;
      p.header.stamp = global_pose.header.stamp;
      if (!isFootprintCollisionFreeInPlanFrame(p)) {
        latched_blocked = true;
        break;
      }
      if (i + 1 < check_poses.size()) {
        const auto & a = check_poses[i].pose;
        const auto & b = check_poses[i + 1].pose;
        const double seg_len = std::hypot(
          b.position.x - a.position.x, b.position.y - a.position.y);
        if (seg_len > 0.025) {
          const size_t steps = static_cast<size_t>(std::ceil(seg_len / 0.025));
          const double yaw_a = tf2::getYaw(a.orientation);
          const double dyaw = angles::shortest_angular_distance(
            yaw_a, tf2::getYaw(b.orientation));
          for (size_t s = 1; s < steps; ++s) {
            const double alpha = static_cast<double>(s) / static_cast<double>(steps);
            geometry_msgs::msg::PoseStamped mid;
            mid.header.frame_id = global_plan_.header.frame_id;
            mid.header.stamp = global_pose.header.stamp;
            mid.pose.position.x = a.position.x + alpha * (b.position.x - a.position.x);
            mid.pose.position.y = a.position.y + alpha * (b.position.y - a.position.y);
            mid.pose.position.z = a.position.z;
            mid.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(
              angles::normalize_angle(yaw_a + alpha * dyaw));
            if (!isFootprintCollisionFreeInPlanFrame(mid)) {
              latched_blocked = true;
              break;
            }
          }
        }
      }
    }
    if (latched_blocked) {
      if (!nominal_terminal_plan_.poses.empty()) {
        global_plan_up_to_inversion_ = nominal_terminal_plan_;
      }
      has_latched_terminal_capture_ = false;
      auto [restored_plan, restored_bound] =
        getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
      transformed_plan = std::move(restored_plan);
      prev_pose_before_prune.reset();
      if (restored_bound != global_plan_up_to_inversion_.poses.begin() &&
        restored_bound <= global_plan_up_to_inversion_.poses.end())
      {
        prev_pose_before_prune = *(restored_bound - 1);
      }
      prunePlan(global_plan_up_to_inversion_, restored_bound);
      cross_track_dist = minDistToPlan(
        global_plan_up_to_inversion_, global_pose, prev_pose_before_prune);
    } else if (cross_track_dist > replan_drift) {
      nav_msgs::msg::Path re_capture_plan;
      if (goal_dist >= min_dist &&
        goal_dist <= max_dist &&
        solveTerminalCapturePath(global_pose, goal_pose, re_capture_plan))
      {
        global_plan_up_to_inversion_ = std::move(re_capture_plan);
        latched_capture_max_distance_ = terminal_capture_max_distance_;
        latched_capture_min_distance_ = terminal_capture_min_distance_;
        latched_capture_min_cross_track_ = terminal_capture_min_cross_track_;
        latched_capture_replan_drift_ = terminal_capture_replan_drift_;
        latched_capture_min_turning_r_ = terminal_capture_min_turning_r_;
        latched_capture_point_spacing_ = terminal_capture_point_spacing_;
        latched_capture_max_residual_y_ = terminal_capture_max_residual_y_;
        auto [cap_plan, cap_bound] =
          getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
        transformed_plan = std::move(cap_plan);
        prev_pose_before_prune.reset();
        if (cap_bound != global_plan_up_to_inversion_.poses.begin() &&
          cap_bound <= global_plan_up_to_inversion_.poses.end())
        {
          prev_pose_before_prune = *(cap_bound - 1);
        }
        prunePlan(global_plan_up_to_inversion_, cap_bound);
        if (clock_) {
          RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
            "[TERMINAL_CAPTURE] Re-latched 1-gear capture path after drift: pts=%zu "
            "goal_dist=%.3fm cross_track=%.3fm",
            global_plan_up_to_inversion_.poses.size(), goal_dist, cross_track_dist);
        }
      } else {
        if (!nominal_terminal_plan_.poses.empty()) {
          global_plan_up_to_inversion_ = nominal_terminal_plan_;
        }
        has_latched_terminal_capture_ = false;
        skip_initial_latch = true;
        auto [restored_plan, restored_bound] =
          getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
        transformed_plan = std::move(restored_plan);
        prev_pose_before_prune.reset();
        if (restored_bound != global_plan_up_to_inversion_.poses.begin() &&
          restored_bound <= global_plan_up_to_inversion_.poses.end())
        {
          prev_pose_before_prune = *(restored_bound - 1);
        }
        prunePlan(global_plan_up_to_inversion_, restored_bound);
      }
    }
  }

  if (!has_latched_terminal_capture_ &&
    !skip_initial_latch &&
    goal_dist >= min_dist &&
    goal_dist <= max_dist &&
    cross_track_dist >= min_cross_track)
  {
    nav_msgs::msg::Path capture_plan;
    if (solveTerminalCapturePath(global_pose, goal_pose, capture_plan)) {
      nominal_terminal_plan_ = global_plan_up_to_inversion_;
      if (prev_pose_before_prune.has_value()) {
        nominal_terminal_plan_.poses.insert(
          nominal_terminal_plan_.poses.begin(), *prev_pose_before_prune);
      }
      global_plan_up_to_inversion_ = std::move(capture_plan);
      has_latched_terminal_capture_ = true;
      latched_capture_max_distance_ = terminal_capture_max_distance_;
      latched_capture_min_distance_ = terminal_capture_min_distance_;
      latched_capture_min_cross_track_ = terminal_capture_min_cross_track_;
      latched_capture_replan_drift_ = terminal_capture_replan_drift_;
      latched_capture_min_turning_r_ = terminal_capture_min_turning_r_;
      latched_capture_point_spacing_ = terminal_capture_point_spacing_;
      latched_capture_max_residual_y_ = terminal_capture_max_residual_y_;
      auto [cap_plan, cap_bound] =
        getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);
      transformed_plan = std::move(cap_plan);
      prunePlan(global_plan_up_to_inversion_, cap_bound);
      if (clock_) {
        RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
          "[TERMINAL_CAPTURE] Latched 1-gear capture path: pts=%zu goal_dist=%.3fm "
          "cross_track=%.3fm",
          global_plan_up_to_inversion_.poses.size(), goal_dist, cross_track_dist);
      }
    }
  }

  if (has_latched_terminal_capture_ &&
    terminal_capture_pullthrough_length_ > 0.0 &&
    !transformed_plan.poses.empty() &&
    !global_plan_up_to_inversion_.poses.empty())
  {
    const double gear = current_segment_is_reverse_ ? -1.0 : 1.0;
    const double goal_nose_yaw = tf2::getYaw(goal_pose.pose.orientation);
    const double goal_travel_yaw =
      gear < 0.0 ? angles::normalize_angle(goal_nose_yaw + M_PI) : goal_nose_yaw;
    const double dx = global_pose.pose.position.x - goal_pose.pose.position.x;
    const double dy = global_pose.pose.position.y - goal_pose.pose.position.y;
    const double along_track_to_goal =
      -(dx * std::cos(goal_travel_yaw) + dy * std::sin(goal_travel_yaw));
    const double release_dist = std::max(terminal_capture_pullthrough_release_, 0.0);
    if (along_track_to_goal > release_dist && along_track_to_goal <= 0.18) {
      const double pull_len = std::max(0.0, terminal_capture_pullthrough_length_);
      const auto & end_plan_pose = global_plan_up_to_inversion_.poses.back();
      const double end_nose_yaw = tf2::getYaw(end_plan_pose.pose.orientation);
      const double end_travel_yaw =
        gear < 0.0 ? angles::normalize_angle(end_nose_yaw + M_PI) : end_nose_yaw;
      const double min_spacing = (costmap_ && costmap_->getCostmap()) ?
        std::max(0.01, static_cast<double>(costmap_->getCostmap()->getResolution())) : 0.01;
      const double spacing = std::max(min_spacing, terminal_capture_point_spacing_);
      const size_t num_pull = std::max<size_t>(
        1u, static_cast<size_t>(std::ceil(pull_len / spacing)));
      for (size_t k = 1u; k <= num_pull; ++k) {
        const double dist_k =
          (pull_len * static_cast<double>(k)) / static_cast<double>(num_pull);
        geometry_msgs::msg::PoseStamped pull_plan_pose;
        pull_plan_pose.header.frame_id = global_plan_.header.frame_id;
        pull_plan_pose.header.stamp = global_pose.header.stamp;
        pull_plan_pose.pose.position.x =
          end_plan_pose.pose.position.x + dist_k * std::cos(end_travel_yaw);
        pull_plan_pose.pose.position.y =
          end_plan_pose.pose.position.y + dist_k * std::sin(end_travel_yaw);
        pull_plan_pose.pose.position.z = end_plan_pose.pose.position.z;
        pull_plan_pose.pose.orientation = end_plan_pose.pose.orientation;
        if (!isFootprintCollisionFreeInPlanFrame(pull_plan_pose)) {
          break;
        }
        geometry_msgs::msg::PoseStamped pull_costmap_pose;
        if (!transformPose(costmap_->getGlobalFrameID(), pull_plan_pose, pull_costmap_pose)) {
          break;
        }
        transformed_plan.poses.push_back(pull_costmap_pose);
      }
    }
  }
}

bool PathHandler::solveTerminalCapturePath(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::PoseStamped & goal_pose,
  nav_msgs::msg::Path & capture_plan) const
{
  const double min_turning_r = std::max(terminal_capture_min_turning_r_, 0.10);
  const double gear = current_segment_is_reverse_ ? -1.0 : 1.0;
  const double goal_nose_yaw = tf2::getYaw(goal_pose.pose.orientation);
  const double robot_nose_yaw = tf2::getYaw(robot_pose.pose.orientation);
  const double goal_travel_yaw =
    gear < 0.0 ? angles::normalize_angle(goal_nose_yaw + M_PI) : goal_nose_yaw;
  const double robot_travel_yaw =
    gear < 0.0 ? angles::normalize_angle(robot_nose_yaw + M_PI) : robot_nose_yaw;

  const double dx = robot_pose.pose.position.x - goal_pose.pose.position.x;
  const double dy = robot_pose.pose.position.y - goal_pose.pose.position.y;
  const double cos_g = std::cos(goal_travel_yaw);
  const double sin_g = std::sin(goal_travel_yaw);

  // Transform robot pose into Goal Travel Frame (goal at (0, 0), travel heading 0)
  const double X0 = dx * cos_g + dy * sin_g;
  const double Y0 = -dx * sin_g + dy * cos_g;
  const double theta0 = angles::shortest_angular_distance(goal_travel_yaw, robot_travel_yaw);

  // Require the robot to be longitudinally behind the goal in its travel direction
  // and facing within +/- 85 deg of the goal travel heading.
  if (X0 >= -0.05 || std::fabs(theta0) >= 1.4835) {
    return false;
  }

  struct CandidateArc
  {
    double R{0.0};
    double s{1.0};
    double theta_m{0.0};
    double L1{0.0};
    double L2{0.0};
    double x_offset{0.0};
    double y_offset{0.0};
  };

  std::vector<CandidateArc> candidates;
  const double A = 2.0 * (std::cos(theta0) - 1.0);
  const double C = X0 * X0 + Y0 * Y0;

  for (const double s : {1.0, -1.0}) {
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

    for (const double R : roots) {
      if (!std::isfinite(R) || R < min_turning_r) {
        continue;
      }
      const double sin_num = R * std::sin(theta0) - s * X0;
      const double cos_num = s * Y0 + R * (1.0 + std::cos(theta0));
      const double theta_m = std::atan2(sin_num, cos_num);
      // Require forward progress throughout (|theta_m| < 85 deg)
      if (std::fabs(theta_m) >= 1.4835) {
        continue;
      }
      const double xm = -s * R * std::sin(theta_m);
      if (xm < X0 - 1e-3 || xm > 1e-3) {
        continue;
      }
      const double dtheta1 = s * (theta_m - theta0);
      const double dtheta2 = s * theta_m;
      const double raw_L1 = R * dtheta1;
      const double raw_L2 = R * dtheta2;
      if (raw_L1 < -1e-4 || raw_L2 < -1e-4) {
        continue;
      }
      const double R1 = s * R;
      const double R2 = -s * R;
      const double xm1 = X0 + R1 * (std::sin(theta_m) - std::sin(theta0));
      const double ym1 = Y0 - R1 * (std::cos(theta_m) - std::cos(theta0));
      const double xm2 = R2 * std::sin(theta_m);
      const double ym2 = R2 * (1.0 - std::cos(theta_m));
      if (std::hypot(xm1 - xm2, ym1 - ym2) > 1e-4) {
        continue;
      }
      const double L1 = std::max(0.0, raw_L1);
      const double L2 = std::max(0.0, raw_L2);
      const double total_len = L1 + L2;
      if (total_len < 0.05 || total_len > 2.0 * std::hypot(X0, Y0)) {
        continue;
      }
      candidates.push_back({R, s, theta_m, L1, L2, 0.0, 0.0});
    }
  }

  // Bounded-R_min best-effort fallback when no exact Y_end = 0 curve satisfies R >= min_turning_r:
  if (candidates.empty() && terminal_capture_max_residual_y_ > 0.0) {
    const double R = min_turning_r;
    const double max_sin = std::sin(1.4835);
    const double x_end = std::min(0.015, -0.05 * X0);
    for (const double s : {1.0, -1.0}) {
      const double sin_theta_m = (R * std::sin(theta0) + s * (x_end - X0)) / (2.0 * R);
      if (std::fabs(sin_theta_m) >= max_sin) {
        continue;
      }
      const double theta_m = std::asin(sin_theta_m);
      const double xm = X0 + s * R * (std::sin(theta_m) - std::sin(theta0));
      if (xm < X0 - 1e-3 || xm > x_end + 1e-3) {
        continue;
      }
      const double dtheta1 = s * (theta_m - theta0);
      const double dtheta2 = s * theta_m;
      const double raw_L1 = R * dtheta1;
      const double raw_L2 = R * dtheta2;
      if (raw_L1 < -1e-4 || raw_L2 < -1e-4) {
        continue;
      }
      const double y_end =
        Y0 + s * R * (1.0 + std::cos(theta0) - 2.0 * std::cos(theta_m));
      if (std::fabs(y_end) > terminal_capture_max_residual_y_ ||
        std::fabs(y_end) > 0.60 * std::fabs(Y0))
      {
        continue;
      }
      const double L1 = std::max(0.0, raw_L1);
      const double L2 = std::max(0.0, raw_L2);
      const double total_len = L1 + L2;
      if (total_len < 0.05 || total_len > 2.0 * std::hypot(X0, Y0)) {
        continue;
      }
      candidates.push_back({R, s, theta_m, L1, L2, x_end, y_end});
    }
  }

  if (candidates.empty()) {
    return false;
  }

  // Prefer exact capture curves first (smallest |y_offset|), then gentlest curvature (largest R)
  std::sort(
    candidates.begin(), candidates.end(),
    [](const CandidateArc & a, const CandidateArc & b) {
      if (std::fabs(std::fabs(a.y_offset) - std::fabs(b.y_offset)) > 1e-6) {
        return std::fabs(a.y_offset) < std::fabs(b.y_offset);
      }
      return a.R > b.R;
    });

  auto evalGoalFramePose = [&](
    const CandidateArc & cand, double s_arc,
    double & xg, double & yg, double & th_travel_g)
    {
      const double R1 = cand.s * cand.R;
      const double R2 = -cand.s * cand.R;
      if (s_arc <= cand.L1) {
        th_travel_g = theta0 + s_arc / R1;
        xg = X0 + R1 * (std::sin(th_travel_g) - std::sin(theta0));
        yg = Y0 - R1 * (std::cos(th_travel_g) - std::cos(theta0));
      } else {
        const double s2 = std::min(s_arc - cand.L1, cand.L2);
        th_travel_g = cand.theta_m + s2 / R2;
        xg = cand.x_offset + R2 * std::sin(th_travel_g);
        yg = cand.y_offset + R2 * (1.0 - std::cos(th_travel_g));
      }
    };

  auto toPlanPose = [&](double xg, double yg, double th_travel_g)
    {
      geometry_msgs::msg::PoseStamped p;
      p.header.frame_id = global_plan_.header.frame_id;
      p.header.stamp = robot_pose.header.stamp;
      p.pose.position.x = goal_pose.pose.position.x + (xg * cos_g - yg * sin_g);
      p.pose.position.y = goal_pose.pose.position.y + (xg * sin_g + yg * cos_g);
      p.pose.position.z = goal_pose.pose.position.z;
      const double plan_travel_yaw = angles::normalize_angle(goal_travel_yaw + th_travel_g);
      const double plan_nose_yaw =
        gear < 0.0 ? angles::normalize_angle(plan_travel_yaw + M_PI) : plan_travel_yaw;
      p.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(plan_nose_yaw);
      return p;
    };

  const double min_spacing = (costmap_ && costmap_->getCostmap()) ?
    std::max(0.01, static_cast<double>(costmap_->getCostmap()->getResolution())) : 0.01;
  const double spacing = std::max(min_spacing, terminal_capture_point_spacing_);
  const double check_spacing = std::min(0.025, spacing);

  for (const auto & cand : candidates) {
    const double total_len = cand.L1 + cand.L2;
    const size_t num_checks = std::max<size_t>(
      2u, static_cast<size_t>(std::ceil(total_len / check_spacing)));
    bool collision_free = true;
    for (size_t i = 0; i <= num_checks; ++i) {
      const double s_arc = (total_len * static_cast<double>(i)) / static_cast<double>(num_checks);
      double xg = 0.0, yg = 0.0, th_g = 0.0;
      evalGoalFramePose(cand, s_arc, xg, yg, th_g);
      const auto plan_pose = toPlanPose(xg, yg, th_g);
      if (!isFootprintCollisionFreeInPlanFrame(plan_pose)) {
        collision_free = false;
        break;
      }
    }
    if (!collision_free) {
      continue;
    }

    capture_plan.header.frame_id = global_plan_.header.frame_id;
    capture_plan.header.stamp = robot_pose.header.stamp;
    capture_plan.poses.clear();
    const size_t num_pts = std::max<size_t>(
      2u, static_cast<size_t>(std::ceil(total_len / spacing)));
    capture_plan.poses.reserve(num_pts + 1u);
    for (size_t i = 0; i < num_pts; ++i) {
      const double s_arc = (total_len * static_cast<double>(i)) / static_cast<double>(num_pts);
      double xg = 0.0, yg = 0.0, th_g = 0.0;
      evalGoalFramePose(cand, s_arc, xg, yg, th_g);
      capture_plan.poses.push_back(toPlanPose(xg, yg, th_g));
    }
    double end_xg = 0.0, end_yg = 0.0, end_th_g = 0.0;
    evalGoalFramePose(cand, total_len, end_xg, end_yg, end_th_g);
    auto exact_goal = (std::fabs(cand.y_offset) <= 1e-6 && std::fabs(cand.x_offset) <= 1e-6) ?
      goal_pose : toPlanPose(end_xg, end_yg, end_th_g);
    exact_goal.header.frame_id = global_plan_.header.frame_id;
    exact_goal.header.stamp = robot_pose.header.stamp;
    if (!isFootprintCollisionFreeInPlanFrame(exact_goal)) {
      continue;
    }
    capture_plan.poses.push_back(exact_goal);
    return true;
  }

  return false;
}

}  // namespace mppi
