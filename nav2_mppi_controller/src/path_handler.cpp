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
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstring>
#include <cstdint>

#include "nav2_mppi_controller/tools/path_handler.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_mppi_controller/tools/diagnostic_capture.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

namespace mppi
{
namespace
{
int segmentDirection(const nav_msgs::msg::Path & path)
{
  for (size_t k = 1; k < path.poses.size(); ++k) {
    const auto & a = path.poses[k - 1].pose;
    const auto & b = path.poses[k].pose;
    const double dx = b.position.x - a.position.x;
    const double dy = b.position.y - a.position.y;
    if (std::hypot(dx, dy) > 1e-6) {
      const double yaw = tf2::getYaw(a.orientation);
      return dx * std::cos(yaw) + dy * std::sin(yaw) >= 0 ? 1 : -1;
    }
  }
  return 0;
}

double nextSegmentLength(const nav_msgs::msg::Path & plan, unsigned int inversion)
{
  if (inversion == 0u || inversion >= plan.poses.size()) {
    return std::numeric_limits<double>::infinity();
  }
  nav_msgs::msg::Path next;
  next.poses.assign(plan.poses.begin() + inversion - 1, plan.poses.end());
  utils::removePosesAfterFirstInversion(next);
  const double length = static_cast<double>(utils::pathLength(next));
  return length > 0.0 ? length : std::numeric_limits<double>::infinity();
}
}  // namespace

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
    getParam(inversion_segment_ratio_, "inversion_segment_ratio", 0.25);
    getParam(inversion_xy_floor_, "inversion_xy_floor", 0.05);
    getParam(inversion_yaw_base_, "inversion_yaw_base", 0.20);
    // Experimental policy: previous unconditional version regressed the loop.
    // Keep opt-in so paired frozen-path diagnostics can test it independently.
    getParam(bound_handoff_by_next_segment_, "bound_handoff_by_next_segment", false);
    getParam(validate_long_forward_departure_, "validate_long_forward_departure", false);
    getParam(validate_short_forward_departure_, "validate_short_forward_departure", false);
    getParam(preserve_cusp_direction_edge_, "preserve_cusp_direction_edge", false);
    getParam(replan_on_cusp_overrun_, "replan_on_cusp_overrun", false);
    getParam(min_inversion_horizon_, "min_inversion_horizon", 0.15);
    getParam(ghost_max_extension_, "ghost_max_extension", 0.35);
    getParam(ghost_point_spacing_, "ghost_point_spacing", 0.05);
    getParam(ghost_min_seed_points_, "ghost_min_seed_points", 4);
    getParam(micro_cusp_length_threshold_, "micro_cusp_length_threshold", 0.25);
    getParam(micro_cusp_yaw_scale_, "micro_cusp_yaw_scale", 2.0);
    getParam(handoff_yaw_relax_per_cycle_, "handoff_yaw_relax_per_cycle", 0.005);
    getParam(handoff_yaw_relax_cap_, "handoff_yaw_relax_cap", 0.8);
    // <=0 disables the escape policy (default: disabled pending oscillation latch)
    getParam(handoff_escape_after_cycles_, "handoff_escape_after_cycles", 0);
    handoff_yaw_relax_per_cycle_ = std::max(handoff_yaw_relax_per_cycle_, 0.0);
    handoff_yaw_relax_cap_ = std::max(
      handoff_yaw_relax_cap_, static_cast<double>(inversion_yaw_tolerance_));
    inversion_segment_ratio_ = std::clamp(inversion_segment_ratio_, 0.05, 1.0);
    inversion_xy_floor_ = std::max(0.0, inversion_xy_floor_);
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
    inversion_locale_ = 0u;
  }
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

  // An active cusp pruned to its endpoint has no real edge defining gear.
  // Retain the preceding genuine edge (including any duplicate cusp poses),
  // only inside the local window. Final-goal pruning remains unchanged.
  if (preserve_cusp_direction_edge_ && enforce_path_inversion_ &&
    inversion_locale_ != 0u && closest_point != begin &&
    std::all_of(closest_point, global_plan_up_to_inversion_.poses.end(),
      [&](const auto & p) {return euclidean_distance(*closest_point, p) < 1e-6;}))
  {
    auto edge_start = closest_point;
    do {--edge_start;} while (edge_start != begin &&
      euclidean_distance(*edge_start, *closest_point) < 1e-6);
    const double edge_length = euclidean_distance(*edge_start, *closest_point);
    auto candidate = *edge_start;
    candidate.header.stamp = global_pose.header.stamp;
    candidate.header.frame_id = global_plan_.header.frame_id;
    geometry_msgs::msg::PoseStamped local_candidate;
    unsigned int mx, my;
    if (edge_length > 1e-6 && edge_length <= prune_distance_ &&
      transformPose(costmap_->getGlobalFrameID(), candidate, local_candidate) &&
      costmap_->getCostmap()->worldToMap(local_candidate.pose.position.x,
        local_candidate.pose.position.y, mx, my))
    {
      closest_point = edge_start;
    }
  }

  PathBinding binding;
  binding.canonical_plan=canonical_plan_;binding.generation=plan_generation_;
  binding.segment=segment_generation_;binding.segment_begin=segment_source_begin_;
  binding.segment_end=segment_source_end_;
  const auto local_frame=costmap_->getGlobalFrameID();
  try {
    if(local_frame==global_plan_.header.frame_id) {
      binding.canonical_to_local.header.frame_id=local_frame;
      binding.canonical_to_local.child_frame_id=global_plan_.header.frame_id;
      binding.canonical_to_local.header.stamp=global_pose.header.stamp;
      binding.canonical_to_local.transform.rotation.w=1.0;
    } else {
      binding.canonical_to_local=tf_buffer_->lookupTransform(local_frame,
        global_plan_.header.frame_id,global_pose.header.stamp,
        tf2::durationFromSec(transform_tolerance_));
    }
  } catch(const tf2::TransformException & ex) {
    path_binding_.reset();throw nav2_core::ControllerTFError(ex.what());
  }
  binding.canonical_to_local.header.stamp=global_pose.header.stamp;
  // Reject malformed canonical transforms before doTransform/worldToMap casts.
  const auto & tf=binding.canonical_to_local;
  const auto & t=tf.transform.translation;const auto & q=tf.transform.rotation;
  const auto finite=diagnostic::isFiniteCaptureValue;
  if(tf.header.frame_id.empty() || tf.child_frame_id.empty() ||
    tf.header.frame_id!=local_frame || tf.child_frame_id!=global_plan_.header.frame_id ||
    !finite(t.x) || !finite(t.y) || !finite(t.z) || !finite(q.x) ||
    !finite(q.y) || !finite(q.z) || !finite(q.w) ||
    std::abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.0)>1e-6 ||
    std::abs(q.x)>1e-6 || std::abs(q.y)>1e-6 || std::abs(t.z)>1e-6)
  {
    path_binding_.reset();
    throw nav2_core::ControllerTFError("Invalid canonical path transform");
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
    tf2::doTransform(*global_plan_pose,costmap_plan_pose,binding.canonical_to_local);

    // Check if pose is inside the costmap
    if (!costmap_->getCostmap()->worldToMap(
        costmap_plan_pose.pose.position.x, costmap_plan_pose.pose.position.y, mx, my))
    {
      path_binding_=std::move(binding);
      return {transformed_plan, closest_point};
    }

    // Filling the transformed plan to return with the transformed pose
    binding.real_ids.push_back(inversion_source_offset_+
      static_cast<size_t>(std::distance(begin,global_plan_pose)));
    transformed_plan.poses.push_back(costmap_plan_pose);
  }

  path_binding_=std::move(binding);
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
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const DepartureValidator & departure_validator)
{
  did_handoff_ = false;
  retained_departure_approach_.reset();
  // Find relevent bounds of path to use
  geometry_msgs::msg::PoseStamped global_pose =
    transformToGlobalPlanFrame(robot_pose);
  auto [transformed_plan, lower_bound] = getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);

  prunePlan(global_plan_up_to_inversion_, lower_bound);

  if (enforce_path_inversion_ && inversion_locale_ != 0u) {
    bool accept_handoff = isWithinInversionTolerances(global_pose);
    const int64_t approach_stamp = static_cast<int64_t>(global_pose.header.stamp.sec) *
      1000000000LL + global_pose.header.stamp.nanosec;
    if (departure_approach_budget_ && (!accept_handoff ||
      departure_approach_budget_->exhausted || approach_stamp < departure_approach_budget_->last_ns ||
      approach_stamp - departure_approach_budget_->started_ns > 2000000000LL))
    {
      departure_approach_budget_->exhausted = true;
      throw nav2_core::FailedToMakeProgress("Retained departure approach left cusp gates");
    }
    const bool short_departure = validate_short_forward_departure_ &&
      next_segment_length_ > 0.0 && next_segment_length_ <= 0.30;
    const bool long_departure = validate_long_forward_departure_ && next_segment_length_ > 0.8;
    if (accept_handoff && current_segment_direction_ == -1 &&
      (short_departure || long_departure))
    {
      // Preview without consuming either the global plan or its incoming cusp.
      nav_msgs::msg::Path outgoing;
      outgoing.header = global_plan_.header;
      outgoing.poses.assign(global_plan_.poses.begin() + inversion_locale_ - 1,
        global_plan_.poses.end());
      const auto outgoing_inversion = utils::removePosesAfterFirstInversion(outgoing);
      // Short endpoint validation is only for another internal cusp, never the final goal.
      if (segmentDirection(outgoing) == 1 && (!short_departure || outgoing_inversion != 0u)) {
      if (!departure_validator) {
        throw nav2_core::NoValidControl("Enabled departure guard has no validator");
      }
        nav_msgs::msg::Path local;
        local.header.frame_id = costmap_->getGlobalFrameID();
        local.header.stamp = global_pose.header.stamp;
        if (short_departure && outgoing.poses.size() > 64) {
          throw nav2_core::FailedToMakeProgress("Short departure exceeds bounded validation size");
        }
        const size_t preview_size = short_departure ? outgoing.poses.size() :
          std::min<size_t>(outgoing.poses.size(), 9);
        for (size_t k = 0; k < preview_size; ++k) {
          auto source = outgoing.poses[k];
          source.header = outgoing.header;
          source.header.stamp = global_pose.header.stamp;
          geometry_msgs::msg::PoseStamped target;
          if (!transformPose(local.header.frame_id, source, target)) {
            throw nav2_core::ControllerTFError("Cannot transform outgoing departure prefix");
          }
          local.poses.push_back(target);
        }
        geometry_msgs::msg::PoseStamped local_robot;
        if (!transformPose(local.header.frame_id, robot_pose, local_robot)) {
          throw nav2_core::ControllerTFError("Cannot transform departure robot pose");
        }
        const auto gates = effectiveInversionTolerances();
        accept_handoff = departure_validator(local_robot, local, gates.first, gates.second, short_departure);
        if (!accept_handoff && short_departure &&
          !retainRejectedShortDeparture(global_pose, local.poses.front(), gates.first))
        {
          // Do not enter a segment with no validated continuation, or wait forever
          // inside the incoming tolerance zone. Preserve the cusp for stopped replanning.
          RCLCPP_WARN(logger_, "[SHORT_DEPARTURE_GUARD] rejected; cusp retained for stopped replan");
          throw nav2_core::FailedToMakeProgress("No validated short forward departure; stopped replan required");
        }
        if (!accept_handoff) {
          RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
            "[DEPARTURE_GUARD] bounded forward search rejected; retaining reverse cusp");
        }
      }
    }
    if (accept_handoff) {
      departure_approach_budget_.reset();
      min_cusp_distance_ = std::numeric_limits<double>::infinity();
      min_goal_distance_ = std::numeric_limits<double>::infinity();
      last_progress_distance_ = std::numeric_limits<double>::infinity();
      goal_stall_cycles_ = 0u;
      const auto & handed_off_cusp = global_plan_up_to_inversion_.poses.back();
      const double handoff_dist = std::hypot(
        global_pose.pose.position.x - handed_off_cusp.pose.position.x,
        global_pose.pose.position.y - handed_off_cusp.pose.position.y);
      const double handoff_yaw_err = angles::shortest_angular_distance(
        tf2::getYaw(global_pose.pose.orientation), tf2::getYaw(handed_off_cusp.pose.orientation));
      const auto [effective_xy_tol, relaxed_yaw_tol] = effectiveInversionTolerances();
      const double robot_yaw = tf2::getYaw(global_pose.pose.orientation);
      const double cusp_yaw = tf2::getYaw(handed_off_cusp.pose.orientation);
      RCLCPP_INFO(logger_,
        "[CUSP_HANDOFF_EVENT] cusp_idx=%u dist=%.4fm (xy_gate=%.4fm) "
        "yaw_err=%.2fdeg (yaw_gate=%.2fdeg) seg_len=%.3fm "
        "robot=(%.3f,%.3f,%.1fdeg) cusp=(%.3f,%.3f,%.1fdeg)",
        inversion_locale_, handoff_dist, effective_xy_tol,
        handoff_yaw_err * 180.0 / M_PI, relaxed_yaw_tol * 180.0 / M_PI,
        current_segment_length_,
        global_pose.pose.position.x, global_pose.pose.position.y, robot_yaw * 180.0 / M_PI,
        handed_off_cusp.pose.position.x, handed_off_cusp.pose.position.y, cusp_yaw * 180.0 / M_PI);

      // inversion_locale_ is the first pose AFTER the shared cusp. Keep the
      // cusp as the start of the outgoing segment: dropping it removes that
      // segment's first edge, and erases a one-edge reversal entirely.
      prunePlan(global_plan_, global_plan_.poses.begin() + inversion_locale_ - 1);
      global_plan_up_to_inversion_ = global_plan_;
      inversion_source_offset_=global_source_offset_;
      ++segment_generation_;segment_source_begin_=global_source_offset_;
      inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
      segment_source_end_=inversion_source_offset_+global_plan_up_to_inversion_.poses.size();
      current_segment_direction_ = segmentDirection(global_plan_up_to_inversion_);
      current_segment_length_ = static_cast<double>(
        utils::pathLength(global_plan_up_to_inversion_));
      next_segment_length_ = nextSegmentLength(global_plan_, inversion_locale_);
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
      prunePlan(global_plan_up_to_inversion_, refreshed_bound);
      did_handoff_ = true;
      cusp_overrun_snapshot_written_ = false;
    }
  }

  real_path_point_count_ = transformed_plan.poses.size();
  // Optional recovery policy, not an assertion of global unreachability:
  // once pruning consumes the real segment, continuing past its endpoint in
  // the commanded gear can keep the displacement-only progress checker alive
  // indefinitely. Stop and let the BT request a fresh plan. Never skip a cusp,
  // issue an escape command, or apply this to the final goal.
  if (replan_on_cusp_overrun_ && enforce_path_inversion_ &&
    inversion_locale_ != 0u && current_segment_direction_ != 0 &&
    (real_path_point_count_ == 1u ||
    (preserve_cusp_direction_edge_ && real_path_point_count_ > 1u)))
  {
    const auto & cusp = global_plan_up_to_inversion_.poses.back().pose;
    const double yaw = tf2::getYaw(cusp.orientation);
    const double overrun = current_segment_direction_ * (
      (global_pose.pose.position.x - cusp.position.x) * std::cos(yaw) +
      (global_pose.pose.position.y - cusp.position.y) * std::sin(yaw));
    const double gate = effectiveInversionTolerances().first;
    // Edge preservation keeps a genuine tangent for the critics; it must not
    // make a consumed endpoint appear unconsumed to this recovery policy.
    // Use the same nearest-position criterion as pruning, without changing
    // either the retained geometry or the cusp handoff tolerances.
    const double cusp_distance = std::hypot(
      global_pose.pose.position.x - cusp.position.x,
      global_pose.pose.position.y - cusp.position.y);
    const bool endpoint_consumed = real_path_point_count_ == 1u || std::all_of(
      global_plan_up_to_inversion_.poses.begin(), global_plan_up_to_inversion_.poses.end(),
      [&](const auto & p) {
        return cusp_distance <= std::hypot(
          global_pose.pose.position.x - p.pose.position.x,
          global_pose.pose.position.y - p.pose.position.y);
      });
    if (endpoint_consumed && overrun > gate) {
      const char * prefix = std::getenv("MPPI_STALL_SNAPSHOT_PATH");
      if (prefix && *prefix && !cusp_overrun_snapshot_written_) {
        cusp_overrun_snapshot_written_ = true;
        try {
          const auto stamp = rclcpp::Time(global_pose.header.stamp).nanoseconds();
          std::ofstream out(std::string(prefix) + ".cusp_overrun." +
            std::to_string(stamp) + ".json");
          out.exceptions(std::ios::failbit | std::ios::badbit);
          out << std::setprecision(17);
          const auto pose = [&out](const geometry_msgs::msg::Pose & p) {
              out << '[' << p.position.x << ',' << p.position.y << ',' <<
                tf2::getYaw(p.orientation) << ']';
            };
          const auto path = [&out, &pose](const nav_msgs::msg::Path & p) {
              out << '[';
              for (size_t i = 0; i < p.poses.size(); ++i) {
                if (i) {out << ',';}
                pose(p.poses[i].pose);
              }
              out << ']';
            };
          out << "{\"stamp_ns\":" << stamp << ",\"frame\":" <<
            std::quoted(global_pose.header.frame_id) << ",\"robot\":";
          pose(global_pose.pose);
          out << ",\"gear\":" << current_segment_direction_ <<
            ",\"inversion_locale\":" << inversion_locale_ <<
            ",\"current_segment_length\":" << current_segment_length_ <<
            ",\"next_segment_length\":" << next_segment_length_ <<
            ",\"gate\":" << gate << ",\"overrun\":" << overrun <<
            ",\"xy_tolerance\":" << inversion_xy_tolerance_ <<
            ",\"yaw_tolerance\":" << inversion_yaw_tolerance_ <<
            ",\"bound_next\":" << (bound_handoff_by_next_segment_ ? "true" : "false") <<
            ",\"real_path_point_count\":" << real_path_point_count_ <<
            ",\"global_plan\":";
          path(global_plan_);
          out << ",\"active_plan\":";
          path(global_plan_up_to_inversion_);
          out << "}\n";
          out.close();
        } catch (const std::exception & e) {
          RCLCPP_WARN(logger_, "Cusp overrun diagnostic capture failed: %s", e.what());
        }
      }
      RCLCPP_WARN(logger_, "[CUSP_OVERRUN] consumed gear=%d overrun=%.4fm gate=%.4fm; "
        "requesting stopped replanning without consuming cusp",
        current_segment_direction_, overrun, gate);
      // The consumed endpoint needs a new plan, not another evaluation of
      // the same segment through ControllerServer's NoValidControl patience
      // loop. Preserve the cusp and let the stopped recovery tree replan.
      throw nav2_core::FailedToMakeProgress("Consumed cusp overrun requires a fresh plan");
    }
  }
  const double real_tgp_length = static_cast<double>(utils::pathLength(transformed_plan));
  double tgp_length = real_tgp_length;
  double cusp_dist = std::numeric_limits<double>::infinity();
  double ghost_heading = 0.0;

  // Bounded ghost extension & stopped-replan safety backstop:
  // If the active segment has been consumed to a single point (or empty) and the robot
  // has passed closest approach by > 5 cm without handoff, terminate ghost extension
  // and throw FailedToMakeProgress to stop the robot and trigger replanning instead
  // of reversing indefinitely into obstacles.
  if (enforce_path_inversion_ && inversion_locale_ != 0u &&
    real_path_point_count_ <= 1u && std::isfinite(min_cusp_distance_))
  {
    cusp_dist = getCuspDistance(global_pose);
    if (cusp_dist > min_cusp_distance_ + 0.05) {
      RCLCPP_WARN(logger_,
        "[CUSP_MISSED] consumed segment past closest approach (min=%.4fm, curr=%.4fm); "
        "requesting stopped replan",
        min_cusp_distance_, cusp_dist);
      throw nav2_core::FailedToMakeProgress("Cusp missed past closest approach; stopped replan required");
    }
  }

  // Terminal goal stall recovery:
  // On the final segment (inversion_locale_ == 0u), when the real path has been
  // consumed to the goal pose (real_path_point_count_ <= 1u), if the vehicle cannot
  // reach the goal tolerance (< 0.10m) and stalls oscillating for > 2.5 seconds (50 cycles)
  // or recedes by > 4 cm past closest approach, throw FailedToMakeProgress to trigger
  // stopped replanning with Smac Planner instead of indefinite limit-cycling.
  if (inversion_locale_ == 0u && !global_plan_.poses.empty()) {
    const auto & goal_pose = global_plan_.poses.back();
    const double goal_dist = std::hypot(
      global_pose.pose.position.x - goal_pose.pose.position.x,
      global_pose.pose.position.y - goal_pose.pose.position.y);
    const double goal_tol = (goal_tolerance_ > 0.0) ? goal_tolerance_ : 0.085;
    const bool near_goal = (real_path_point_count_ <= 3u || goal_dist < (goal_tol + 0.15));
    if (near_goal) {
      if (!std::isfinite(last_progress_distance_)) {
        last_progress_distance_ = goal_dist;
      }
      if (last_progress_distance_ - goal_dist > 0.010) {
        last_progress_distance_ = goal_dist;
        goal_stall_cycles_ = 0u;
      } else {
        ++goal_stall_cycles_;
      }
      if (goal_dist < min_goal_distance_) {
        min_goal_distance_ = goal_dist;
      }
      if (min_goal_distance_ > goal_tol &&
          (goal_dist > min_goal_distance_ + 0.04 || goal_stall_cycles_ > 120u))
      {
        RCLCPP_WARN(logger_,
          "[TERMINAL_STALL] goal_dist=%.4fm min=%.4fm last_prog=%.4fm stall_cycles=%u pts=%zu tol=%.4fm; requesting stopped replan",
          goal_dist, min_goal_distance_, last_progress_distance_, goal_stall_cycles_, real_path_point_count_, goal_tol);
        min_goal_distance_ = std::numeric_limits<double>::infinity();
        last_progress_distance_ = std::numeric_limits<double>::infinity();
        goal_stall_cycles_ = 0u;
        throw nav2_core::FailedToMakeProgress("Terminal goal progress stalled; stopped replan required");
      }
    }
  } else {
    min_goal_distance_ = std::numeric_limits<double>::infinity();
    last_progress_distance_ = std::numeric_limits<double>::infinity();
    goal_stall_cycles_ = 0u;
  }

  // Layer 2 escape (notes/cusp_handoff_fix_plan.md): the handoff has been
  // blocked long enough that extrapolating the consumed segment cannot help â€”
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
      // the reverse move the escape encourages â€” use the post-cusp pose's
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
  if (!escape_handled && shouldUseGhostPath(
      transformed_plan, global_pose, real_tgp_length, cusp_dist,
      ghost_heading))
  {
    // Cap ghost extension proportional to remaining segment length.
    // For micro-segments (e.g. 11cm), extending by ghost_max_extension_ (70cm)
    // overwhelms the actual maneuver â€” MPPI commits to a long trajectory in the
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
  } else if (!escape_handled && real_tgp_length < min_inversion_horizon_ &&
    inversion_locale_ != 0u)
  {
    // Log WHY ghost didn't fire when TGP is short
    cusp_dist = getCuspDistance(global_pose);
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
      "[GHOST_SKIP] tgp=%.4fm pts=%zu inv=%u cusp=%.3fm",
      real_tgp_length, transformed_plan.poses.size(), inversion_locale_, cusp_dist);
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
  if (!enforce_path_inversion_ || min_inversion_horizon_ <= 0.0 || inversion_locale_ == 0u) {
    return false;
  }

  if (transformed_plan.poses.empty() || tgp_length >= min_inversion_horizon_) {
    return false;
  }

  cusp_dist = getCuspDistance(global_pose);

  // Bounded ghost extension: do not extrapolate ghost path if the robot
  // has passed closest approach by more than 5 cm.
  if (std::isfinite(min_cusp_distance_) && cusp_dist > min_cusp_distance_ + 0.05) {
    if (clock_) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 500,
        "[GHOST_BOUND] cusp_dist=%.4fm > min_cusp_dist=%.4fm + 0.05m; suppressing ghost extension",
        cusp_dist, min_cusp_distance_);
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
  // reverse segments and flip by 180Â°. The pose orientation is the robot's
  // nose direction; for reverse segments, travel direction is opposite.
  if (transformed_plan.poses.size() == 1) {
    double pose_yaw = tf2::getYaw(transformed_plan.poses[0].pose.orientation);

    // Once pruning removes the active segment's last edge, recover its gear
    // from the last real edge before the current cusp in the unpruned plan
    // (advanced at every handoff). Compare displacement with the heading AT
    // THAT EDGE, not an earlier heading on a long curve. Skip duplicate points
    // and exclude future segments, whose gear may be opposite.
    const auto & segment = global_plan_up_to_inversion_.poses.size() >= 2
      ? global_plan_up_to_inversion_ : global_plan_;
    const size_t end = (&segment == &global_plan_ && inversion_locale_ > 0u)
      ? std::min<size_t>(inversion_locale_, segment.poses.size()) : segment.poses.size();
    for (size_t i = end; i > 1; --i) {
      const auto & p0 = segment.poses[i - 2].pose;
      const auto & p1 = segment.poses[i - 1].pose;
      const double dx = p1.position.x - p0.position.x;
      const double dy = p1.position.y - p0.position.y;
      if (std::hypot(dx, dy) < 1e-6) {
        continue;
      }
      const double edge_yaw = tf2::getYaw(p0.orientation);
      if (dx * std::cos(edge_yaw) + dy * std::sin(edge_yaw) < 0.0) {
        pose_yaw = angles::normalize_angle(pose_yaw + M_PI);
        RCLCPP_INFO(logger_,
          "[GHOST_FLIP] Single-point reverse detected: pose=%.0fdeg -> travel=%.0fdeg",
          tf2::getYaw(transformed_plan.poses[0].pose.orientation) * 180.0 / M_PI,
          pose_yaw * 180.0 / M_PI);
      }
      break;
    }

    heading = pose_yaw;
    heading_variance = 0.0;
    seed_arc_length = 0.0;  // Single point â€” no arc, but heading is valid
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
  // heading describes travel, whereas path orientations describe the chassis.
  // A reverse continuation must keep the chassis facing opposite its travel;
  // otherwise orientation-aware critics see a synthetic 180-degree turn.
  const double chassis_heading = tf2::getYaw(next_pose.pose.orientation);
  const bool reversing = std::cos(heading - chassis_heading) < 0.0;
  next_pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(
    heading + (reversing ? M_PI : 0.0));

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
  departure_approach_budget_.reset();
  retained_departure_approach_.reset();
  min_cusp_distance_ = std::numeric_limits<double>::infinity();
  min_goal_distance_ = std::numeric_limits<double>::infinity();
  last_progress_distance_ = std::numeric_limits<double>::infinity();
  goal_stall_cycles_ = 0u;
  canonical_plan_=std::make_shared<const nav_msgs::msg::Path>(plan);
  ++plan_generation_;++segment_generation_;path_binding_.reset();
  global_source_offset_=0;inversion_source_offset_=0;segment_source_begin_=0;
  global_plan_ = plan;
  cusp_overrun_snapshot_written_ = false;
  global_plan_up_to_inversion_ = global_plan_;
  current_segment_direction_ = 0;
  if (enforce_path_inversion_) {
    inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
    current_segment_direction_ = segmentDirection(global_plan_up_to_inversion_);
    current_segment_length_ = static_cast<double>(
      utils::pathLength(global_plan_up_to_inversion_));
    next_segment_length_ = nextSegmentLength(global_plan_, inversion_locale_);
  }
  segment_source_end_=global_plan_up_to_inversion_.poses.size();
}

nav_msgs::msg::Path & PathHandler::getPath() {return global_plan_;}

void PathHandler::prunePlan(nav_msgs::msg::Path & plan, const PathIterator end)
{
  const auto removed=static_cast<size_t>(std::distance(plan.poses.begin(),end));
  if(&plan==&global_plan_) {global_source_offset_+=removed;}
  if(&plan==&global_plan_up_to_inversion_) {inversion_source_offset_+=removed;}
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

std::pair<double, double> PathHandler::effectiveInversionTolerances() const
{
  // Relax yaw tolerance for micro-cusps: segments shorter than the threshold
  // are planner artifacts, not meaningful maneuvers. Demanding precise yaw
  // alignment on a 4cm segment overconstrains the Ackermann controller.
  const bool is_micro_cusp =
    current_segment_length_ < micro_cusp_length_threshold_;
  const double effective_yaw_tol = is_micro_cusp
    ? static_cast<double>(inversion_yaw_tolerance_) * micro_cusp_yaw_scale_
    : std::min<double>(inversion_yaw_tolerance_, inversion_yaw_base_);

  // A fixed handoff radius can encompass an entire short segment, consuming
  // its cusp before any of the required motion takes place. Restrict that
  // radius to a fraction (inversion_segment_ratio_) of the segment length,
  // floored at inversion_xy_floor_ to avoid deadband stalls on micro-segments.
  // Longer segments retain the configured gate.
  const double gate_segment_length = bound_handoff_by_next_segment_
    ? std::min(current_segment_length_, next_segment_length_) : current_segment_length_;
  const double max_allowed = static_cast<double>(inversion_xy_tolerance_);
  const double scaled_gate = inversion_segment_ratio_ * gate_segment_length;
  const double floor_val = std::min<double>(inversion_xy_floor_, max_allowed);
  const double effective_xy_tol = is_micro_cusp
    ? max_allowed
    : std::max<double>(floor_val, std::min<double>(max_allowed, scaled_gate));

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

  return {effective_xy_tol, relaxed_yaw_tol};
}

bool PathHandler::retainRejectedShortDeparture(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::PoseStamped & local_cusp, double xy_gate)
{
  // Diagnostic policy bounds, not a reachability claim. The controller must
  // separately validate the final emitted forecast against this stop margin.
  const auto finite = [](double value) {
      uint64_t bits; std::memcpy(&bits, &value, sizeof(bits));
      return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    };
  const auto & cusp = global_plan_up_to_inversion_.poses.back().pose;
  const double dx = pose.pose.position.x - cusp.position.x;
  const double dy = pose.pose.position.y - cusp.position.y;
  const double yaw = tf2::getYaw(cusp.orientation);
  const double remaining = dx * std::cos(yaw) + dy * std::sin(yaw);
  const double distance = std::hypot(dx,dy);
  const double stop_margin = 0.1 * xy_gate;
  const int64_t stamp = static_cast<int64_t>(pose.header.stamp.sec) * 1000000000LL +
    pose.header.stamp.nanosec;
  const auto reject = [&]() {
      if (departure_approach_budget_) {departure_approach_budget_->exhausted = true;}
      return false;
    };
  if (!finite(remaining) || !finite(distance) || !finite(xy_gate) ||
    !(xy_gate > 0) || remaining <= stop_margin || stamp < 0 ||
    current_segment_direction_ != -1) {return reject();}
  if (!departure_approach_budget_) {
    if (remaining <= 0.5 * xy_gate) {return false;}
    departure_approach_budget_ = DepartureApproachBudget{
      stamp, stamp, stamp, remaining, remaining, distance, remaining, 0, false};
  }
  auto & budget = *departure_approach_budget_;
  if (budget.exhausted || stamp < budget.last_ns || stamp - budget.started_ns > 2000000000LL ||
    ++budget.calls > 64 || budget.first_remaining - remaining > 0.5 * xy_gate ||
    remaining > budget.best_remaining + 0.001 || distance > budget.best_distance + 0.001)
  {
    return reject();
  }
  if (stamp - budget.progress_ns >= 1000000000LL) {
    if (budget.progress_remaining - remaining < 0.002) {return reject();}
    budget.progress_ns = stamp; budget.progress_remaining = remaining;
  }
  budget.last_ns = stamp;
  budget.best_remaining = std::min(budget.best_remaining,remaining);
  budget.best_distance = std::min(budget.best_distance,distance);
  auto stamped_cusp = local_cusp;
  stamped_cusp.header.frame_id = costmap_->getGlobalFrameID();
  stamped_cusp.header.stamp = pose.header.stamp;
  retained_departure_approach_ = RetainedDepartureApproach{stamped_cusp,stop_margin};
  return true;
}

bool PathHandler::isWithinInversionTolerances(
  const geometry_msgs::msg::PoseStamped & robot_pose) const
{
  const auto last_pose = global_plan_up_to_inversion_.poses.back();
  const float distance = hypotf(
    robot_pose.pose.position.x - last_pose.pose.position.x,
    robot_pose.pose.position.y - last_pose.pose.position.y);
  const float angle_distance = angles::shortest_angular_distance(
    tf2::getYaw(robot_pose.pose.orientation), tf2::getYaw(last_pose.pose.orientation));
  const auto [effective_xy_tol, relaxed_yaw_tol] = effectiveInversionTolerances();
  const bool is_micro_cusp = current_segment_length_ < micro_cusp_length_threshold_;
  if (distance < min_cusp_distance_) {
    min_cusp_distance_ = distance;
  }

  const bool within_xy = distance <= effective_xy_tol;
  const bool within_yaw = std::fabs(angle_distance) <= relaxed_yaw_tol;

  // Closest-approach handoff for near misses:
  // If the robot reached within 2 cm of the effective xy gate with heading aligned,
  // and has now passed closest approach (distance increasing by > 5 mm),
  // accept handoff so non-holonomic near misses do not stall or reverse into obstacles.
  constexpr float closest_approach_margin = 0.02f;
  constexpr float closest_approach_delta = 0.005f;
  const bool closest_approach_handoff = within_yaw &&
    (min_cusp_distance_ <= effective_xy_tol + closest_approach_margin) &&
    (distance > min_cusp_distance_ + closest_approach_delta);

  bool result = (within_xy && within_yaw) || closest_approach_handoff;

  if (closest_approach_handoff && !within_xy && clock_) {
    RCLCPP_INFO(logger_,
      "[CUSP_HANDOFF_CLOSEST_APPROACH] min_dist=%.4fm dist=%.4fm xy_gate=%.4fm "
      "yaw_err=%.2fdeg(gate=%.2fdeg)",
      min_cusp_distance_, distance, effective_xy_tol,
      angle_distance * 180.0 / M_PI, relaxed_yaw_tol * 180.0 / M_PI);
  }

  // Optional forensic trace of the exact pose and gates used at this call.
  // Sampled /tf or ground truth cannot establish that a handoff evaluation
  // coincided with a brief gate entry. This trace never changes the decision.
  const char * gate_trace = std::getenv("MPPI_CUSP_GATE_TRACE_PATH");
  if (gate_trace && *gate_trace && distance <= 0.1f) {
    try {
      std::ofstream out(gate_trace, std::ios::app);
      out.exceptions(std::ios::failbit | std::ios::badbit);
      const auto pose = [&out](const geometry_msgs::msg::Pose & p) {
          out << '[' << p.position.x << ',' << p.position.y << ',' <<
            tf2::getYaw(p.orientation) << ']';
        };
      out << std::setprecision(17) << "{\"stamp_ns\":" <<
        rclcpp::Time(robot_pose.header.stamp).nanoseconds() <<
        ",\"frame\":" << std::quoted(robot_pose.header.frame_id) << ",\"robot\":";
      pose(robot_pose.pose);
      out << ",\"cusp\":";
      pose(last_pose.pose);
      out << ",\"distance_m\":" << distance <<
        ",\"yaw_error_rad\":" << angle_distance <<
        ",\"xy_gate_m\":" << effective_xy_tol <<
        ",\"yaw_gate_rad\":" << relaxed_yaw_tol <<
        ",\"blocked_cycles_before\":" << handoff_blocked_cycles_ <<
        ",\"accepted\":" << (result ? "true" : "false") << "}\n";
      out.close();
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "Cusp gate diagnostic trace failed: %s", e.what());
    }
  }

  if (!result && distance <= effective_xy_tol) {
    ++handoff_blocked_cycles_;
    // clock_ is only set by initialize(); guard so the tolerance check stays
    // callable on a bare PathHandler (unit tests construct it without a node â€”
    // dereferencing the null clock here was a SIGSEGV in path_handler_test).
    if (clock_) {
      RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
        "[HANDOFF_BLOCKED] xy=%.4fm(<%.2f) yaw=%.3frad(need<%.2f%s) "
        "seg_len=%.3fm blocked=%u",
        distance, effective_xy_tol,
        std::fabs(angle_distance), relaxed_yaw_tol,
        is_micro_cusp ? " MICRO" : "",
        current_segment_length_, handoff_blocked_cycles_);
    }
  } else {
    // Unlocked, or left the xy tolerance: relaxation must not bank across
    // separate approaches.
    handoff_blocked_cycles_ = 0u;
    if (result && is_micro_cusp) {
      RCLCPP_INFO(logger_,
        "[HANDOFF_MICRO] xy=%.4fm yaw=%.3frad(tol=%.2f) seg_len=%.3fm",
        distance, std::fabs(angle_distance), relaxed_yaw_tol,
        current_segment_length_);
    }
  }
  return result;
}

bool PathHandler::computeEscapeHeading(double & heading) const
{
  // Layer 2: after enough consecutive blocked cycles, offer the optimizer the
  // FUTURE â€” the travel direction of the segment after the cusp â€” instead of
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

}  // namespace mppi
