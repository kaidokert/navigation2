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

#include <cmath>
#include <limits>

#include "nav2_mppi_controller/tools/path_handler.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

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
    getParam(ghost_min_seed_arc_length_, "ghost_min_seed_arc_length", 0.05);
    getParam(ghost_min_seed_points_, "ghost_min_seed_points", 4);
    getParam(micro_cusp_length_threshold_, "micro_cusp_length_threshold", 0.25);
    getParam(micro_cusp_yaw_scale_, "micro_cusp_yaw_scale", 2.0);
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
    ghost_min_seed_arc_length_ = std::max(ghost_min_seed_arc_length_, 0.0);
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

  auto closest_pose_upper_bound =
    nav2_util::geometry_utils::first_after_integrated_distance(
    global_plan_up_to_inversion_.poses.begin(), global_plan_up_to_inversion_.poses.end(),
    max_robot_pose_search_dist_);

  auto closest_point = nav2_util::geometry_utils::min_by(
    begin, closest_pose_upper_bound,
    [&global_pose](const geometry_msgs::msg::PoseStamped & ps) {
      return euclidean_distance(global_pose, ps);
    });

  nav_msgs::msg::Path transformed_plan;
  transformed_plan.header.frame_id = costmap_->getGlobalFrameID();
  transformed_plan.header.stamp = global_pose.header.stamp;

  auto pruned_plan_end =
    nav2_util::geometry_utils::first_after_integrated_distance(
    closest_point, global_plan_up_to_inversion_.poses.end(), prune_distance_);

  unsigned int mx, my;
  for (auto global_plan_pose = closest_point; global_plan_pose != pruned_plan_end;
    ++global_plan_pose)
  {
    geometry_msgs::msg::PoseStamped costmap_plan_pose;
    global_plan_pose->header.stamp = global_pose.header.stamp;
    global_plan_pose->header.frame_id = global_plan_.header.frame_id;
    transformPose(costmap_->getGlobalFrameID(), *global_plan_pose, costmap_plan_pose);

    if (!costmap_->getCostmap()->worldToMap(
        costmap_plan_pose.pose.position.x, costmap_plan_pose.pose.position.y, mx, my))
    {
      return {transformed_plan, closest_point};
    }

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
  geometry_msgs::msg::PoseStamped global_pose = transformToGlobalPlanFrame(robot_pose);
  auto [transformed_plan, lower_bound] = getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);

  prunePlan(global_plan_up_to_inversion_, lower_bound);

  if (enforce_path_inversion_ && inversion_locale_ != 0u) {
    if (isWithinInversionTolerances(global_pose)) {
      prunePlan(global_plan_, global_plan_.poses.begin() + inversion_locale_);
      global_plan_up_to_inversion_ = global_plan_;
      inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
      current_segment_length_ = static_cast<double>(
        utils::pathLength(global_plan_up_to_inversion_));
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
    }
  }

  const double real_tgp_length = static_cast<double>(utils::pathLength(transformed_plan));
  double tgp_length = real_tgp_length;
  double cusp_dist = std::numeric_limits<double>::infinity();
  double ghost_heading = 0.0;
  if (shouldUseGhostPath(
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
  } else if (real_tgp_length < min_inversion_horizon_ && inversion_locale_ != 0u) {
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

    // Peek at the active segment plan to detect reverse travel direction.
    // Use global_plan_up_to_inversion_ which is pruned near the robot each tick,
    // so its first points reflect the local travel direction. global_plan_ is NOT
    // pruned each tick and may retain points from the segment start, giving a stale
    // direction on long curved segments (>90° arcs).
    if (global_plan_up_to_inversion_.poses.size() >= 2) {
      const auto & p0 = global_plan_up_to_inversion_.poses[0].pose.position;
      const auto & p1 = global_plan_up_to_inversion_.poses[1].pose.position;
      double travel_dir = std::atan2(p1.y - p0.y, p1.x - p0.x);
      if (std::fabs(angles::shortest_angular_distance(pose_yaw, travel_dir)) > M_PI_2) {
        pose_yaw = angles::normalize_angle(pose_yaw + M_PI);
        RCLCPP_INFO(logger_,
          "[GHOST_FLIP] Single-point reverse detected: pose=%.0fdeg -> travel=%.0fdeg",
          tf2::getYaw(transformed_plan.poses[0].pose.orientation) * 180.0 / M_PI,
          pose_yaw * 180.0 / M_PI);
      }
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
  next_pose.pose.orientation = nav2_util::geometry_utils::orientationAroundZAxis(heading);

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
  global_plan_ = plan;
  global_plan_up_to_inversion_ = global_plan_;
  if (enforce_path_inversion_) {
    inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
    current_segment_length_ = static_cast<double>(
      utils::pathLength(global_plan_up_to_inversion_));
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
  const geometry_msgs::msg::PoseStamped & robot_pose) const
{
  const auto last_pose = global_plan_up_to_inversion_.poses.back();
  const double distance = std::hypot(
    robot_pose.pose.position.x - last_pose.pose.position.x,
    robot_pose.pose.position.y - last_pose.pose.position.y);

  const double angle_distance = angles::shortest_angular_distance(
    tf2::getYaw(robot_pose.pose.orientation),
    tf2::getYaw(last_pose.pose.orientation));

  // Relax yaw tolerance for micro-cusps: segments shorter than the threshold
  // are planner artifacts, not meaningful maneuvers. Demanding precise yaw
  // alignment on a 4cm segment overconstrains the Ackermann controller.
  const bool is_micro_cusp =
    current_segment_length_ < micro_cusp_length_threshold_;
  const double effective_yaw_tol = is_micro_cusp
    ? inversion_yaw_tolerance_ * micro_cusp_yaw_scale_
    : inversion_yaw_tolerance_;

  bool result = distance <= inversion_xy_tolerance_ &&
         std::fabs(angle_distance) <= effective_yaw_tol;
  if (result && is_micro_cusp) {
    RCLCPP_INFO(logger_,
      "[HANDOFF_MICRO] xy=%.4fm yaw=%.3frad(tol=%.2f) seg_len=%.3fm",
      distance, std::fabs(angle_distance), effective_yaw_tol,
      current_segment_length_);
  } else if (!result && distance <= inversion_xy_tolerance_) {
    RCLCPP_INFO_THROTTLE(logger_, *clock_, 500,
      "[HANDOFF_BLOCKED] xy=%.4fm(<%.2f) yaw=%.3frad(need<%.2f%s) seg_len=%.3fm",
      distance, inversion_xy_tolerance_,
      std::fabs(angle_distance), effective_yaw_tol,
      is_micro_cusp ? " MICRO" : "",
      current_segment_length_);
  }
  return result;
}

}  // namespace mppi
