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
    getParam(inversion_yaw_tolerance, "inversion_yaw_tolerance", 0.4);
    getParam(min_inversion_horizon_, "min_inversion_horizon", 0.15);
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
  geometry_msgs::msg::PoseStamped global_pose =
    transformToGlobalPlanFrame(robot_pose);
  auto [transformed_plan, lower_bound] = getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);

  prunePlan(global_plan_up_to_inversion_, lower_bound);

  // Standard inversion handoff — persistent, unchanged from upstream.
  // No rebuild of transformed_plan here: the original 1-tick lag is preserved
  // to avoid control spikes from instantaneous plan jumps at the cusp.
  if (enforce_path_inversion_ && inversion_locale_ != 0u) {
    if (isWithinInversionTolerances(global_pose)) {
      prunePlan(global_plan_, global_plan_.poses.begin() + inversion_locale_);
      global_plan_up_to_inversion_ = global_plan_;
      inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
    }
  }

  // Virtual Lookahead — stateless horizon extension for micro-segment cusps.
  //
  // Trigger: the current segment has structurally few points before the next
  // inversion (inversion_locale_ < 10), indicating a planner micro-segment
  // artifact, AND the visible TGP is too short for MPPI to produce meaningful
  // cost gradients.
  //
  // Action: borrow points from global_plan_ starting at the inversion boundary
  // (index-based, not Euclidean search) and append them to transformed_plan
  // for this cycle only. No persistent state is modified.
  //
  // This does NOT fire during normal approach to a real inversion (e.g., M8's
  // 0.97m reverse segment has inversion_locale_ ~35, well above the threshold).
  // Virtual Lookahead — stateless horizon extension for micro-segment cusps.
  //
  // When the visible TGP is shorter than min_inversion_horizon_ and there is
  // a next inversion boundary, borrow points from global_plan_ past that
  // boundary. This prevents MPPI's cost gradient from collapsing to zero
  // at planner-generated micro-segments (Ackermann cusp artifacts).
  // No persistent state is modified — only the returned transformed_plan.
  float tgp_length = utils::pathLength(transformed_plan);
  if (enforce_path_inversion_ && min_inversion_horizon_ > 0.0f &&
      inversion_locale_ != 0u &&
      tgp_length < min_inversion_horizon_ &&
      !transformed_plan.poses.empty())
  {
    size_t appended = 0;
    unsigned int mx, my;
    for (size_t i = inversion_locale_;
         i < global_plan_.poses.size() && tgp_length < min_inversion_horizon_;
         ++i)
    {
      // Create a local copy to avoid mutating global_plan_ headers
      geometry_msgs::msg::PoseStamped global_pose_copy = global_plan_.poses[i];
      global_pose_copy.header.stamp = global_pose.header.stamp;
      global_pose_copy.header.frame_id = global_plan_.header.frame_id;

      geometry_msgs::msg::PoseStamped costmap_pose;
      if (transformPose(costmap_->getGlobalFrameID(), global_pose_copy, costmap_pose)) {
        // Respect costmap bounds — same check as normal plan builder
        if (!costmap_->getCostmap()->worldToMap(
            costmap_pose.pose.position.x, costmap_pose.pose.position.y, mx, my))
        {
          break;
        }
        // Incrementally track path length to avoid O(N²) recomputation
        if (!transformed_plan.poses.empty()) {
          const auto & prev = transformed_plan.poses.back();
          tgp_length += hypotf(
            costmap_pose.pose.position.x - prev.pose.position.x,
            costmap_pose.pose.position.y - prev.pose.position.y);
        }
        transformed_plan.poses.push_back(costmap_pose);
        appended++;
      }
    }

    if (appended > 0) {
      RCLCPP_DEBUG(logger_,
        "[HORIZON_GUARD] Virtual lookahead: appended %zu pts from idx %u, "
        "TGP now %zu pts / %.4fm (inv_locale=%u)",
        appended, inversion_locale_, transformed_plan.poses.size(),
        tgp_length, inversion_locale_);
    }
  }

  if (transformed_plan.poses.empty()) {
    throw nav2_core::InvalidPath("Resulting plan has 0 poses in it.");
  }

  // Diagnostic: TGP metadata after inversion crop + virtual lookahead
  {
    double cusp_dist = 0.0;
    if (!global_plan_up_to_inversion_.poses.empty()) {
      const auto & cusp = global_plan_up_to_inversion_.poses.back();
      cusp_dist = hypot(
        global_pose.pose.position.x - cusp.pose.position.x,
        global_pose.pose.position.y - cusp.pose.position.y);
    }
    RCLCPP_DEBUG_THROTTLE(logger_, *clock_, 200,
      "[PathHandler] TGP_PTS=%zu TGP_LEN=%.4fm INV_LOCALE=%u CUSP_DIST=%.4fm",
      transformed_plan.poses.size(), tgp_length, inversion_locale_,
      cusp_dist);
  }

  return transformed_plan;
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

bool PathHandler::isWithinInversionTolerances(const geometry_msgs::msg::PoseStamped & robot_pose)
{
  // Keep full path if we are within tolerance of the inversion pose
  const auto last_pose = global_plan_up_to_inversion_.poses.back();
  float distance = hypotf(
    robot_pose.pose.position.x - last_pose.pose.position.x,
    robot_pose.pose.position.y - last_pose.pose.position.y);

  float angle_distance = angles::shortest_angular_distance(
    tf2::getYaw(robot_pose.pose.orientation),
    tf2::getYaw(last_pose.pose.orientation));

  return distance <= inversion_xy_tolerance_ && fabs(angle_distance) <= inversion_yaw_tolerance;
}

}  // namespace mppi
