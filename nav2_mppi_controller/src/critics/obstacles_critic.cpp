// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "nav2_mppi_controller/critics/obstacles_critic.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"
#include "nav2_core/controller_exceptions.hpp"

namespace mppi::critics
{

void ObstaclesCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(consider_footprint_, "consider_footprint", false);
  // Footprint availability is checked at configure time. Reject later writes,
  // including when another component predeclared this parameter without a
  // read-only descriptor. Do not register a dynamic storage callback for it.
  const auto hard_check_name = name_ + ".collision_check_footprint_only";
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  descriptor.description = "Hard footprint collision mode; requires controller restart";
  nav2_util::declare_parameter_if_not_declared(
    parent_.lock(), hard_check_name, rclcpp::ParameterValue(false), descriptor);
  getParam(collision_check_footprint_only_, "collision_check_footprint_only",
    false, ParameterType::Static);
  hard_check_param_callback_ = parent_.lock()->add_on_set_parameters_callback(
    [hard_check_name](const std::vector<rclcpp::Parameter> & parameters) {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      for (const auto & parameter : parameters) {
        if (parameter.get_name() == hard_check_name) {
          result.successful = false;
          result.reason = "collision_check_footprint_only is read-only; restart the controller";
          break;
        }
      }
      return result;
    });
  getParam(power_, "cost_power", 1);
  getParam(repulsion_weight_, "repulsion_weight", 1.5f);
  getParam(critical_weight_, "critical_weight", 20.0f);
  getParam(collision_cost_, "collision_cost", 100000.0f);
  getParam(collision_margin_distance_, "collision_margin_distance", 0.10f);
  getParam(near_goal_distance_, "near_goal_distance", 0.5f);
  getParam(goal_truncation_distance_, "goal_truncation_distance", 0.0f);
  getParam(inflation_layer_name_, "inflation_layer_name", std::string(""));

  if (goal_truncation_distance_ > 0.0f) {
    RCLCPP_WARN(
      logger_, "goal_truncation_distance is ignored: collision checking covers the full rollout");
  }

  collision_checker_.setCostmap(costmap_);
  possible_collision_cost_ = findCircumscribedCost(costmap_ros_);

  if (collision_check_footprint_only_ && costmap_ros_->getUseRadius()) {
    throw nav2_core::ControllerException(
      "Hard footprint collision checking requires a configured robot footprint.");
  }

  if (possible_collision_cost_ < 1.0f) {
    RCLCPP_ERROR(
      logger_,
      "Inflation layer either not found or inflation is not set sufficiently for "
      "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
      " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
      "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
      " for full instructions. This will substantially impact run-time performance.");
  }

  if (costmap_ros_->getUseRadius() ==
    (consider_footprint_ || collision_check_footprint_only_))
  {
    RCLCPP_WARN(
    logger_,
    "Inconsistent configuration in collision checking. Please verify the robot's shape settings "
    "in both the costmap and the obstacle critic.");
    if (costmap_ros_->getUseRadius()) {
      throw nav2_core::ControllerException(
      "Considering footprint in collision checking but no robot footprint provided in the "
      "costmap.");
    }
  }

  RCLCPP_INFO(
    logger_,
    "ObstaclesCritic instantiated with %d power and %f / %f weights. "
    "Critic will collision check based on %s cost.",
    power_, critical_weight_, repulsion_weight_,
    (consider_footprint_ || collision_check_footprint_only_) ?
    "footprint" : "circular");
}

float ObstaclesCritic::findCircumscribedCost(
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap)
{
  double result = -1.0;
  const double circum_radius = costmap->getLayeredCostmap()->getCircumscribedRadius();
  if (static_cast<float>(circum_radius) == circumscribed_radius_) {
    // early return if footprint size is unchanged
    return circumscribed_cost_;
  }

  // check if the costmap has an inflation layer
  const auto inflation_layer = nav2_costmap_2d::InflationLayer::getInflationLayer(
    costmap,
    inflation_layer_name_);
  if (inflation_layer != nullptr) {
    const double resolution = costmap->getCostmap()->getResolution();
    result = inflation_layer->computeCost(circum_radius / resolution);
    inflation_scale_factor_ = static_cast<float>(inflation_layer->getCostScalingFactor());
    inflation_radius_ = static_cast<float>(inflation_layer->getInflationRadius());
  } else {
    RCLCPP_WARN(
      logger_,
      "No inflation layer found in costmap configuration. "
      "If this is an SE2-collision checking plugin, it cannot use costmap potential "
      "field to speed up collision checking by only checking the full footprint "
      "when robot is within possibly-inscribed radius of an obstacle. This may "
      "significantly slow down planning times and not avoid anything but absolute collisions!");
  }

  circumscribed_radius_ = static_cast<float>(circum_radius);
  circumscribed_cost_ = static_cast<float>(result);

  return circumscribed_cost_;
}

float ObstaclesCritic::distanceToObstacle(const CollisionCost & cost)
{
  if (score_cache_active_ && dist_lut_ready_ &&
    cost.cost >= 1.0f && cost.cost < 256.0f)
  {
    const unsigned int idx = static_cast<unsigned int>(cost.cost);
    if (static_cast<float>(idx) == cost.cost) {
      return cost.using_footprint ? dist_lut_footprint_[idx] : dist_lut_point_[idx];
    }
  }

  const float scale_factor = inflation_scale_factor_;
  const float min_radius = costmap_ros_->getLayeredCostmap()->getInscribedRadius();
  float dist_to_obj = (scale_factor * min_radius - log(cost.cost) + log(253.0f)) / scale_factor;

  // If not footprint collision checking, the cost is using the center point cost and
  // needs the radius subtracted to obtain the closest distance to the object
  if (!cost.using_footprint) {
    dist_to_obj -= min_radius;
  }

  return dist_to_obj;
}

void ObstaclesCritic::prepareScoreCache()
{
  cached_is_tracking_unknown_ =
    costmap_ros_->getLayeredCostmap()->isTrackingUnknown();

  dist_lut_ready_ = false;
  if (inflation_radius_ != 0.0f && inflation_scale_factor_ != 0.0f) {
    for (unsigned int c = 1; c < 256; ++c) {
      float c_f = static_cast<float>(c);
#if defined(__GNUC__) || defined(__clang__)
      asm volatile("" : "+r" (c_f));
#endif
      dist_lut_footprint_[c] = distanceToObstacle(CollisionCost{c_f, true});
      dist_lut_point_[c] = distanceToObstacle(CollisionCost{c_f, false});
    }
    dist_lut_ready_ = true;
  }

  prefilter_ready_ = false;
  if (!consider_footprint_ && !collision_check_footprint_only_) {
    return;
  }

  cached_footprint_ = costmap_ros_->getRobotFootprint();
  if (cached_footprint_.empty() || costmap_ == nullptr) {
    return;
  }

  if (!collision_check_footprint_only_ && possible_collision_cost_ >= 1.0f) {
    return;
  }

  const unsigned int size_x = costmap_->getSizeInCellsX();
  const unsigned int size_y = costmap_->getSizeInCellsY();
  const double resolution = costmap_->getResolution();
  const unsigned char * const char_map = costmap_->getCharMap();
  if (size_x == 0u || size_y == 0u || resolution <= 0.0 || char_map == nullptr) {
    return;
  }

  double max_sq_radius = 0.0;
  for (const auto & pt : cached_footprint_) {
    max_sq_radius = std::max(max_sq_radius, pt.x * pt.x + pt.y * pt.y);
  }
  const double max_vertex_radius = std::max(
    std::sqrt(max_sq_radius),
    static_cast<double>(costmap_ros_->getLayeredCostmap()->getCircumscribedRadius()));
  const unsigned int radius =
    static_cast<unsigned int>(std::ceil(max_vertex_radius / resolution)) + 1u;

  prefilter_size_x_ = size_x;
  prefilter_size_y_ = size_y;
  prefilter_radius_ = radius;

  if (size_x / 2u <= radius || size_y / 2u <= radius ||
    size_x > 65535u || size_y > 65535u)
  {
    return;
  }

  const unsigned int stride = size_x + 1u;
  const size_t integral_cells =
    static_cast<size_t>(stride) * static_cast<size_t>(size_y + 1u);
  prefilter_integral_.assign(integral_cells, 0u);

  for (unsigned int y = 0; y < size_y; ++y) {
    const unsigned char * const src_row = char_map + static_cast<size_t>(y) * size_x;
    const uint32_t * const prev_row =
      prefilter_integral_.data() + static_cast<size_t>(y) * stride;
    uint32_t * const cur_row =
      prefilter_integral_.data() + static_cast<size_t>(y + 1u) * stride;
    uint32_t row_sum = 0u;
    if (consider_footprint_) {
      for (unsigned int x = 0; x < size_x; ++x) {
        row_sum += static_cast<uint32_t>(src_row[x] != nav2_costmap_2d::FREE_SPACE);
        cur_row[x + 1u] = prev_row[x + 1u] + row_sum;
      }
    } else {
      for (unsigned int x = 0; x < size_x; ++x) {
        row_sum += static_cast<uint32_t>(src_row[x] >= nav2_costmap_2d::LETHAL_OBSTACLE);
        cur_row[x + 1u] = prev_row[x + 1u] + row_sum;
      }
    }
  }

  prefilter_ready_ = true;
}

void ObstaclesCritic::score(CriticData & data)
{
  using xt::evaluation_strategy::immediate;
  if (!enabled_) {
    return;
  }

  if (consider_footprint_ || collision_check_footprint_only_) {
    // footprint may have changed since initialization if user has dynamic footprints
    possible_collision_cost_ = findCircumscribedCost(costmap_ros_);
  }

  struct ScoreCacheGuard
  {
    explicit ScoreCacheGuard(ObstaclesCritic & critic)
    : critic_(critic)
    {
      critic_.prepareScoreCache();
      critic_.score_cache_active_ = true;
    }
    ~ScoreCacheGuard()
    {
      critic_.score_cache_active_ = false;
    }
    ObstaclesCritic & critic_;
  } cache_guard(*this);

  // If near the goal, don't apply the preferential term since the goal is near obstacles
  bool near_goal = false;
  if (utils::withinPositionGoalTolerance(near_goal_distance_, data.state.pose.pose, data.goal)) {
    near_goal = true;
  }

  auto && raw_cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
  auto && repulsive_cost = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});

  const size_t traj_len = data.trajectories.x.shape(1);
  bool all_trajectories_collide = true;
  for (size_t i = 0; i < data.trajectories.x.shape(0); ++i) {
    bool trajectory_collide = false;
    float traj_cost = 0.0f;
    const auto & traj = data.trajectories;
    CollisionCost pose_cost;
    raw_cost[i] = 0.0f;
    repulsive_cost[i] = 0.0f;

    for (size_t j = 0; j < traj_len; j++) {
      // A rollout is a motion forecast, including its tail after entering the goal radius.
      // Only a stopping forecast may establish that the tail is not executed.
      pose_cost = costAtPose(traj.x(i, j), traj.y(i, j), traj.yaws(i, j));
      if (inCollision(pose_cost.cost)) {
        trajectory_collide = true;
        break;
      }

      if (pose_cost.cost < 1.0f) {continue;}  // In free space

      // Cannot process repulsion if inflation layer does not exist
      if (inflation_radius_ == 0.0f || inflation_scale_factor_ == 0.0f) {
        continue;
      }

      const float dist_to_obj = distanceToObstacle(pose_cost);

      // Let near-collision trajectory points be punished severely
      if (dist_to_obj < collision_margin_distance_) {
        traj_cost += (collision_margin_distance_ - dist_to_obj);
      }

      // Generally prefer trajectories further from obstacles
      if (!near_goal) {
        repulsive_cost[i] += inflation_radius_ - dist_to_obj;
      }
    }

    if (!trajectory_collide) {all_trajectories_collide = false;}
    raw_cost[i] = trajectory_collide ? collision_cost_ : traj_cost;
  }

  // Normalize repulsive cost by trajectory length & lowest score to not overweight importance
  // This is a preferential cost, not collision cost, to be tuned relative to desired behaviors
  auto && repulsive_cost_normalized =
    (repulsive_cost - xt::amin(repulsive_cost, immediate)) / traj_len;

  if (power_ > 1u) {
    data.costs += xt::pow(
      (critical_weight_ * raw_cost) +
      (repulsion_weight_ * repulsive_cost_normalized),
      power_);
  } else {
    data.costs += (critical_weight_ * raw_cost) +
      (repulsion_weight_ * repulsive_cost_normalized);
  }

  data.fail_flag = all_trajectories_collide;
}

/**
  * @brief Checks if cost represents a collision
  * @param cost Costmap cost
  * @return bool if in collision
  */
bool ObstaclesCritic::inCollision(float cost) const
{
  bool is_tracking_unknown = score_cache_active_ ?
    cached_is_tracking_unknown_ :
    costmap_ros_->getLayeredCostmap()->isTrackingUnknown();

  using namespace nav2_costmap_2d; // NOLINT
  switch (static_cast<unsigned char>(cost)) {
    case (LETHAL_OBSTACLE):
      return true;
    case (INSCRIBED_INFLATED_OBSTACLE):
      return consider_footprint_ ? false : true;
    case (NO_INFORMATION):
      return is_tracking_unknown ? false : true;
  }

  return false;
}

CollisionCost ObstaclesCritic::costAtPose(float x, float y, float theta)
{
  CollisionCost collision_cost;
  float & cost = collision_cost.cost;
  collision_cost.using_footprint = false;
  unsigned int x_i, y_i;
  if (!collision_checker_.worldToMap(x, y, x_i, y_i)) {
    cost = nav2_costmap_2d::NO_INFORMATION;
    return collision_cost;
  }
  cost = collision_checker_.pointCost(x_i, y_i);
  if (collision_check_footprint_only_ && cost == nav2_costmap_2d::LETHAL_OBSTACLE) {
    return collision_cost;  // A clear perimeter cannot erase a lethal reference cell.
  }

  // The continuous circumscribed-radius threshold is not conservative for
  // rasterized footprint edges. A corner can visit a lethal cell while the
  // reference cell is one inflation-cost bin below that threshold. Hard-only
  // checking must not depend on the inflation threshold.
  if (collision_check_footprint_only_ || (consider_footprint_ &&
    (cost >= possible_collision_cost_ || possible_collision_cost_ < 1.0f)))
  {
    float footprint_cost = 0.0f;
    if (score_cache_active_ && prefilter_ready_) {
      const unsigned int stride = prefilter_size_x_ + 1u;
      if (x_i >= prefilter_radius_ && x_i + prefilter_radius_ < prefilter_size_x_ &&
        y_i >= prefilter_radius_ && y_i + prefilter_radius_ < prefilter_size_y_)
      {
        const unsigned int x0 = x_i - prefilter_radius_;
        const unsigned int x1 = x_i + prefilter_radius_ + 1u;
        const unsigned int y0 = y_i - prefilter_radius_;
        const unsigned int y1 = y_i + prefilter_radius_ + 1u;
        const uint32_t disqualifying_cells =
          prefilter_integral_[static_cast<size_t>(y1) * stride + x1] -
          prefilter_integral_[static_cast<size_t>(y0) * stride + x1] -
          prefilter_integral_[static_cast<size_t>(y1) * stride + x0] +
          prefilter_integral_[static_cast<size_t>(y0) * stride + x0];
        if (disqualifying_cells == 0u) {
          if (consider_footprint_) {
            cost = 0.0f;
            collision_cost.using_footprint = true;
          }
          return collision_cost;
        }
      }

      const size_t fp_size = cached_footprint_.size();
      if (fp_size >= 2u && fp_size <= 16u) {
        const double xd = static_cast<double>(x);
        const double yd = static_cast<double>(y);
        const double thetad = static_cast<double>(theta);
        const double cos_th = cos(thetad);
        const double sin_th = sin(thetad);
        unsigned int mx[16];
        unsigned int my[16];
        unsigned int min_x = prefilter_size_x_;
        unsigned int max_x = 0u;
        unsigned int min_y = prefilter_size_y_;
        unsigned int max_y = 0u;
        bool all_in_bounds = true;
        for (size_t k = 0; k < fp_size; ++k) {
          const double wx = xd + (cached_footprint_[k].x * cos_th -
            cached_footprint_[k].y * sin_th);
          const double wy = yd + (cached_footprint_[k].x * sin_th +
            cached_footprint_[k].y * cos_th);
          if (!collision_checker_.worldToMap(wx, wy, mx[k], my[k]) ||
            mx[k] >= prefilter_size_x_ || my[k] >= prefilter_size_y_)
          {
            all_in_bounds = false;
            break;
          }
          min_x = std::min(min_x, mx[k]);
          max_x = std::max(max_x, mx[k]);
          min_y = std::min(min_y, my[k]);
          max_y = std::max(max_y, my[k]);
        }
        if (!all_in_bounds) {
          footprint_cost = static_cast<float>(nav2_costmap_2d::LETHAL_OBSTACLE);
        } else {
          const unsigned int x1 = max_x + 1u;
          const unsigned int y1 = max_y + 1u;
          const uint32_t aabb_disqualifying =
            prefilter_integral_[static_cast<size_t>(y1) * stride + x1] -
            prefilter_integral_[static_cast<size_t>(min_y) * stride + x1] -
            prefilter_integral_[static_cast<size_t>(y1) * stride + min_x] +
            prefilter_integral_[static_cast<size_t>(min_y) * stride + min_x];
          if (aabb_disqualifying == 0u) {
            if (consider_footprint_) {
              cost = 0.0f;
              collision_cost.using_footprint = true;
            }
            return collision_cost;
          }
          double max_edge_cost = 0.0;
          for (size_t k = 0; k < fp_size - 1u; ++k) {
            max_edge_cost = std::max(
              collision_checker_.lineCost(mx[k], mx[k + 1u], my[k], my[k + 1u]),
              max_edge_cost);
            if (max_edge_cost == static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE)) {
              break;
            }
          }
          if (max_edge_cost != static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE)) {
            max_edge_cost = std::max(
              collision_checker_.lineCost(mx[0], mx[fp_size - 1u], my[0], my[fp_size - 1u]),
              max_edge_cost);
          }
          footprint_cost = static_cast<float>(max_edge_cost);
        }
      } else {
        footprint_cost = static_cast<float>(
          collision_checker_.footprintCostAtPose(x, y, theta, cached_footprint_));
      }
    } else {
      footprint_cost = static_cast<float>(
        score_cache_active_ ?
        collision_checker_.footprintCostAtPose(x, y, theta, cached_footprint_) :
        collision_checker_.footprintCostAtPose(x, y, theta, costmap_ros_->getRobotFootprint()));
    }
    // Full-footprint soft scoring remains unchanged. In hard-only mode, preserve
    // the existing point cost unless the perimeter is lethal or outside the map.
    // An inscribed inflation cost on the perimeter is not a physical collision.
    if (consider_footprint_ || footprint_cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
      footprint_cost == nav2_costmap_2d::NO_INFORMATION)
    {
      cost = footprint_cost;
      collision_cost.using_footprint = true;
    }
  }

  return collision_cost;
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  mppi::critics::ObstaclesCritic,
  mppi::critics::CriticFunction)
