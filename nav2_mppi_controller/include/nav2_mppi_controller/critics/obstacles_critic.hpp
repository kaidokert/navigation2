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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLES_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLES_CRITIC_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_costmap_2d/inflation_layer.hpp"
#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::ConstraintCritic
 * @brief Critic objective function for avoiding obstacles, allowing it to deviate off
 * the planned path. This is important to tune in tandem with PathAlign to make a balance
 * between path-tracking and dynamic obstacle avoidance capabilities as desirable for a
 * particular application
 */
class ObstaclesCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
   * @brief Evaluate cost related to obstacle avoidance
   *
   * @param costs [out] add obstacle cost values to this tensor
   */
  void score(CriticData & data) override;

protected:
  /**
    * @brief Checks if cost represents a collision
    * @param cost Costmap cost
    * @return bool if in collision
    */
  bool inCollision(float cost) const;

  /**
    * @brief cost at a robot pose
    * @param x X of pose
    * @param y Y of pose
    * @param theta theta of pose
    * @return Collision information at pose
    */
  CollisionCost costAtPose(float x, float y, float theta);

  /**
    * @brief Distance to obstacle from cost
    * @param cost Costmap cost
    * @return float Distance to the obstacle represented by cost
    */
  float distanceToObstacle(const CollisionCost & cost);

  /**
    * @brief Find the min cost of the inflation decay function for which the robot MAY be
    * in collision in any orientation
    * @param costmap Costmap2DROS to get minimum inscribed cost (e.g. 128 in inflation layer documentation)
    * @return double circumscribed cost, any higher than this and need to do full footprint collision checking
    * since some element of the robot could be in collision
    */
  float findCircumscribedCost(std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap);

  /**
    * @brief Precompute per-score-cycle footprint prefilter and distance lookup table
    */
  void prepareScoreCache();

  /**
    * @brief Check whether the active score cache matches the current costmap state
    * @return true if the precomputed score cache can be reused for this iteration
    */
  bool canReuseScoreCache() const;

  /**
    * @brief Retrieve the latest padded robot footprint using thread-safe atomic access
    * @return Copy of the current padded footprint polygon
    */
  nav2_costmap_2d::Footprint getLatestFootprint() const;

  /**
    * @brief Compute the maximum cost along a grid edge using the cached charmap and integral image
    * @param x0 Starting cell X coordinate
    * @param x1 Ending cell X coordinate
    * @param y0 Starting cell Y coordinate
    * @param y1 Ending cell Y coordinate
    * @return Maximum cell cost along the segment, or 0.0 if prefiltered as free/non-lethal
    */
  double lineCostCached(
    unsigned int x0, unsigned int x1, unsigned int y0, unsigned int y1) const;

protected:
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>
  collision_checker_{nullptr};

  bool consider_footprint_{true};
  // Optional hard perimeter veto while retaining point-based soft clearance costs.
  bool collision_check_footprint_only_{false};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr hard_check_param_callback_;
  float collision_cost_{0};
  float inflation_scale_factor_{0}, inflation_radius_{0};

  float possible_collision_cost_;
  float collision_margin_distance_;
  float near_goal_distance_;
  // Stop scoring a trajectory once it comes within this distance of the goal:
  // the executed robot stops there, so post-goal rollout points are fantasy.
  // 0 disables (upstream behavior). Fixes the goal-near-wall standoff veto
  // (M6, notes/cusp_handoff_fix_plan.md 2026-09-25).
  float goal_truncation_distance_{0};
  float circumscribed_cost_{0}, circumscribed_radius_{0}, inscribed_radius_{0};
  double cached_circum_resolution_{0.0};

  unsigned int power_{0};
  float repulsion_weight_, critical_weight_{0};
  std::string inflation_layer_name_;

  nav2_costmap_2d::Footprint cached_footprint_;
  std::vector<uint32_t> prefilter_integral_;
  std::array<float, 256> dist_lut_footprint_{};
  std::array<float, 256> dist_lut_point_{};
  const unsigned char * cached_char_map_{nullptr};
  double cached_origin_x_{0.0};
  double cached_origin_y_{0.0};
  double cached_resolution_{0.0};
  unsigned int prefilter_size_x_{0};
  unsigned int prefilter_size_y_{0};
  unsigned int prefilter_radius_{0};
  size_t prepare_score_cache_count_{0};
  bool cached_is_tracking_unknown_{false};
  bool dist_lut_ready_{false};
  bool prefilter_ready_{false};
  bool cache_built_once_{false};
  bool score_cache_active_{false};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__OBSTACLES_CRITIC_HPP_
