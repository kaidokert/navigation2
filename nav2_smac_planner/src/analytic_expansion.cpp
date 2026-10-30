// Copyright (c) 2021, Samsung Research America
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
// limitations under the License. Reserved.

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>

#include <algorithm>
#include <vector>
#include <memory>

#include "nav2_smac_planner/analytic_expansion.hpp"

namespace nav2_smac_planner
{

template<typename NodeT>
AnalyticExpansion<NodeT>::AnalyticExpansion(
  const MotionModel & motion_model,
  const SearchInfo & search_info,
  const bool & traverse_unknown,
  const unsigned int & dim_3_size)
: _motion_model(motion_model),
  _search_info(search_info),
  _traverse_unknown(traverse_unknown),
  _dim_3_size(dim_3_size),
  _collision_checker(nullptr)
{
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::setCollisionChecker(
  GridCollisionChecker * collision_checker)
{
  _collision_checker = collision_checker;
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::NodePtr AnalyticExpansion<NodeT>::tryAnalyticExpansion(
  const NodePtr & current_node, const NodePtr & goal_node,
  const NodeGetter & getter, int & analytic_iterations,
  int & closest_distance)
{
  // This must be a valid motion model for analytic expansion to be attempted
  if (_motion_model == MotionModel::DUBIN || _motion_model == MotionModel::REEDS_SHEPP ||
    _motion_model == MotionModel::STATE_LATTICE)
  {
    // See if we are closer and should be expanding more often
    const Coordinates node_coords =
      NodeT::getCoords(
      current_node->getIndex(), _collision_checker->getCostmap()->getSizeInCellsX(), _dim_3_size);
    closest_distance = std::min(
      closest_distance,
      static_cast<int>(NodeT::getHeuristicCost(node_coords, goal_node->pose)));

    // We want to expand at a rate of d/expansion_ratio,
    // but check to see if we are so close that we would be expanding every iteration
    // If so, limit it to the expansion ratio (rounded up)
    int desired_iterations = std::max(
      static_cast<int>(closest_distance / _search_info.analytic_expansion_ratio),
      static_cast<int>(std::ceil(_search_info.analytic_expansion_ratio)));

    // If we are closer now, we should update the target number of iterations to go
    analytic_iterations =
      std::min(analytic_iterations, desired_iterations);

    // Always run the expansion on the first run in case there is a
    // trivial path to be found
    if (analytic_iterations <= 0) {
      // Reset the counter and try the analytic path expansion
      analytic_iterations = desired_iterations;
      if (current_node->parent == nullptr &&
        _search_info.enable_single_gear_analytic &&
        _motion_model == MotionModel::REEDS_SHEPP)
      {
        AnalyticExpansionNodes single_gear_nodes =
          getSingleGearAnalyticPath(current_node, goal_node, getter);
        if (!single_gear_nodes.empty()) {
          return setAnalyticPath(current_node, goal_node, single_gear_nodes);
        }
      }
      AnalyticExpansionNodes analytic_nodes =
        getAnalyticPath(current_node, goal_node, getter, current_node->motion_table.state_space);
      if (!analytic_nodes.empty()) {
        // If we have a valid path, attempt to refine it
        NodePtr node = current_node;
        NodePtr test_node = current_node;
        AnalyticExpansionNodes refined_analytic_nodes;
        for (int i = 0; i < 8; i++) {
          // Attempt to create better paths in 5 node increments, need to make sure
          // they exist for each in order to do so (maximum of 40 points back).
          if (test_node->parent && test_node->parent->parent && test_node->parent->parent->parent &&
            test_node->parent->parent->parent->parent &&
            test_node->parent->parent->parent->parent->parent)
          {
            test_node = test_node->parent->parent->parent->parent->parent;
            refined_analytic_nodes =
              getAnalyticPath(test_node, goal_node, getter, test_node->motion_table.state_space);
            if (refined_analytic_nodes.empty()) {
              break;
            }
            analytic_nodes = refined_analytic_nodes;
            node = test_node;
          } else {
            break;
          }
        }

        // The analytic expansion can short-cut near obstacles when closer to a goal
        // So, we can attempt to refine it more by increasing the possible radius
        // higher than the minimum turning radius and use the best solution based on
        // a scoring function similar to that used in traveral cost estimation.
        auto scoringFn = [&](const AnalyticExpansionNodes & expansion) {
            if (expansion.size() < 2) {
              return std::numeric_limits<float>::max();
            }

            float score = 0.0;
            float normalized_cost = 0.0;
            const float & weight = expansion[0].node->motion_table.cost_penalty;
            if (!_search_info.analytic_directional_scoring) {
              // Analytic expansions are consistently spaced
              const float distance = hypotf(
                expansion[1].proposed_coords.x - expansion[0].proposed_coords.x,
                expansion[1].proposed_coords.y - expansion[0].proposed_coords.y);
              for (auto iter = expansion.begin(); iter != expansion.end(); ++iter) {
                normalized_cost = iter->node->getCost() / 252.0f;
                // Search's Traversal Cost Function
                score += distance * (1.0 + weight * normalized_cost);
              }
              return score;
            }

            const float rev_pen = std::max(1.0f, _search_info.reverse_penalty);
            const float chg_pen = std::max(0.0f, _search_info.change_penalty) *
              node->motion_table.min_turning_radius;
            Coordinates prev_coords = node->pose;
            int prev_gear = 0;
            for (auto iter = expansion.begin(); iter != expansion.end(); ++iter) {
              const float dx = iter->proposed_coords.x - prev_coords.x;
              const float dy = iter->proposed_coords.y - prev_coords.y;
              const float step_dist = hypotf(dx, dy);
              const float prev_yaw = node->motion_table.getAngleFromBin(
                static_cast<unsigned int>(prev_coords.theta));
              const float fwd_proj = dx * std::cos(prev_yaw) + dy * std::sin(prev_yaw);
              int gear = prev_gear;
              if (step_dist > 1e-3f) {
                gear = (fwd_proj < -1e-4f) ? -1 : 1;
              }
              if (prev_gear != 0 && gear != 0 && gear != prev_gear) {
                score += chg_pen;
              }
              if (gear != 0) {
                prev_gear = gear;
              }
              const float gear_mult = (gear < 0) ? rev_pen : 1.0f;
              normalized_cost = iter->node->getCost() / 252.0f;
              score += step_dist * gear_mult * (1.0f + weight * normalized_cost);
              prev_coords = iter->proposed_coords;
            }
            return score;
          };

        float best_score = scoringFn(analytic_nodes);
        float score = std::numeric_limits<float>::max();
        float min_turn_rad = node->motion_table.min_turning_radius;
        const float max_min_turn_rad = 4.0 * min_turn_rad;  // Up to 4x the turning radius
        while (min_turn_rad < max_min_turn_rad) {
          min_turn_rad += 0.5;  // In Grid Coords, 1/2 cell steps
          ompl::base::StateSpacePtr state_space;
          if (node->motion_table.motion_model == MotionModel::DUBIN) {
            state_space = std::make_shared<ompl::base::DubinsStateSpace>(min_turn_rad);
          } else {
            state_space = std::make_shared<ompl::base::ReedsSheppStateSpace>(min_turn_rad);
          }
          refined_analytic_nodes = getAnalyticPath(node, goal_node, getter, state_space);
          score = scoringFn(refined_analytic_nodes);
          if (score <= best_score) {
            analytic_nodes = refined_analytic_nodes;
            best_score = score;
          }
        }

        return setAnalyticPath(node, goal_node, analytic_nodes);
      }
    }

    analytic_iterations--;
  }

  // No valid motion model - return nullptr
  return NodePtr(nullptr);
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::AnalyticExpansionNodes AnalyticExpansion<NodeT>::getAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal,
  const NodeGetter & node_getter,
  const ompl::base::StateSpacePtr & state_space)
{
  static ompl::base::ScopedState<> from(state_space), to(state_space), s(state_space);
  const float d_axle = (_motion_model == MotionModel::REEDS_SHEPP ||
    _motion_model == MotionModel::DUBIN) ? _search_info.rear_axle_offset : 0.0f;
  const double from_yaw = node->motion_table.getAngleFromBin(node->pose.theta);
  const double to_yaw = node->motion_table.getAngleFromBin(goal->pose.theta);
  from[0] = node->pose.x - d_axle * std::cos(from_yaw);
  from[1] = node->pose.y - d_axle * std::sin(from_yaw);
  from[2] = from_yaw;
  to[0] = goal->pose.x - d_axle * std::cos(to_yaw);
  to[1] = goal->pose.y - d_axle * std::sin(to_yaw);
  to[2] = to_yaw;

  float d = state_space->distance(from(), to());

  // Feasibility-preserving sampling: cap heading change per step at
  // max_dtheta so chords between samples never imply R < R_min.
  // spacing (cells) = R_min(cells) * max_dtheta, capped at legacy sqrt(2)
  // so long gentle expansions keep today's pose count.
  static const float sqrt_2 = sqrtf(2.0f);
  const float max_dtheta = 0.0873f;  // 5 deg ~= one angular bin (72 bins)
  const float spacing = std::min(
    sqrt_2, node->motion_table.min_turning_radius * max_dtheta);

  // If the length is too far or too short, exit. This prevents unsafe shortcutting of paths
  // into higher cost areas far out from the goal itself, let search to the work of getting
  // close before the analytic expansion brings it home.
  if (d > _search_info.analytic_expansion_max_length || d < sqrt_2) {
    return AnalyticExpansionNodes();
  }

  // Generate per-segment interpolation fractions ensuring samples land exactly
  // at cusps (segment boundaries) and each segment is densely sampled to avoid
  // sub-feasible chords.
  std::vector<float> fractions;
  auto rs_space = std::dynamic_pointer_cast<ompl::base::ReedsSheppStateSpace>(state_space);
  auto dubins_space = std::dynamic_pointer_cast<ompl::base::DubinsStateSpace>(state_space);

  if (rs_space) {
    auto rs_path = rs_space->reedsShepp(from(), to());
    double total_len = rs_path.totalLength_;
    if (total_len > 1e-6) {
      double t_accum = 0.0;
      for (int s_idx = 0; s_idx < 5; ++s_idx) {
        double seg_len = std::abs(rs_path.length_[s_idx]);
        if (seg_len < 1e-6) {
          continue;
        }
        double seg_cell_dist = seg_len * node->motion_table.min_turning_radius;
        const float seg_spacing =
          (rs_path.type_[s_idx] == ompl::base::ReedsSheppStateSpace::RS_STRAIGHT) ?
          sqrt_2 : spacing;
        unsigned int num_seg_intervals = std::max(
          1u, static_cast<unsigned int>(std::ceil(seg_cell_dist / seg_spacing)));
        for (unsigned int k = 1; k <= num_seg_intervals; ++k) {
          double t = t_accum + (static_cast<double>(k) / num_seg_intervals) * (seg_len / total_len);
          fractions.push_back(static_cast<float>(std::min(1.0, t)));
        }
        t_accum += seg_len / total_len;
      }
    }
  } else if (dubins_space) {
    auto dubins_path = dubins_space->dubins(from(), to());
    double total_len = dubins_path.length();
    if (total_len > 1e-6) {
      double t_accum = 0.0;
      for (int s_idx = 0; s_idx < 3; ++s_idx) {
        double seg_len = std::abs(dubins_path.length_[s_idx]);
        if (seg_len < 1e-6) {
          continue;
        }
        double seg_cell_dist = seg_len * node->motion_table.min_turning_radius;
        const float seg_spacing =
          ((*dubins_path.type_)[s_idx] == ompl::base::DubinsStateSpace::DUBINS_STRAIGHT) ?
          sqrt_2 : spacing;
        unsigned int num_seg_intervals = std::max(
          1u, static_cast<unsigned int>(std::ceil(seg_cell_dist / seg_spacing)));
        for (unsigned int k = 1; k <= num_seg_intervals; ++k) {
          double t = t_accum + (static_cast<double>(k) / num_seg_intervals) * (seg_len / total_len);
          fractions.push_back(static_cast<float>(std::min(1.0, t)));
        }
        t_accum += seg_len / total_len;
      }
    }
  }

  if (fractions.empty()) {
    unsigned int num_intervals = std::max(
      1u, static_cast<unsigned int>(std::ceil(d / spacing)));
    for (unsigned int i = 1; i <= num_intervals; ++i) {
      fractions.push_back(static_cast<float>(i) / num_intervals);
    }
  }

  AnalyticExpansionNodes possible_nodes;
  // When "from" and "to" are zero or one cell away,
  // fractions.empty()
  possible_nodes.reserve(fractions.size());  // We won't store this node or the goal
  std::vector<double> reals;
  double theta;

  // Pre-allocate
  NodePtr prev(node);
  uint64_t index = 0;
  NodePtr next(nullptr);
  float angle = 0.0;
  Coordinates proposed_coordinates;
  bool failure = false;
  std::vector<float> node_costs;
  node_costs.reserve(fractions.size());

  // Check intermediary poses (non-goal, non-start)
  for (const float & interp_fraction : fractions) {
    state_space->interpolate(from(), to(), interp_fraction, s());
    reals = s.reals();
    // Make sure in range [0, 2PI)
    theta = (reals[2] < 0.0) ? (reals[2] + 2.0 * M_PI) : reals[2];
    theta = (theta > 2.0 * M_PI) ? (theta - 2.0 * M_PI) : theta;
    angle = node->motion_table.getAngle(theta);
    if (angle >= static_cast<float>(_dim_3_size)) {
      angle -= static_cast<float>(_dim_3_size);
    }

    const float bx = static_cast<float>(reals[0]) +
      d_axle * static_cast<float>(std::cos(theta));
    const float by = static_cast<float>(reals[1]) +
      d_axle * static_cast<float>(std::sin(theta));
    if (bx < 0.0f || by < 0.0f) {
      failure = true;
      break;
    }

    // Turn the pose into a node, and check if it is valid
    index = NodeT::getIndex(
      static_cast<unsigned int>(bx),
      static_cast<unsigned int>(by),
      static_cast<unsigned int>(angle) % _dim_3_size);
    // Get the node from the graph
    if (node_getter(index, next)) {
      Coordinates initial_node_coords = (next == prev && !possible_nodes.empty()) ?
        possible_nodes.back().initial_coords : next->pose;
      proposed_coordinates = {bx, by, angle};
      next->setPose(proposed_coordinates);
      if (next->isNodeValid(_traverse_unknown, _collision_checker)) {
        // Save the node, and its previous coordinates in case we need to abort
        possible_nodes.emplace_back(next, initial_node_coords, proposed_coordinates);
        node_costs.emplace_back(next->getCost());
        prev = next;
      } else {
        // Abort
        next->setPose(initial_node_coords);
        failure = true;
        break;
      }
    } else {
      // Abort
      failure = true;
      break;
    }
  }

  if (!failure) {
    // We found 'a' valid expansion. Now to tell if its a quality option...
    const float max_cost = _search_info.analytic_expansion_max_cost;
    auto max_cost_it = std::max_element(node_costs.begin(), node_costs.end());
    if (max_cost_it != node_costs.end() && *max_cost_it > max_cost) {
      // If any element is above the comfortable cost limit, check edge cases:
      // (1) Check if goal is in greater than max_cost space requiring
      //  entering it, but only entering it on final approach, not in-and-out
      // (2) Checks if goal is in normal space, but enters costed space unnecessarily
      //  mid-way through, skirting obstacle or in non-globally confined space
      bool cost_exit_high_cost_region = false;
      for (auto iter = node_costs.rbegin(); iter != node_costs.rend(); ++iter) {
        const float & curr_cost = *iter;
        if (curr_cost <= max_cost) {
          cost_exit_high_cost_region = true;
        } else if (curr_cost > max_cost && cost_exit_high_cost_region) {
          failure = true;
          break;
        }
      }

      // (3) Handle exception: there may be no other option close to goal
      // if max cost is set too low (optional)
      if (failure) {
        if (d < 2.0f * M_PI * goal->motion_table.min_turning_radius &&
          _search_info.analytic_expansion_max_cost_override)
        {
          failure = false;
        }
      }
    }
  }

  // Reset to initial poses to not impact future searches
  for (auto it = possible_nodes.rbegin(); it != possible_nodes.rend(); ++it) {
    it->node->setPose(it->initial_coords);
  }

  if (failure) {
    return AnalyticExpansionNodes();
  }

  return possible_nodes;
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::NodePtr AnalyticExpansion<NodeT>::setAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal_node,
  const AnalyticExpansionNodes & expanded_nodes)
{
  _detached_nodes.clear();
  if (expanded_nodes.empty()) {
    return nullptr;
  }
  // Legitimate final path - set the parent relationships, states, and poses
  NodePtr prev = node;
  for (const auto & node_pose : expanded_nodes) {
    if (&node_pose == &expanded_nodes.back()) {
      continue;
    }
    auto n = node_pose.node;
    if (n->getIndex() == node->getIndex() ||
      n->getIndex() == goal_node->getIndex() ||
      n->wasVisited() ||
      n == prev)
    {
      _detached_nodes.push_back(std::make_unique<NodeT>(-1));
      n = _detached_nodes.back().get();
    }
    cleanNode(n);
    n->parent = prev;
    n->pose = node_pose.proposed_coords;
    n->visited();
    prev = n;
  }
  if (goal_node != prev) {
    goal_node->parent = prev;
    goal_node->pose = expanded_nodes.back().proposed_coords;
    cleanNode(goal_node);
    goal_node->visited();
  }
  return goal_node;
}

template<>
void AnalyticExpansion<NodeLattice>::cleanNode(const NodePtr & node)
{
  node->setMotionPrimitive(nullptr);
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::cleanNode(const NodePtr & /*expanded_nodes*/)
{
}

template<typename NodeT>
typename AnalyticExpansion<NodeT>::AnalyticExpansionNodes
AnalyticExpansion<NodeT>::getSingleGearAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal,
  const NodeGetter & node_getter)
{
  if (!_search_info.enable_single_gear_analytic || _motion_model != MotionModel::REEDS_SHEPP) {
    return AnalyticExpansionNodes();
  }
  const float d_axle = _search_info.rear_axle_offset;

  const float sx = node->pose.x;
  const float sy = node->pose.y;
  const float syaw = node->motion_table.getAngleFromBin(
    static_cast<unsigned int>(node->pose.theta));
  const float gx = goal->pose.x;
  const float gy = goal->pose.y;
  const float gyaw = node->motion_table.getAngleFromBin(
    static_cast<unsigned int>(goal->pose.theta));

  const float abs_dyaw = std::fabs(
    std::atan2(std::sin(gyaw - syaw), std::cos(gyaw - syaw)));
  if (abs_dyaw < 1.2f) {
    return AnalyticExpansionNodes();
  }

  const float R_min = node->motion_table.min_turning_radius;
  const float max_err_cells = ((abs_dyaw >= 2.2f) ? 0.25f : 0.142857f) * R_min;
  const float dr_step = 0.017857f * R_min;
  const float ds_unit = 0.035714f * R_min;
  const float weight = node->motion_table.cost_penalty;
  const float rev_pen = std::max(1.0f, _search_info.reverse_penalty);
  const float max_cost = _search_info.analytic_expansion_max_cost;
  static const float sqrt_2 = sqrtf(2.0f);
  const float max_dtheta = 0.0873f;

  const float srx = sx - d_axle * std::cos(syaw);
  const float sry = sy - d_axle * std::sin(syaw);

  AnalyticExpansionNodes best_nodes;
  float best_score = std::numeric_limits<float>::max();

  const std::vector<float> dth_offs = (abs_dyaw >= 2.2f) ?
    std::vector<float>{0.0f, -0.045f, 0.045f} :
  std::vector<float>{0.0f};
  const float ds_list[5] = {
    0.0f, -ds_unit, ds_unit, -2.0f * ds_unit, 2.0f * ds_unit};

  struct SamplePose
  {
    float bx;
    float by;
    float bth;
    float step_len;
  };
  std::vector<SamplePose> samples;
  samples.reserve(48);

  for (int gear : {1, -1}) {
    const float gear_f = static_cast<float>(gear);
    const float gear_mult = (_search_info.analytic_directional_scoring && gear < 0) ?
      rev_pen : 1.0f;
    const float ux_s = gear_f * std::cos(syaw);
    const float uy_s = gear_f * std::sin(syaw);

    for (int turn_sign : {1, -1}) {
      const float turn_f = static_cast<float>(turn_sign);
      const float sigma = gear_f * turn_f;

      for (float dth_off : dth_offs) {
        const float th_end = gyaw + dth_off;
        float dtheta = (th_end - syaw) * sigma;
        while (dtheta < 0.0f) {
          dtheta += static_cast<float>(2.0 * M_PI);
        }
        while (dtheta >= static_cast<float>(2.0 * M_PI)) {
          dtheta -= static_cast<float>(2.0 * M_PI);
        }
        if (dtheta < 1.1f || dtheta > static_cast<float>(1.75 * M_PI)) {
          continue;
        }

        const float grx = gx - d_axle * std::cos(th_end);
        const float gry = gy - d_axle * std::sin(th_end);
        const float ux_g = gear_f * std::cos(th_end);
        const float uy_g = gear_f * std::sin(th_end);
        const float det = ux_s * uy_g - uy_s * ux_g;

        for (int k_r = 0; k_r <= 12; ++k_r) {
          const float R = R_min + dr_step * static_cast<float>(k_r);
          if (R <= 1.0f) {
            continue;
          }
          const float sub_r_pen = 0.0f;
          const float arc_dx = turn_f * R * (std::sin(th_end) - std::sin(syaw));
          const float arc_dy = turn_f * R * (-std::cos(th_end) + std::cos(syaw));
          const float rx_rem = (grx - srx) - arc_dx;
          const float ry_rem = (gry - sry) - arc_dy;

          const float base_s0 = std::max(0.0f, rx_rem * ux_s + ry_rem * uy_s);
          const float base_s1 = std::max(0.0f, rx_rem * ux_g + ry_rem * uy_g);

          std::vector<std::pair<float, float>> cands_s01;
          cands_s01.reserve(11);
          for (float ds : ds_list) {
            if (base_s0 + ds >= 0.0f) {
              cands_s01.emplace_back(base_s0 + ds, 0.0f);
            }
            if (base_s1 + ds >= 0.0f) {
              cands_s01.emplace_back(0.0f, base_s1 + ds);
            }
          }
          if (std::fabs(det) > 0.1f) {
            const float s0_lin = (rx_rem * uy_g - ry_rem * ux_g) / det;
            const float s1_lin = (ux_s * ry_rem - uy_s * rx_rem) / det;
            if (s0_lin >= 0.0f && s1_lin >= 0.0f) {
              cands_s01.emplace_back(s0_lin, s1_lin);
            }
          }

          for (const auto & s01 : cands_s01) {
            const float s0 = s01.first;
            const float s1 = s01.second;
            const float ex = s0 * ux_s + s1 * ux_g - rx_rem;
            const float ey = s0 * uy_s + s1 * uy_g - ry_rem;
            const float err_cells = hypotf(ex, ey);
            if (err_cells > max_err_cells) {
              continue;
            }
            const float total_len = s0 + R * dtheta + s1;
            if (total_len > _search_info.analytic_expansion_max_length || total_len < sqrt_2) {
              continue;
            }
            const float min_possible_score =
              total_len * gear_mult + 3.0f * err_cells + sub_r_pen;
            if (min_possible_score >= best_score) {
              continue;
            }

            samples.clear();
            float cur_rx = srx;
            float cur_ry = sry;
            float cur_th = syaw;

            if (s0 > 1e-3f) {
              const unsigned int n0 = std::max(
                1u, static_cast<unsigned int>(std::ceil(s0 / sqrt_2)));
              const float step0 = s0 / static_cast<float>(n0);
              for (unsigned int k = 1; k <= n0; ++k) {
                const float frac = static_cast<float>(k) / static_cast<float>(n0);
                const float px = cur_rx + frac * s0 * ux_s;
                const float py = cur_ry + frac * s0 * uy_s;
                samples.push_back(
                  {px + d_axle * std::cos(cur_th),
                    py + d_axle * std::sin(cur_th), cur_th, step0});
              }
              cur_rx += s0 * ux_s;
              cur_ry += s0 * uy_s;
            }

            const float cx = cur_rx - turn_f * R * std::sin(cur_th);
            const float cy = cur_ry + turn_f * R * std::cos(cur_th);
            const unsigned int n_arc = std::max(
              1u, static_cast<unsigned int>(std::ceil(dtheta / max_dtheta)));
            const float step_arc = (R * dtheta) / static_cast<float>(n_arc);
            for (unsigned int k = 1; k <= n_arc; ++k) {
              const float th = cur_th +
                sigma * dtheta * (static_cast<float>(k) / static_cast<float>(n_arc));
              const float px = cx + turn_f * R * std::sin(th);
              const float py = cy - turn_f * R * std::cos(th);
              samples.push_back(
                {px + d_axle * std::cos(th),
                  py + d_axle * std::sin(th), th, step_arc});
            }
            cur_th = th_end;
            cur_rx = cx + turn_f * R * std::sin(cur_th);
            cur_ry = cy - turn_f * R * std::cos(cur_th);

            if (s1 > 1e-3f) {
              const unsigned int n1 = std::max(
                1u, static_cast<unsigned int>(std::ceil(s1 / sqrt_2)));
              const float step1 = s1 / static_cast<float>(n1);
              for (unsigned int k = 1; k <= n1; ++k) {
                const float frac = static_cast<float>(k) / static_cast<float>(n1);
                const float px = cur_rx + frac * s1 * ux_g;
                const float py = cur_ry + frac * s1 * uy_g;
                samples.push_back(
                  {px + d_axle * std::cos(cur_th),
                    py + d_axle * std::sin(cur_th), cur_th, step1});
              }
            }

            if (samples.empty()) {
              continue;
            }

            AnalyticExpansionNodes possible_nodes;
            possible_nodes.reserve(samples.size());
            std::vector<float> node_costs;
            node_costs.reserve(samples.size());
            NodePtr prev(node);
            NodePtr next(nullptr);
            bool failure = false;
            float cand_score = 3.0f * err_cells + sub_r_pen;

            for (const auto & sp : samples) {
              if (sp.bx < 0.0f || sp.by < 0.0f) {
                failure = true;
                break;
              }
              double theta = std::fmod(static_cast<double>(sp.bth), 2.0 * M_PI);
              if (theta < 0.0) {
                theta += 2.0 * M_PI;
              }
              float angle = static_cast<float>(node->motion_table.getAngle(theta));
              if (angle >= static_cast<float>(_dim_3_size)) {
                angle -= static_cast<float>(_dim_3_size);
              }
              const unsigned int angle_bin = static_cast<unsigned int>(angle) % _dim_3_size;
              const uint64_t index = NodeT::getIndex(
                static_cast<unsigned int>(sp.bx),
                static_cast<unsigned int>(sp.by),
                angle_bin);
              if (node_getter(index, next)) {
                Coordinates initial_node_coords = (next == prev && !possible_nodes.empty()) ?
                  possible_nodes.back().initial_coords : next->pose;
                Coordinates proposed_coordinates = {sp.bx, sp.by, angle};
                next->setPose(proposed_coordinates);
                if (next->isNodeValid(_traverse_unknown, _collision_checker)) {
                  const float c = next->getCost();
                  if (_search_info.analytic_directional_scoring &&
                    gear > 0 && abs_dyaw >= 2.2f && c > 0.0f)
                  {
                    next->setPose(initial_node_coords);
                    failure = true;
                    break;
                  }
                  possible_nodes.emplace_back(next, initial_node_coords, proposed_coordinates);
                  node_costs.emplace_back(c);
                  cand_score += sp.step_len * gear_mult * (1.0f + weight * (c / 252.0f));
                  prev = next;
                } else {
                  next->setPose(initial_node_coords);
                  failure = true;
                  break;
                }
              } else {
                failure = true;
                break;
              }
            }

            if (!failure && !node_costs.empty()) {
              auto max_cost_it = std::max_element(node_costs.begin(), node_costs.end());
              if (max_cost_it != node_costs.end() && *max_cost_it > max_cost) {
                bool cost_exit_high_cost_region = false;
                for (auto iter = node_costs.rbegin(); iter != node_costs.rend(); ++iter) {
                  const float & curr_cost = *iter;
                  if (curr_cost <= max_cost) {
                    cost_exit_high_cost_region = true;
                  } else if (curr_cost > max_cost && cost_exit_high_cost_region) {
                    failure = true;
                    break;
                  }
                }
                if (failure &&
                  total_len < 2.0f * static_cast<float>(M_PI) * R_min &&
                  _search_info.analytic_expansion_max_cost_override)
                {
                  failure = false;
                }
              }
            }

            for (auto it = possible_nodes.rbegin(); it != possible_nodes.rend(); ++it) {
              it->node->setPose(it->initial_coords);
            }

            if (!failure && cand_score < best_score) {
              best_score = cand_score;
              best_nodes = std::move(possible_nodes);
            }
          }
        }
      }
    }
  }

  return best_nodes;
}

template<>
typename AnalyticExpansion<Node2D>::AnalyticExpansionNodes AnalyticExpansion<Node2D>::
getSingleGearAnalyticPath(
  const NodePtr & /*node*/,
  const NodePtr & /*goal*/,
  const NodeGetter & /*node_getter*/)
{
  return AnalyticExpansionNodes();
}

template<>
typename AnalyticExpansion<Node2D>::AnalyticExpansionNodes AnalyticExpansion<Node2D>::
getAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal,
  const NodeGetter & node_getter,
  const ompl::base::StateSpacePtr & state_space)
{
  return AnalyticExpansionNodes();
}

template<>
typename AnalyticExpansion<Node2D>::NodePtr AnalyticExpansion<Node2D>::setAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal_node,
  const AnalyticExpansionNodes & expanded_nodes)
{
  return NodePtr(nullptr);
}

template<>
typename AnalyticExpansion<Node2D>::NodePtr AnalyticExpansion<Node2D>::tryAnalyticExpansion(
  const NodePtr & current_node, const NodePtr & goal_node,
  const NodeGetter & getter, int & analytic_iterations,
  int & closest_distance)
{
  return NodePtr(nullptr);
}

template class AnalyticExpansion<Node2D>;
template class AnalyticExpansion<NodeHybrid>;
template class AnalyticExpansion<NodeLattice>;

}  // namespace nav2_smac_planner
