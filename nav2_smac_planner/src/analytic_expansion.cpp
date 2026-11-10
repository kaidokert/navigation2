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
#include <cmath>
#include <vector>
#include <memory>
#include <iomanip>
#include <limits>
#include <sstream>

#include "nav2_smac_planner/analytic_expansion.hpp"

namespace nav2_smac_planner
{

namespace
{
template<typename NodeT>
int getIncomingGear(const NodeT * /*node*/)
{
  return 0;
}

template<>
int getIncomingGear<NodeHybrid>(const NodeHybrid * node)
{
  if (node == nullptr || node->parent == nullptr ||
    node->getMotionPrimitiveIndex() == std::numeric_limits<unsigned int>::max())
  {
    return 0;
  }
  const TurnDirection dir = node->getTurnDirection();
  if (dir == TurnDirection::UNKNOWN) {
    return 0;
  }
  return (dir == TurnDirection::REVERSE ||
         dir == TurnDirection::REV_LEFT ||
         dir == TurnDirection::REV_RIGHT) ? -1 : 1;
}

template<>
int getIncomingGear<NodeLattice>(const NodeLattice * node)
{
  if (node == nullptr || node->parent == nullptr) {
    return 0;
  }
  auto * mutable_node = const_cast<NodeLattice *>(node);
  if (mutable_node->getMotionPrimitive() == nullptr) {
    return 0;
  }
  return mutable_node->isBackward() ? -1 : 1;
}
}  // namespace

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
    if (_trace_callback) {
      ++_trace_opportunities;
    }
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

    if (current_node->parent == nullptr) {
      _gear_change_deferrals_used = 0;
    }

    // Always run the expansion on the first run in case there is a
    // trivial path to be found
    if (analytic_iterations <= 0) {
      // Reset the counter and try the analytic path expansion
      analytic_iterations = desired_iterations;
      _trace_candidate_turning_radius = current_node->motion_table.min_turning_radius;
      _trace_candidate_kind =
        _motion_model == MotionModel::DUBIN ? "dubins" : "reeds_shepp";
      AnalyticExpansionNodes analytic_nodes =
        getAnalyticPath(current_node, goal_node, getter, current_node->motion_table.state_space);
      if (!analytic_nodes.empty()) {
        uint64_t selected_attempt_id = _last_trace_attempt_id;
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
            _trace_candidate_turning_radius = test_node->motion_table.min_turning_radius;
            refined_analytic_nodes =
              getAnalyticPath(test_node, goal_node, getter, test_node->motion_table.state_space);
            if (refined_analytic_nodes.empty()) {
              break;
            }
            analytic_nodes = refined_analytic_nodes;
            selected_attempt_id = _last_trace_attempt_id;
            node = test_node;
          } else {
            break;
          }
        }

        const float gear_change_penalty = _search_info.gear_change_penalty;
        const bool gear_aware =
          (gear_change_penalty > 0.0f) || (_search_info.analytic_gear_change_deferrals > 0);
        const int initial_gear = getIncomingGear(node);
        // Count direction reversals along the shot (including the handoff from node),
        // from the sample-to-sample displacement measured against the heading at the
        // start of each step.
        auto countGearChanges = [&](const AnalyticExpansionNodes & expansion) {
            int changes = 0;
            int gear = initial_gear;
            float x = node->pose.x;
            float y = node->pose.y;
            float heading = node->motion_table.getAngleFromBin(node->pose.theta);
            for (const auto & sample : expansion) {
              const float dx = sample.proposed_coords.x - x;
              const float dy = sample.proposed_coords.y - y;
              if (hypotf(dx, dy) > 1e-3f) {
                const int step_gear =
                  (dx * cosf(heading) + dy * sinf(heading)) >= 0.0f ? 1 : -1;
                changes += (gear != 0 && step_gear != gear) ? 1 : 0;
                gear = step_gear;
              }
              x = sample.proposed_coords.x;
              y = sample.proposed_coords.y;
              heading = node->motion_table.getAngleFromBin(sample.proposed_coords.theta);
            }
            return changes;
          };

        // The analytic expansion can short-cut near obstacles when closer to a goal
        // So, we can attempt to refine it more by increasing the possible radius
        // higher than the minimum turning radius and use the best solution based on
        // a scoring function similar to that used in traveral cost estimation.
        auto scoringFn = [&](const AnalyticExpansionNodes & expansion) {
            if (expansion.size() < 2) {
              return std::numeric_limits<float>::max();
            }

            if (gear_aware) {
              // Samples are not evenly spaced: integrate over the real step lengths.
              float integrated = 0.0f;
              float px = node->pose.x;
              float py = node->pose.y;
              const float & cost_weight = expansion[0].node->motion_table.cost_penalty;
              for (const auto & sample : expansion) {
                const float step = hypotf(
                  sample.proposed_coords.x - px, sample.proposed_coords.y - py);
                const float normalized_cost = sample.cell_cost / 252.0f;
                const float cost_factor = _search_info.use_quadratic_cost_penalty ?
                  (1.0f + cost_weight * normalized_cost * normalized_cost) :
                  (1.0f + cost_weight * normalized_cost);
                integrated += step * cost_factor;
                px = sample.proposed_coords.x;
                py = sample.proposed_coords.y;
              }
              return integrated + gear_change_penalty * countGearChanges(expansion);
            }

            float score = 0.0;
            float normalized_cost = 0.0;
            // Analytic expansions are consistently spaced
            const float distance = hypotf(
              expansion[1].proposed_coords.x - expansion[0].proposed_coords.x,
              expansion[1].proposed_coords.y - expansion[0].proposed_coords.y);
            const float & weight = expansion[0].node->motion_table.cost_penalty;
            for (auto iter = expansion.begin(); iter != expansion.end(); ++iter) {
              normalized_cost = iter->cell_cost / 252.0f;
              // Search's Traversal Cost Function
              score += distance * (1.0 + weight * normalized_cost);
            }
            return score;
          };

        AnalyticExpansionNodes best_single_gear_nodes;
        float best_single_gear_score = std::numeric_limits<float>::max();
        uint64_t best_single_gear_attempt_id = 0;
        auto updateBestSingleGear = [&](
          const AnalyticExpansionNodes & candidate,
          float candidate_score,
          uint64_t attempt_id)
          {
            if (gear_aware && !candidate.empty() &&
              candidate_score < best_single_gear_score &&
              countGearChanges(candidate) == 0)
            {
              best_single_gear_nodes = candidate;
              best_single_gear_score = candidate_score;
              best_single_gear_attempt_id = attempt_id;
            }
          };

        float best_score = scoringFn(analytic_nodes);
        updateBestSingleGear(analytic_nodes, best_score, selected_attempt_id);
        if (_trace_callback) {
          std::ostringstream event;
          event << std::setprecision(9)
                << "{\"schema\":1,\"event\":\"score\",\"request_id\":"
                << _trace_request_id << ",\"attempt_id\":" << selected_attempt_id
                << ",\"score\":" << best_score
                << ",\"gear_changes\":" << countGearChanges(analytic_nodes)
                << ",\"gear_aware\":" << (gear_aware ? "true" : "false") << "}";
          _trace_callback(event.str());
        }
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
          _trace_candidate_turning_radius = min_turn_rad;
          refined_analytic_nodes = getAnalyticPath(node, goal_node, getter, state_space);
          score = scoringFn(refined_analytic_nodes);
          const uint64_t attempt_id = _last_trace_attempt_id;
          if (score <= best_score) {
            analytic_nodes = refined_analytic_nodes;
            best_score = score;
            selected_attempt_id = attempt_id;
          }
          updateBestSingleGear(refined_analytic_nodes, score, attempt_id);
          if (_trace_callback) {
            std::ostringstream event;
            event << std::setprecision(9)
                  << "{\"schema\":1,\"event\":\"score\",\"request_id\":"
                  << _trace_request_id << ",\"attempt_id\":" << attempt_id
                  << ",\"score\":" << score
                  << ",\"gear_changes\":" << countGearChanges(refined_analytic_nodes)
                  << ",\"gear_aware\":" << (gear_aware ? "true" : "false") << "}";
            _trace_callback(event.str());
          }
        }

        if (gear_aware && _motion_model != MotionModel::DUBIN &&
          countGearChanges(analytic_nodes) > 0)
        {
          // Reeds-Shepp returns only the shortest curve, which may reverse. Offer
          // forward-only and reverse-only curves over the same radius ladder.
          const float base_radius = node->motion_table.min_turning_radius;
          for (const bool reverse_only : {false, true}) {
            _trace_candidate_kind =
              reverse_only ? "single_gear_reverse" : "single_gear_forward";
            for (float radius = base_radius; radius < 4.0f * base_radius; radius += 0.5f) {
              auto dubins_space = std::make_shared<ompl::base::DubinsStateSpace>(radius);
              _trace_candidate_turning_radius = radius;
              refined_analytic_nodes =
                getAnalyticPath(node, goal_node, getter, dubins_space, reverse_only);
              score = scoringFn(refined_analytic_nodes);
              const uint64_t attempt_id = _last_trace_attempt_id;
              if (score < best_score) {
                analytic_nodes = refined_analytic_nodes;
                best_score = score;
                selected_attempt_id = attempt_id;
              }
              updateBestSingleGear(refined_analytic_nodes, score, attempt_id);
              if (_trace_callback) {
                std::ostringstream event;
                event << std::setprecision(9)
                      << "{\"schema\":1,\"event\":\"score\",\"request_id\":"
                      << _trace_request_id << ",\"attempt_id\":" << attempt_id
                      << ",\"score\":" << score
                      << ",\"gear_changes\":" << countGearChanges(refined_analytic_nodes)
                      << ",\"gear_aware\":" << (gear_aware ? "true" : "false") << "}";
                _trace_callback(event.str());
              }
            }
          }
          _trace_candidate_kind = "reeds_shepp";
        }

        // Let the search try for a one-gear route before settling for a reversal.
        // If a valid 0-reversal candidate exists from this node, fall back to it;
        // otherwise defer and let A* advance.
        bool used_single_gear_fallback = false;
        if (gear_aware && countGearChanges(analytic_nodes) > 0 &&
          _gear_change_deferrals_used < _search_info.analytic_gear_change_deferrals)
        {
          ++_gear_change_deferrals_used;
          if (!best_single_gear_nodes.empty()) {
            analytic_nodes = best_single_gear_nodes;
            best_score = best_single_gear_score;
            selected_attempt_id = best_single_gear_attempt_id;
            used_single_gear_fallback = true;
          } else {
            analytic_iterations--;
            return NodePtr(nullptr);
          }
        }

        if (_trace_callback) {
          std::ostringstream event;
          event << std::setprecision(9)
                << "{\"schema\":1,\"event\":\"selection\",\"request_id\":"
                << _trace_request_id << ",\"attempt_id\":" << selected_attempt_id
                << ",\"score\":" << best_score << ",\"selected_by\":\""
                << (used_single_gear_fallback ? "gear_aware_single_gear_fallback" :
                   (gear_aware ? "gear_aware_min_score" : "legacy_min_score"))
                << "\"}";
          _trace_callback(event.str());
        }

        return setAnalyticPath(node, goal_node, analytic_nodes);
      }
    } else {
      if (_trace_callback) {
        ++_trace_schedule_skips;
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
  const ompl::base::StateSpacePtr & state_space,
  const bool reverse_only)
{
  const uint64_t attempt_id = ++_trace_attempt_id;
  _last_trace_attempt_id = attempt_id;
  if (_trace_callback) {
    ++_trace_request_attempts;
  }
  ompl::base::ScopedState<> from(state_space), to(state_space), s(state_space);
  from[0] = node->pose.x;
  from[1] = node->pose.y;
  // A reverse-only curve is a forward curve of the vehicle turned around.
  const double gear_heading_offset = reverse_only ? M_PI : 0.0;
  from[2] = node->motion_table.getAngleFromBin(node->pose.theta) + gear_heading_offset;
  to[0] = goal->pose.x;
  to[1] = goal->pose.y;
  to[2] = node->motion_table.getAngleFromBin(goal->pose.theta) + gear_heading_offset;
  const float distance = state_space->distance(from(), to());
  std::string trace_segments = "[]";
  bool has_rejected_sample = false;
  int rejected_sample_index = -1;
  double rejected_sample_x = 0.0;
  double rejected_sample_y = 0.0;
  double rejected_sample_yaw = 0.0;
  float rejected_sample_cost = -1.0f;
  auto publish_attempt = [this, attempt_id, node, goal, distance, &has_rejected_sample,
      &rejected_sample_index, &rejected_sample_x, &rejected_sample_y, &rejected_sample_yaw,
      &rejected_sample_cost](
    const char * outcome, size_t sample_count, size_t valid_count,
    const std::string & segments)
    {
      if (!_trace_callback) {
        return;
      }
      std::ostringstream event;
      event << std::setprecision(9)
            << "{\"schema\":1,\"event\":\"candidate\",\"request_id\":"
            << _trace_request_id << ",\"attempt_id\":" << attempt_id
            << ",\"outcome\":\"" << outcome << "\",\"kind\":\"" << _trace_candidate_kind
            << "\",\"start_grid\":["
            << node->pose.x << "," << node->pose.y << ","
            << node->motion_table.getAngleFromBin(node->pose.theta) << "],\"goal_grid\":["
            << goal->pose.x << "," << goal->pose.y << ","
            << node->motion_table.getAngleFromBin(goal->pose.theta) << "],\"distance\":"
            << distance << ",\"turning_radius_cells\":"
            << _trace_candidate_turning_radius << ",\"sample_count\":"
            << sample_count << ",\"valid_sample_count\":" << valid_count
            << ",\"segments\":" << segments << ",\"rejected_sample\":";
      if (has_rejected_sample) {
        event << "{\"index\":" << rejected_sample_index << ",\"x\":"
              << rejected_sample_x << ",\"y\":" << rejected_sample_y
              << ",\"yaw\":" << rejected_sample_yaw << ",\"cost\":";
        if (std::isfinite(rejected_sample_cost) && rejected_sample_cost >= 0.0f) {
          event << rejected_sample_cost;
        } else {
          event << "null";
        }
        event << "}";
      } else {
        event << "null";
      }
      event << "}";
      _trace_callback(event.str());
    };
  const float d = distance;

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
    publish_attempt("distance_gate", 0, 0, "[]");
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
    if (_trace_callback) {
      std::ostringstream segments;
      segments << std::setprecision(9) << "[";
      bool first = true;
      for (int i = 0; i < 5; ++i) {
        if (std::abs(rs_path.length_[i]) < 1e-6) {
          continue;
        }
        if (!first) {segments << ",";}
        first = false;
        const char * type = "unknown";
        switch (rs_path.type_[i]) {
          case ompl::base::ReedsSheppStateSpace::RS_LEFT: type = "left"; break;
          case ompl::base::ReedsSheppStateSpace::RS_STRAIGHT: type = "straight"; break;
          case ompl::base::ReedsSheppStateSpace::RS_RIGHT: type = "right"; break;
          default: break;
        }
        segments << "{\"type\":\"" << type << "\",\"signed_length\":"
                 << rs_path.length_[i] << "}";
      }
      segments << "]";
      trace_segments = segments.str();
    }
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
    if (_trace_callback) {
      std::ostringstream segments;
      segments << std::setprecision(9) << "[";
      for (int i = 0; i < 3; ++i) {
        if (i > 0) {segments << ",";}
        const char * type = "unknown";
        switch ((*dubins_path.type_)[i]) {
          case ompl::base::DubinsStateSpace::DUBINS_LEFT: type = "left"; break;
          case ompl::base::DubinsStateSpace::DUBINS_STRAIGHT: type = "straight"; break;
          case ompl::base::DubinsStateSpace::DUBINS_RIGHT: type = "right"; break;
          default: break;
        }
        segments << "{\"type\":\"" << type << "\",\"signed_length\":"
                 << (reverse_only ? -1.0 : 1.0) * dubins_path.length_[i] << "}";
      }
      segments << "]";
      trace_segments = segments.str();
    }
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
  const char * rejection_reason = "none";
  const char * candidate_outcome = "valid";
  const size_t planned_sample_count = fractions.size();
  std::vector<float> node_costs;
  node_costs.reserve(fractions.size());

  // Check intermediary poses (non-goal, non-start)
  for (size_t sample_index = 0; sample_index < fractions.size(); ++sample_index) {
    const float interp_fraction = fractions[sample_index];
    state_space->interpolate(from(), to(), interp_fraction, s());
    reals = s.reals();
    reals[2] -= gear_heading_offset;

    const double max_x =
      static_cast<double>(_collision_checker->getCostmap()->getSizeInCellsX());
    const double max_y =
      static_cast<double>(_collision_checker->getCostmap()->getSizeInCellsY());
    if (!std::isfinite(reals[0]) || !std::isfinite(reals[1]) || !std::isfinite(reals[2]) ||
      reals[0] < 0.0 || reals[1] < 0.0 || reals[0] >= max_x || reals[1] >= max_y ||
      static_cast<float>(reals[0]) >= static_cast<float>(max_x) ||
      static_cast<float>(reals[1]) >= static_cast<float>(max_y))
    {
      failure = true;
      rejection_reason = "invalid_sample";
      has_rejected_sample = true;
      rejected_sample_index = static_cast<int>(sample_index);
      rejected_sample_x = reals[0];
      rejected_sample_y = reals[1];
      rejected_sample_yaw = reals[2];
      break;
    }

    // Make sure in range [0, 2PI)
    theta = reals[2];
    while (theta < 0.0) {
      theta += 2.0 * M_PI;
    }
    while (theta >= 2.0 * M_PI) {
      theta -= 2.0 * M_PI;
    }
    angle = node->motion_table.getAngle(theta);
    if (angle >= static_cast<float>(_dim_3_size)) {
      angle -= static_cast<float>(_dim_3_size);
    }
    if (angle < 0.0f) {
      angle = 0.0f;
    }

    // Turn the pose into a node, and check if it is valid
    index = NodeT::getIndex(
      static_cast<unsigned int>(reals[0]),
      static_cast<unsigned int>(reals[1]),
      static_cast<unsigned int>(angle));
    // Get the node from the graph
    if (node_getter(index, next)) {
      Coordinates initial_node_coords = (next == prev && !possible_nodes.empty()) ?
        possible_nodes.back().initial_coords : next->pose;
      proposed_coordinates = {static_cast<float>(reals[0]), static_cast<float>(reals[1]), angle};
      next->setPose(proposed_coordinates);
      if (next->isNodeValid(_traverse_unknown, _collision_checker)) {
        // Save the node, its previous coordinates, and the evaluated cell cost
        const float evaluated_cost = next->getCost();
        possible_nodes.emplace_back(
          next, initial_node_coords, proposed_coordinates, evaluated_cost);
        node_costs.emplace_back(evaluated_cost);
        prev = next;
      } else {
        // Abort
        rejected_sample_cost = next->getCost();
        next->setPose(initial_node_coords);
        failure = true;
        rejection_reason = "invalid_sample";
        has_rejected_sample = true;
        rejected_sample_index = static_cast<int>(sample_index);
        rejected_sample_x = reals[0];
        rejected_sample_y = reals[1];
        rejected_sample_yaw = reals[2];
        break;
      }
    } else {
      // Abort
      failure = true;
      rejection_reason = "node_lookup_failed";
      has_rejected_sample = true;
      rejected_sample_index = static_cast<int>(sample_index);
      rejected_sample_x = reals[0];
      rejected_sample_y = reals[1];
      rejected_sample_yaw = reals[2];
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
          rejection_reason = "max_cost";
          const size_t sample_index = static_cast<size_t>(
            std::distance(node_costs.begin(), iter.base()) - 1);
          if (sample_index < possible_nodes.size()) {
            has_rejected_sample = true;
            rejected_sample_index = static_cast<int>(sample_index);
            rejected_sample_x = possible_nodes[sample_index].proposed_coords.x;
            rejected_sample_y = possible_nodes[sample_index].proposed_coords.y;
            rejected_sample_yaw = node->motion_table.getAngleFromBin(
              possible_nodes[sample_index].proposed_coords.theta);
            rejected_sample_cost = curr_cost;
          }
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
          candidate_outcome = "valid_max_cost_override";
          rejection_reason = "none";
          has_rejected_sample = false;
          rejected_sample_index = -1;
          rejected_sample_cost = -1.0f;
        }
      }
    }
  }

  // Reset to initial poses to not impact future searches
  for (auto it = possible_nodes.rbegin(); it != possible_nodes.rend(); ++it) {
    it->node->setPose(it->initial_coords);
  }

  if (failure) {
    publish_attempt(rejection_reason, planned_sample_count, possible_nodes.size(),
      trace_segments);
    return AnalyticExpansionNodes();
  }

  publish_attempt(candidate_outcome, planned_sample_count, possible_nodes.size(), trace_segments);
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

template<>
void AnalyticExpansion<NodeHybrid>::cleanNode(const NodePtr & node)
{
  node->setMotionPrimitiveIndex(
    std::numeric_limits<unsigned int>::max(), TurnDirection::UNKNOWN);
}

template<typename NodeT>
void AnalyticExpansion<NodeT>::cleanNode(const NodePtr & /*expanded_nodes*/)
{
}

template<>
typename AnalyticExpansion<Node2D>::AnalyticExpansionNodes AnalyticExpansion<Node2D>::
getAnalyticPath(
  const NodePtr & node,
  const NodePtr & goal,
  const NodeGetter & node_getter,
  const ompl::base::StateSpacePtr & state_space,
  const bool /*reverse_only*/)
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
