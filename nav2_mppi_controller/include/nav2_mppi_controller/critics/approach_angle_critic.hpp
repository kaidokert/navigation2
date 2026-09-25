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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__APPROACH_ANGLE_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__APPROACH_ANGLE_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi::critics
{

/**
 * @class mppi::critics::ApproachAngleCritic
 * @brief Final-approach heading shaping for car-like robots
 * (Layer 0 of the cusp-handoff plan). Scores each trajectory's TERMINAL
 * yaw against data.goal's orientation. Note data.goal is the FULL plan's
 * final pose (path_handler getTransformedGoal) — it equals the mission
 * goal by construction, and unlike data.path's last pose it is immune to
 * inversion truncation and ghost extension (GoalAngleCritic's failure
 * mode). Hard-gated: exactly zero cost beyond activation_distance,
 * linearly ramped inside it so activation is continuous.
 * KNOWN LIMITS (review 2026-09-25, why this ships disabled): terminal
 * yaw of a ~2s horizon lies past the goal at approach speed, and the
 * yaw cost is symmetric in vx sign, so near-zero-mean vx sampling
 * cancels its gradient — it cannot rescue a sub-feasible terminal plan
 * stub. Revisit after terminal-stub handling lands.
 */
class ApproachAngleCritic : public CriticFunction
{
public:
  /**
    * @brief Initialize critic
    */
  void initialize() override;

  /**
   * @brief Evaluate terminal-heading cost during the final approach
   * @param data Data to use (costs added in place)
   */
  void score(CriticData & data) override;

protected:
  float weight_{0};
  unsigned int power_{0};
  float activation_distance_{0};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__APPROACH_ANGLE_CRITIC_HPP_
