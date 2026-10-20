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

#ifndef NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
#define NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <optional>
#include "nav2_core/controller_exceptions.hpp"

#include "nav2_mppi_controller/models/control_sequence.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/constraints.hpp"

// xtensor creates warnings that needs to be ignored as we are building with -Werror
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <xtensor/xmath.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#pragma GCC diagnostic pop

#include "nav2_mppi_controller/tools/parameters_handler.hpp"

namespace mppi
{

/**
 * @class mppi::MotionModel
 * @brief Abstract motion model for modeling a vehicle
 */
class MotionModel
{
public:
  /**
    * @brief Constructor for mppi::MotionModel
    */
  MotionModel() = default;

  /**
    * @brief Destructor for mppi::MotionModel
    */
  virtual ~MotionModel() = default;

  /**
    * @brief Initialize motion model on bringup and set required variables
    * @param control_constraints Constraints on control
    * @param model_dt duration of a time step
    */
  void initialize(const models::ControlConstraints & control_constraints, float model_dt)
  {
    control_constraints_ = control_constraints;
    model_dt_ = model_dt;
  }

  /**
   * @brief With input velocities, find the vehicle's output velocities
   * @param state Contains control velocities to use to populate vehicle velocities
   */
  virtual void predict(models::State & state)
  {
    // Previously completed via tensor views, but found to be 10x slower
    // using namespace xt::placeholders;  // NOLINT
    // xt::noalias(xt::view(state.vx, xt::all(), xt::range(1, _))) =
    //   xt::noalias(xt::view(state.cvx, xt::all(), xt::range(0, -1)));
    // xt::noalias(xt::view(state.wz, xt::all(), xt::range(1, _))) =
    //   xt::noalias(xt::view(state.cwz, xt::all(), xt::range(0, -1)));
    // if (isHolonomic()) {
    //   xt::noalias(xt::view(state.vy, xt::all(), xt::range(1, _))) =
    //     xt::noalias(xt::view(state.cvy, xt::all(), xt::range(0, -1)));
    // }

    const bool is_holo = isHolonomic();
    float max_delta_vx = model_dt_ * control_constraints_.ax_max;
    float min_delta_vx = model_dt_ * control_constraints_.ax_min;
    float max_delta_vy = model_dt_ * control_constraints_.ay_max;
    float min_delta_vy = model_dt_ * control_constraints_.ay_min;
    float max_delta_wz = model_dt_ * control_constraints_.az_max;
    for (unsigned int i = 0; i != state.vx.shape(0); i++) {
      float vx_last = state.vx(i, 0);
      float vy_last = state.vy(i, 0);
      float wz_last = state.wz(i, 0);
      for (unsigned int j = 1; j != state.vx.shape(1); j++) {
        float & cvx_curr = state.cvx(i, j - 1);
        if (vx_last > 0) {
          cvx_curr = std::clamp(cvx_curr, vx_last + min_delta_vx, vx_last + max_delta_vx);
        } else {
          cvx_curr = std::clamp(cvx_curr, vx_last - max_delta_vx, vx_last - min_delta_vx);
        }
        state.vx(i, j) = cvx_curr;
        vx_last = cvx_curr;

        float & cwz_curr = state.cwz(i, j - 1);
        cwz_curr = std::clamp(cwz_curr, wz_last - max_delta_wz, wz_last + max_delta_wz);
        state.wz(i, j) = cwz_curr;
        wz_last = cwz_curr;

        if (is_holo) {
          float & cvy_curr = state.cvy(i, j - 1);
          if (vy_last > 0) {
            cvy_curr = std::clamp(cvy_curr, vy_last + min_delta_vy, vy_last + max_delta_vy);
          } else {
            cvy_curr = std::clamp(cvy_curr, vy_last - max_delta_vy, vy_last - min_delta_vy);
          }
          state.vy(i, j) = cvy_curr;
          vy_last = cvy_curr;
        }
      }
    }
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  virtual bool isHolonomic() = 0;

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   * @param control_sequence Control sequence to apply constraints to
   */
  virtual void applyConstraints(models::ControlSequence & /*control_sequence*/) {}

protected:
  float model_dt_{0.0};
  models::ControlConstraints control_constraints_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f};
};

/**
 * @class mppi::AckermannMotionModel
 * @brief Ackermann motion model
 */
class AckermannMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::AckermannMotionModel
    */
  explicit AckermannMotionModel(ParametersHandler * param_handler, const std::string & name)
  {
    auto getParam = param_handler->getParamGetter(name + ".AckermannConstraints");
    getParam(min_turning_r_, "min_turning_r", 0.2);
    // Optional measured plant response, not a change to command constraints.
    // Configure before activation; zero preserves the existing model exactly.
    getParam(response_radius_offset_, "response_radius_offset", 0.0, ParameterType::Static);
    if (response_radius_offset_ < 0.0f) {
      throw std::invalid_argument("Ackermann response_radius_offset must be nonnegative");
    }
    getParam(predict_steering_conditioning_, "predict_steering_conditioning", false,
      ParameterType::Static);
    if (predict_steering_conditioning_) {
      getParam(wheelbase_, "steering_wheelbase", 0.213, ParameterType::Static);
      getParam(steering_max_angle_, "steering_max_angle", 0.576, ParameterType::Static);
      getParam(steering_vx_threshold_, "steering_vx_threshold", 0.05, ParameterType::Static);
      getParam(steering_rate_base_, "steering_rate_base", 0.5, ParameterType::Static);
      getParam(steering_rate_gain_, "steering_rate_gain", 10.0, ParameterType::Static);
      getParam(steering_rate_max_, "steering_rate_max", 6.0, ParameterType::Static);
      getParam(steering_update_period_, "steering_update_period", 0.1, ParameterType::Static);
      // Separate actuator slew from the command conditioning filter. Zero
      // preserves the legacy predictor; configure from measured plant limits.
      getParam(steering_actuator_rate_, "steering_actuator_rate", 0.0, ParameterType::Static);
      if (wheelbase_ <= 0 || steering_max_angle_ <= 0 || steering_max_angle_ >= M_PI_2 ||
        steering_vx_threshold_ < 0 || steering_rate_base_ < 0 || steering_rate_gain_ < 0 ||
        steering_rate_max_ <= 0 || steering_update_period_ <= 0 || steering_actuator_rate_ < 0)
      {
        throw std::invalid_argument("Invalid Ackermann steering prediction parameters");
      }
    }
  }

  // Explicit feedback takes precedence over the moving-state kinematic estimate.
  // Clear on stale/missing feedback; standstill cannot infer steering from yaw/vx.
  void setInitialSteering(const std::optional<float> & angle)
  {
    // The build uses -ffast-math: inspect IEEE bits rather than isfinite.
    if (angle) {
      uint32_t bits;
      std::memcpy(&bits, &*angle, sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u || std::abs(*angle) > steering_max_angle_) {
        initial_steering_.reset();
        throw nav2_core::NoValidControl("Invalid measured steering feedback");
      }
    }
    initial_steering_ = angle;
  }

  bool predictsSteeringConditioning() const {return predict_steering_conditioning_;}
  std::optional<float> initialSteering() const {return initial_steering_;}
  std::optional<float> initialConditionedSteering() const {return initial_conditioned_steering_;}
  bool requiresConditionedSteering() const
  {return predict_steering_conditioning_ && steering_actuator_rate_ > 0;}

  void setInitialConditionedSteering(const std::optional<float> & angle)
  {
    initial_conditioned_steering_.reset();
    if (angle) {
      uint32_t bits;
      std::memcpy(&bits, &*angle, sizeof(bits));
      if ((bits & 0x7f800000u) == 0x7f800000u || std::abs(*angle) > steering_max_angle_) {
        throw nav2_core::NoValidControl("Invalid conditioned steering feedback");
      }
    }
    initial_conditioned_steering_ = angle;
  }

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return false;
  }

  /**
   * @brief Apply hard vehicle constraints to a control sequence
   * @param control_sequence Control sequence to apply constraints to
   */
  void applyConstraints(models::ControlSequence & control_sequence) override
  {
    auto & vx = control_sequence.vx;
    auto & wz = control_sequence.wz;

    auto view = xt::masked_view(wz, (xt::fabs(vx) / xt::fabs(wz)) < min_turning_r_);
    view = xt::sign(wz) * xt::fabs(vx) / min_turning_r_;
  }

  /**
   * @brief Predict a calibrated excess turning radius without altering commands.
   * For commanded radius |vx/wz|, the predicted radius is that radius plus
   * the measured offset. Zero-speed/zero-yaw pairs remain finite. Callers
   * exclude measured initial state; raw sampled commands are never rewritten.
   */
  template<typename VelocityX, typename VelocityYaw>
  void applyResponse(VelocityX && vx, VelocityYaw && wz) const
  {
    if (response_radius_offset_ > 0.0f) {
      wz = wz * xt::fabs(vx) / xt::maximum(
        xt::fabs(vx) + response_radius_offset_ * xt::fabs(wz), 1e-9f);
    }
  }

  /**
   * @brief Predict rollout velocities with the Ackermann turning limit
   * enforced per step. Upstream applied the limit only to the winning
   * control sequence, so the optimizer scored pivot fantasies
   * (|vx|/|wz| < r_min) the vehicle cannot execute (stage-3,
   * notes/cusp_handoff_fix_plan.md). Review-amended (2026-09-25): the
   * projection is applied to the CONTROLS (cwz, written back — so scoring,
   * averaging and the importance weights all see the same feasible values),
   * inside the acceleration chain (wz_last carries projected values, so
   * az_max still holds), and never touches column 0 (measured state).
   * @param state State with sampled controls to roll out
   */
  void predict(models::State & state) override
  {
    MotionModel::predict(state);

    if (predict_steering_conditioning_) {
      const float dt = std::clamp(model_dt_, 0.005f, 0.15f);
      const float command_limit = std::min(steering_max_angle_,
          std::atan(wheelbase_ / min_turning_r_));
      for (unsigned int b = 0; b < state.vx.shape(0); ++b) {
        float angle;
        if (initial_steering_) {
          angle = *initial_steering_;
        } else if (std::abs(state.vx(b, 0)) > 0.001f) {
          angle = std::atan(wheelbase_ * state.wz(b, 0) / state.vx(b, 0));
        } else {
          throw nav2_core::NoValidControl("Steering prediction requires feedback at standstill");
        }
        float physical_angle = angle;
        if (requiresConditionedSteering()) {
          if (!initial_conditioned_steering_) {
            throw nav2_core::NoValidControl(
                "Actuator prediction requires conditioned target feedback");
          }
          angle = *initial_conditioned_steering_;
        }
        for (unsigned int j = 1; j < state.vx.shape(1); ++j) {
          const float speed = std::abs(state.vx(b, j - 1));
          const float requested = std::abs(state.cvx(b, j - 1)) > 1e-9f ?
            std::atan(wheelbase_ * state.cwz(b, j - 1) / state.cvx(b, j - 1)) : 0.0f;
          const float raw = std::clamp(requested, -command_limit, command_limit);
          const float alpha = steering_vx_threshold_ > 0 ?
            std::min(1.0f, speed / steering_vx_threshold_) : 1.0f;
          // Match the hardware update at its period; fractional model steps
          // use an equivalent exponential blend, not extra full-strength ticks.
          const float blend = 1.0f - std::pow(1.0f - alpha, model_dt_ / steering_update_period_);
          const float target = std::clamp(blend * raw + (1.0f - blend) * angle,
              -steering_max_angle_, steering_max_angle_);
          const float rate = std::min(steering_rate_max_,
              steering_rate_base_ + steering_rate_gain_ * speed);
          angle += std::clamp(target - angle, -rate * dt, rate * dt);
          if (steering_actuator_rate_ > 0) {
            const float servo_delta = steering_actuator_rate_ * model_dt_;
            physical_angle += std::clamp(angle - physical_angle, -servo_delta, servo_delta);
          } else {
            physical_angle = angle;
          }
          state.wz(b, j) = state.vx(b, j) * std::tan(physical_angle) / wheelbase_;
        }
      }
      using namespace xt::placeholders;  // NOLINT
      applyResponse(xt::view(state.vx, xt::all(), xt::range(1, _)),
        xt::view(state.wz, xt::all(), xt::range(1, _)));
      return;
    }

    // Project ROLLOUT wz (columns 1+; column 0 is the measured state) onto
    // the feasible Ackermann cone so scoring only rewards executable motion.
    //
    // DELIBERATELY NOT written back into cwz, and deliberately post-hoc
    // rather than inside the acceleration chain: the "consistent" variant
    // (per-step cone clamp written back into the controls, review
    // 2026-09-25) was implemented and EMPIRICALLY FALSIFIED — at maneuver
    // start vx≈0 collapses the cone, the zeroed samples then dominate the
    // control-sequence average, and the mean wz can never leave zero: a
    // self-locking straight line (M5 drove 1.9m AWAY from its goal, trio
    // gate 2/15 vs 6/15 with this version; radius change ruled out by a
    // one-variable arm). The scoring/averaging incoherence this leaves
    // (averaged cwz can exceed the scored wz; final applyConstraints clamps
    // the winner) is a known, documented trade.
    using namespace xt::placeholders;  // NOLINT
    auto vx = xt::view(state.vx, xt::all(), xt::range(1, _));
    auto wz = xt::view(state.wz, xt::all(), xt::range(1, _));

    auto view = xt::masked_view(wz, (xt::fabs(vx) / xt::fabs(wz)) < min_turning_r_);
    view = xt::sign(wz) * xt::fabs(vx) / min_turning_r_;
    applyResponse(vx, wz);
  }

  /**
   * @brief Get minimum turning radius of ackermann drive
   * @return Minimum turning radius
   */
  float getMinTurningRadius() {return min_turning_r_;}

private:
  float min_turning_r_{0};
  float response_radius_offset_{0};
  bool predict_steering_conditioning_{false};
  float wheelbase_{0.213f};
  float steering_max_angle_{0.576f};
  float steering_vx_threshold_{0.05f};
  float steering_rate_base_{0.5f};
  float steering_rate_gain_{10.0f};
  float steering_rate_max_{6.0f};
  float steering_update_period_{0.1f};
  float steering_actuator_rate_{0.0f};
  std::optional<float> initial_steering_;
  std::optional<float> initial_conditioned_steering_;
};

/**
 * @class mppi::DiffDriveMotionModel
 * @brief Differential drive motion model
 */
class DiffDriveMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::DiffDriveMotionModel
    */
  DiffDriveMotionModel() = default;

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return false;
  }
};

/**
 * @class mppi::OmniMotionModel
 * @brief Omnidirectional motion model
 */
class OmniMotionModel : public MotionModel
{
public:
  /**
    * @brief Constructor for mppi::OmniMotionModel
    */
  OmniMotionModel() = default;

  /**
   * @brief Whether the motion model is holonomic, using Y axis
   * @return Bool If holonomic
   */
  bool isHolonomic() override
  {
    return true;
  }
};
}  // namespace mppi

#endif  // NAV2_MPPI_CONTROLLER__MOTION_MODELS_HPP_
