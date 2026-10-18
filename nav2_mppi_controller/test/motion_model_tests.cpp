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

#include <chrono>
#include <thread>
#include <limits>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/control_sequence.hpp"

// Tests motion models

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

using namespace mppi;  // NOLINT

TEST(MotionModelTests, ConditionedAckermannSeparatesServoSlewFromCommandFilter)
{
  // Office103358: physical steering remains positive after the requested
  // forward-turn angle becomes negative. SDF steering velocity limit is1rad/s,
  // independently of the speed-dependent command filter (1.5rad/s at0.1m/s).
  for (const float velocity : {-0.1f, 0.1f}) {
    for (const float response_offset : {0.0f, 0.093f}) {
      auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("servo_slew");
      node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
      node->declare_parameter("test.AckermannConstraints.steering_actuator_rate", 1.0);
      node->declare_parameter("test.AckermannConstraints.response_radius_offset",
        static_cast<double>(response_offset));
      ParametersHandler params(node);
      AckermannMotionModel model(&params, "test");
      models::ControlConstraints constraints{};
      constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
      model.initialize(constraints, 0.1);
      model.setInitialSteering(0.35f);
      model.setInitialConditionedSteering(0.35f);
      models::State state;
      state.reset(1, 5);
      state.vx(0, 0) = velocity;
      state.wz(0, 0) = velocity * std::tan(0.35f) / 0.213f;
      state.cvx.fill(velocity);
      state.cwz.fill(velocity * std::tan(-0.48f) / 0.213f);
      const auto controls = state.cwz;
      model.predict(state);
      for (unsigned int j = 1; j < 5; ++j) {
        const float physical_angle = 0.35f - 0.1f * j;
        const float nominal_w = velocity * std::tan(physical_angle) / 0.213f;
        const float expected_w = std::copysign(
          std::abs(velocity) / (std::abs(velocity / nominal_w) + response_offset), nominal_w);
        EXPECT_NEAR(state.wz(0, j), expected_w, 1e-6);
      }
      EXPECT_EQ(state.cwz, controls);
      EXPECT_EQ(state.vx(0, 0), velocity);
    }
  }
}

TEST(MotionModelTests, ServoPredictionKeepsConditionerMemorySeparateAtStandstill)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("distinct_servo_state");
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("test.AckermannConstraints.steering_actuator_rate", 1.0);
  ParametersHandler params(node);
  AckermannMotionModel model(&params, "test");
  models::ControlConstraints constraints{};
  constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
  model.initialize(constraints, 0.1);
  model.setInitialSteering(0.0f);
  models::State state;
  state.reset(1, 2);
  state.cvx.fill(0.05f);state.cwz.fill(-0.1f);
  EXPECT_THROW(model.predict(state), nav2_core::NoValidControl);
  model.setInitialConditionedSteering(0.3f);
  state.reset(1, 2);
  state.cvx.fill(0.05f);state.cwz.fill(-0.1f);
  model.predict(state);
  // At zero measured speed the filter holds its last TARGET, not wheel angle.
  // The actuator continues toward+.3rad, moving+.1rad this prediction step.
  EXPECT_NEAR(state.wz(0, 1), 0.05f * std::tan(0.1f) / 0.213f, 1e-6);
  for (float invalid : {0.6f, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity()})
  {
    EXPECT_THROW(model.setInitialConditionedSteering(invalid), nav2_core::NoValidControl);
    EXPECT_THROW(model.predict(state), nav2_core::NoValidControl);
  }
}

TEST(MotionModelTests, ConditionedAckermannRecordedLowSpeedTurnSign)
{
  // Run062401 at48764.908s: shared Python filter one-step replay matches
  // recorded output exactly. Prior target+.4232, raw request-.4883rad.
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("conditioned_ackermann");
  node->declare_parameter("test.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  ParametersHandler params(node);
  AckermannMotionModel model(&params, "test");
  models::ControlConstraints constraints{};
  constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
  model.initialize(constraints, 0.1);
  models::State state;
  state.reset(1, 2);
  const float velocity = -0.007825244222396109;
  const float previous = 0.42319609755391957;
  state.vx(0, 0) = velocity;
  state.wz(0, 0) = velocity * std::tan(previous) / 0.213f;
  state.cvx.fill(velocity);
  state.cwz.fill(velocity * std::tan(-0.48827317469942144f) / 0.213f);
  model.predict(state);
  EXPECT_LT(state.wz(0, 1), 0.0f);
  EXPECT_NEAR(state.wz(0, 1), -0.01405410640539719, 1e-6);
  EXPECT_EQ(state.vx(0, 0), velocity);
}

TEST(MotionModelTests, ConditionedAckermannStandstillNeedsExplicitSteering)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("standstill_steering");
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  ParametersHandler params(node);
  AckermannMotionModel model(&params, "test");
  models::ControlConstraints constraints{};
  constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
  model.initialize(constraints, 0.05);
  models::State state;
  state.reset(1, 2);
  state.cvx.fill(0.05f);state.cwz.fill(-0.1f);
  EXPECT_THROW(model.predict(state), nav2_core::NoValidControl);
  model.setInitialSteering(0.3f);
  state.reset(1, 2);
  state.cvx.fill(0.05f);state.cwz.fill(-0.1f);
  EXPECT_NO_THROW(model.predict(state));
  // No measured motion: hardware alpha guard holds the known steering angle.
  EXPECT_NEAR(state.wz(0, 1), 0.05f * std::tan(0.3f) / 0.213f, 1e-6);
  EXPECT_EQ(state.vx(0, 0), 0.0f);
  EXPECT_EQ(state.wz(0, 0), 0.0f);
  EXPECT_EQ(state.cwz(0, 0), -0.1f);
  model.setInitialSteering(std::nullopt);
  state.reset(1, 2);
  EXPECT_THROW(model.predict(state), nav2_core::NoValidControl);
}

TEST(MotionModelTests, ConditionedAckermannMirrorsVelocityWithoutChangingCommands)
{
  float yaw_rates[2];
  for (int k = 0; k < 2; ++k) {
    const float velocity = k == 0 ? -0.005f : 0.005f;
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("mirrored_steering");
    node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
    ParametersHandler params(node);
    AckermannMotionModel model(&params, "test");
    models::ControlConstraints constraints{};
    constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
    model.initialize(constraints, 0.1);
    model.setInitialSteering(0.4f);
    models::State state;
    state.reset(1, 2);
    state.vx(0, 0) = velocity;
    state.wz(0, 0) = velocity * std::tan(0.4f) / 0.213f;
    state.cvx.fill(velocity);
    state.cwz.fill(velocity * std::tan(-0.4f) / 0.213f);
    const auto controls = state.cwz;
    model.predict(state);
    EXPECT_EQ(state.cwz, controls);
    EXPECT_NEAR(state.wz(0, 1), velocity * std::tan(0.345f) / 0.213f, 1e-6);
    yaw_rates[k] = state.wz(0, 1);
  }
  EXPECT_NEAR(yaw_rates[0], -yaw_rates[1], 1e-7);
}

TEST(MotionModelTests, ConditionedAckermannRejectsInvalidFeedbackAndClearsOldHint)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("invalid_steering");
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  ParametersHandler params(node);
  AckermannMotionModel model(&params, "test");
  models::ControlConstraints constraints{};
  constraints.ax_max = 5;constraints.ax_min = -5;constraints.az_max = 100;
  model.initialize(constraints, 0.05);
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      0.577f, -0.577f})
  {
    model.setInitialSteering(0.3f);
    EXPECT_THROW(model.setInitialSteering(bad), nav2_core::NoValidControl);
    models::State state;
    state.reset(1, 2);
    EXPECT_THROW(model.predict(state), nav2_core::NoValidControl);
  }
}

TEST(MotionModelTests, DiffDriveTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  std::unique_ptr<DiffDriveMotionModel> model =
    std::make_unique<DiffDriveMotionModel>();

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * xt::ones<float>({batches, timesteps});
  state.cvy = 5 * xt::ones<float>({batches, timesteps});
  state.cwz = 1 * xt::ones<float>({batches, timesteps});

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  xt::view(state.vx, xt::all(), 0) = 10;
  xt::view(state.wz, xt::all(), 0) = 1;

  model->predict(state);

  EXPECT_EQ(state.vx, state.cvx);
  EXPECT_EQ(state.vy, xt::zeros<float>({batches, timesteps}));  // non-holonomic
  EXPECT_EQ(state.wz, state.cwz);

  // Check that application of constraints are empty for Diff Drive
  for (unsigned int i = 0; i != control_sequence.vx.shape(0); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.wz(i) = i * i * i;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  EXPECT_EQ(initial_control_sequence.vx, control_sequence.vx);
  EXPECT_EQ(initial_control_sequence.vy, control_sequence.vy);
  EXPECT_EQ(initial_control_sequence.wz, control_sequence.wz);

  // Check that Diff Drive is properly non-holonomic
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, OmniTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  std::unique_ptr<OmniMotionModel> model =
    std::make_unique<OmniMotionModel>();

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * xt::ones<float>({batches, timesteps});
  state.cvy = 5 * xt::ones<float>({batches, timesteps});
  state.cwz = 1 * xt::ones<float>({batches, timesteps});

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  xt::view(state.vx, xt::all(), 0) = 10;
  xt::view(state.vy, xt::all(), 0) = 5;
  xt::view(state.wz, xt::all(), 0) = 1;

  model->predict(state);

  EXPECT_EQ(state.vx, state.cvx);
  EXPECT_EQ(state.vy, state.cvy);  // holonomic
  EXPECT_EQ(state.wz, state.cwz);

  // Check that application of constraints are empty for Omni Drive
  for (unsigned int i = 0; i != control_sequence.vx.shape(0); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.vy(i) = i * i * i;
    control_sequence.wz(i) = i * i * i;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  EXPECT_EQ(initial_control_sequence.vx, control_sequence.vx);
  EXPECT_EQ(initial_control_sequence.vy, control_sequence.vy);
  EXPECT_EQ(initial_control_sequence.wz, control_sequence.wz);

  // Check that Omni Drive is properly holonomic
  EXPECT_EQ(model->isHolonomic(), true);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, AckermannTest)
{
  models::ControlSequence control_sequence;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  ParametersHandler param_handler(node);
  std::unique_ptr<AckermannMotionModel> model =
    std::make_unique<AckermannMotionModel>(&param_handler, node->get_name());

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * xt::ones<float>({batches, timesteps});
  state.cvy = 5 * xt::ones<float>({batches, timesteps});
  state.cwz = 1 * xt::ones<float>({batches, timesteps});

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  xt::view(state.vx, xt::all(), 0) = 10;
  xt::view(state.wz, xt::all(), 0) = 1;

  model->predict(state);

  EXPECT_EQ(state.vx, state.cvx);
  EXPECT_EQ(state.vy, xt::zeros<float>({batches, timesteps}));  // non-holonomic
  EXPECT_EQ(state.wz, state.cwz);

  // Check that application of constraints are non-empty for Ackermann Drive
  for (unsigned int i = 0; i != control_sequence.vx.shape(0); i++) {
    control_sequence.vx(i) = i * i * i;
    control_sequence.wz(i) = i * i * i * i;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_EQ(initial_control_sequence.vx, control_sequence.vx);
  EXPECT_NE(initial_control_sequence.wz, control_sequence.wz);
  for (unsigned int i = 1; i != control_sequence.wz.shape(0); i++) {
    EXPECT_GT(control_sequence.wz(i), 0.0);
  }

  // Now, check the specifics of the minimum curvature constraint
  EXPECT_NEAR(model->getMinTurningRadius(), 0.2, 1e-6);
  for (unsigned int i = 1; i != control_sequence.vx.shape(0); i++) {
    EXPECT_TRUE(fabs(control_sequence.vx(i)) / fabs(control_sequence.wz(i)) >= 0.2);
  }

  // Check that Ackermann Drive is properly non-holonomic and parameterized
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, AckermannReversingTest)
{
  models::ControlSequence control_sequence;
  models::ControlSequence control_sequence2;
  models::State state;
  int batches = 1000;
  int timesteps = 50;
  control_sequence.reset(timesteps);  // populates with zeros
  control_sequence2.reset(timesteps);  // populates with zeros
  state.reset(batches, timesteps);  // populates with zeros
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  ParametersHandler param_handler(node);
  std::unique_ptr<AckermannMotionModel> model =
    std::make_unique<AckermannMotionModel>(&param_handler, node->get_name());

  // Check that predict properly populates the trajectory velocities with the control velocities
  state.cvx = 10 * xt::ones<float>({batches, timesteps});
  state.cvy = 5 * xt::ones<float>({batches, timesteps});
  state.cwz = 1 * xt::ones<float>({batches, timesteps});

  // Manually set state index 0 from initial conditions which would be the speed of the robot
  xt::view(state.vx, xt::all(), 0) = 10;
  xt::view(state.wz, xt::all(), 0) = 1;

  model->predict(state);

  EXPECT_EQ(state.vx, state.cvx);
  EXPECT_EQ(state.vy, xt::zeros<float>({batches, timesteps}));  // non-holonomic
  EXPECT_EQ(state.wz, state.cwz);

  // Check that application of constraints are non-empty for Ackermann Drive
  for (unsigned int i = 0; i != control_sequence.vx.shape(0); i++) {
    float idx = static_cast<float>(i);
    control_sequence.vx(i) = -idx * idx * idx;  // now reversing
    control_sequence.wz(i) = idx * idx * idx * idx;
  }

  models::ControlSequence initial_control_sequence = control_sequence;
  model->applyConstraints(control_sequence);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_EQ(initial_control_sequence.vx, control_sequence.vx);
  EXPECT_NE(initial_control_sequence.wz, control_sequence.wz);
  for (unsigned int i = 1; i != control_sequence.wz.shape(0); i++) {
    EXPECT_GT(control_sequence.wz(i), 0.0);
  }

  // Repeat with negative rotation direction
  for (unsigned int i = 0; i != control_sequence2.vx.shape(0); i++) {
    float idx = static_cast<float>(i);
    control_sequence2.vx(i) = -idx * idx * idx;  // now reversing
    control_sequence2.wz(i) = -idx * idx * idx * idx;
  }

  models::ControlSequence initial_control_sequence2 = control_sequence2;
  model->applyConstraints(control_sequence2);
  // VX equal since this doesn't change, the WZ is reduced if breaking the constraint
  EXPECT_EQ(initial_control_sequence2.vx, control_sequence2.vx);
  EXPECT_NE(initial_control_sequence2.wz, control_sequence2.wz);
  for (unsigned int i = 1; i != control_sequence2.wz.shape(0); i++) {
    EXPECT_LT(control_sequence2.wz(i), 0.0);
  }

  // Now, check the specifics of the minimum curvature constraint
  EXPECT_NEAR(model->getMinTurningRadius(), 0.2, 1e-6);
  for (unsigned int i = 1; i != control_sequence2.vx.shape(0); i++) {
    EXPECT_TRUE(fabs(control_sequence2.vx(i)) / fabs(control_sequence2.wz(i)) >= 0.2);
  }

  // Check that Ackermann Drive is properly non-holonomic and parameterized
  EXPECT_EQ(model->isHolonomic(), false);

  // Check it cleanly destructs
  model.reset();
}

TEST(MotionModelTests, AckermannPredictEnforcesTurningRadiusOnSamples)
{
  // Stage-3 regression (notes/cusp_handoff_fix_plan.md): sampled trajectories
  // used to roll out with wz the vehicle cannot execute (|vx|/|wz| < r_min).
  // Scoring those pivot-fantasy rollouts, then clamping only the winning
  // sequence, produced the full-lock crawl (M3 start stalls, M6 mid-reverse
  // stall, reverse-start stalls on cuspless plans).
  models::State state;
  int batches = 100;
  int timesteps = 30;
  state.reset(batches, timesteps);
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  ParametersHandler param_handler(node);
  auto model = std::make_unique<AckermannMotionModel>(&param_handler, node->get_name());
  const float r_min = model->getMinTurningRadius();

  // Base predict applies acceleration limits from initialize(); without it
  // the deltas are zero and wz freezes at its initial row (vacuous test).
  models::ControlConstraints constraints{0.5f, -0.35f, 0.0f, 1.9f, 3.0f, -3.0f, 0.0f, 0.0f, 3.5f};
  model->initialize(constraints, 0.05f);

  // Infeasible samples: slow reverse creep with a hard turn demand —
  // exactly the pivot-in-place fantasy MPPI kept selecting
  state.cvx = -0.05f * xt::ones<float>({batches, timesteps});
  state.cwz = 1.0f * xt::ones<float>({batches, timesteps});

  model->predict(state);

  // Every rolled-out (vx, wz) pair must satisfy the vehicle's turning limit
  // once the acceleration chain has had time to reach the cone (t >= 3 here)
  bool nonvacuous = false;
  for (int b = 0; b != batches; b++) {
    for (int t = 3; t != timesteps; t++) {
      const float vx = state.vx(b, t);
      const float wz = state.wz(b, t);
      if (std::fabs(wz) > 1e-6f) {
        nonvacuous = true;
        EXPECT_GE(std::fabs(vx) / std::fabs(wz), r_min - 1e-4f) <<
          "sampled rollout violates min turning radius at (" << b << "," << t << ")";
      }
    }
  }
  EXPECT_TRUE(nonvacuous);
  // Direction of turn preserved
  EXPECT_GT(state.wz(0, 5), 0.0f);
  // Column 0 is the measured initial state — never edited (review amendment 3).
  // NOTE: write-back into cwz (review amendments 1-2) was implemented and
  // empirically falsified — see the predict() comment in motion_models.hpp.
  EXPECT_FLOAT_EQ(state.wz(0, 0), 0.0f);
}

TEST(MotionModelTests, CalibratedAckermannRadiusKeepsCommandsAndMeasuredState)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("radius_response");
  node->declare_parameter("radius_response.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("radius_response.AckermannConstraints.response_radius_offset", 0.094);
  ParametersHandler parameters(node);
  AckermannMotionModel model(&parameters, node->get_name());
  models::ControlConstraints limits{0.5f, -0.3f, 0.0f, 1.9f, 5.0f, -5.0f,
    0.0f, 0.0f, 3.5f};
  model.initialize(limits, 0.05f);
  for (float direction : {1.0f, -1.0f}) {
    for (float radius : {0.401f, 0.854f}) {
      models::State state;
      state.reset(1, 20);
      state.cvx = direction * 0.08f * xt::ones<float>({1, 20});
      state.cwz = direction * (0.08f / radius) * xt::ones<float>({1, 20});
      state.vx(0, 0) = direction * 0.08f;
      state.wz(0, 0) = direction * 0.123f;
      model.predict(state);
      EXPECT_FLOAT_EQ(state.wz(0, 0), direction * 0.123f);
      EXPECT_NEAR(state.wz(0, 10), direction * 0.08f / (radius + 0.094f), 1e-6f);
      EXPECT_NEAR(state.cwz(0, 10), direction * 0.08f / radius, 1e-6f);
      EXPECT_FLOAT_EQ(state.vx(0, 10), direction * 0.08f);
      EXPECT_NEAR(model.getMinTurningRadius(), 0.401f, 1e-6f);
    }
  }
  models::ControlSequence commands;
  commands.reset(5);
  commands.vx.fill(0.08f);
  commands.wz.fill(0.08f / 0.401f);
  model.applyConstraints(commands);
  EXPECT_NEAR(commands.wz(0), 0.08f / 0.401f, 1e-6f);
  models::State stopped;
  stopped.reset(1, 5);
  model.predict(stopped);
  EXPECT_FLOAT_EQ(stopped.wz(0, 4), 0.0f);
}
