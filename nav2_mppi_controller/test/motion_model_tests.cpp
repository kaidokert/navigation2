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
