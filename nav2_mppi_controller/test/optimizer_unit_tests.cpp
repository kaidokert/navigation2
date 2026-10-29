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
#include "nav2_mppi_controller/optimizer.hpp"

// Tests main optimizer functions

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

using namespace mppi;  // NOLINT
using namespace mppi::critics;  // NOLINT
using namespace mppi::utils;  // NOLINT
using xt::evaluation_strategy::immediate;

class OptimizerTester : public Optimizer
{
public:
  OptimizerTester()
  : Optimizer() {}

  int finalGuardFallbackBudget() const
  {
    return final_guard_fallback_candidates_;
  }

  void testSetDiffModel()
  {
    EXPECT_EQ(motion_model_.get(), nullptr);
    EXPECT_NO_THROW(setMotionModel("DiffDrive"));
    EXPECT_NE(motion_model_.get(), nullptr);
    EXPECT_TRUE(dynamic_cast<DiffDriveMotionModel *>(motion_model_.get()));
    EXPECT_FALSE(isHolonomic());
  }

  void testSetOmniModel()
  {
    EXPECT_EQ(motion_model_.get(), nullptr);
    EXPECT_NO_THROW(setMotionModel("Omni"));
    EXPECT_NE(motion_model_.get(), nullptr);
    EXPECT_TRUE(dynamic_cast<OmniMotionModel *>(motion_model_.get()));
    EXPECT_TRUE(isHolonomic());
  }

  void testSetAckModel()
  {
    EXPECT_EQ(motion_model_.get(), nullptr);
    EXPECT_NO_THROW(setMotionModel("Ackermann"));
    EXPECT_NE(motion_model_.get(), nullptr);
    EXPECT_TRUE(dynamic_cast<AckermannMotionModel *>(motion_model_.get()));
    EXPECT_FALSE(isHolonomic());
  }

  void testSetRandModel()
  {
    EXPECT_EQ(motion_model_.get(), nullptr);
    try {
      setMotionModel("Random");
      FAIL();
    } catch (...) {
      SUCCEED();
    }
    EXPECT_EQ(motion_model_.get(), nullptr);
  }

  void resetMotionModel()
  {
    motion_model_.reset();
  }

  void setOffsetWrapper(const double freq)
  {
    return setOffset(freq);
  }

  bool getShiftControlSequence()
  {
    return settings_.shift_control_sequence;
  }

  void fillOptimizerWithGarbage()
  {
    state_.vx = 0.43432 * xt::ones<float>({1000, 10});
    control_sequence_.vx = 342.0 * xt::ones<float>({30});
    control_history_[0] = {43, 5646, 32432};
    costs_ = 5.32 * xt::ones<float>({56453});
    generated_trajectories_.x = 432.234 * xt::ones<float>({7865, 1});
  }

  void testReset()
  {
    reset();

    EXPECT_EQ(state_.vx, xt::zeros<float>({1000, 50}));
    EXPECT_EQ(control_sequence_.vx, xt::zeros<float>({50}));
    EXPECT_EQ(control_history_[0].vx, 0.0);
    EXPECT_EQ(control_history_[0].vy, 0.0);
    EXPECT_NEAR(xt::sum(costs_, immediate)(), 0, 1e-6);
    EXPECT_EQ(generated_trajectories_.x, xt::zeros<float>({1000, 50}));
  }

  bool fallbackWrapper(bool fail)
  {
    return fallback(fail);
  }

  void testPrepare(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::Pose & goal,
    nav2_core::GoalChecker * goal_checker)
  {
    prepare(robot_pose, robot_speed, plan, goal, goal_checker);

    EXPECT_EQ(critics_data_.goal_checker, nullptr);
    EXPECT_NEAR(xt::sum(costs_, immediate)(), 0, 1e-6);  // should be reset
    EXPECT_FALSE(critics_data_.fail_flag);  // should be reset
    EXPECT_FALSE(critics_data_.motion_model->isHolonomic());  // object is valid + diff drive
    EXPECT_FALSE(critics_data_.furthest_reached_path_point.has_value());  // val is not set
    EXPECT_FALSE(critics_data_.path_pts_valid.has_value());  // val is not set
    EXPECT_EQ(state_.pose.pose.position.x, 999);
    EXPECT_EQ(state_.speed.linear.y, 4.0);
    EXPECT_EQ(path_.x.shape(0), 17u);
  }

  void shiftControlSequenceWrapper()
  {
    return shiftControlSequence();
  }

  std::pair<double, double> getVelLimits()
  {
    auto & s = settings_;
    return {s.constraints.vx_min, s.constraints.vx_max};
  }

  models::ControlConstraints & getControlConstraints()
  {
    return settings_.constraints;
  }

  void applyControlSequenceConstraintsWrapper()
  {
    return applyControlSequenceConstraints();
  }

  models::ControlSequence & grabControlSequence()
  {
    return control_sequence_;
  }

  void testupdateStateVels()
  {
    // updateInitialStateVelocities
    models::State state;
    state.reset(1000, 50);
    state.speed.linear.x = 5.0;
    state.speed.linear.y = 1.0;
    state.speed.angular.z = 6.0;
    state.cvx = 0.75 * xt::ones<float>({1000, 50});
    state.cvy = 0.5 * xt::ones<float>({1000, 50});
    state.cwz = 0.1 * xt::ones<float>({1000, 50});
    updateInitialStateVelocities(state);
    EXPECT_NEAR(state.vx(0, 0), 5.0, 1e-6);
    EXPECT_NEAR(state.vy(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(state.wz(0, 0), 6.0, 1e-6);

    // propagateStateVelocitiesFromInitials
    float max_delta_vx = settings_.constraints.ax_max * settings_.model_dt;
    float min_delta_vx = settings_.constraints.ax_min * settings_.model_dt;
    float max_delta_vy = settings_.constraints.ay_max * settings_.model_dt;
    float min_delta_vy = settings_.constraints.ay_min * settings_.model_dt;
    float max_delta_wz = settings_.constraints.az_max * settings_.model_dt;
    propagateStateVelocitiesFromInitials(state);
    EXPECT_NEAR(state.vx(0, 0), 5.0, 1e-6);
    EXPECT_NEAR(state.vy(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(state.wz(0, 0), 6.0, 1e-6);
    EXPECT_NEAR(state.vx(0, 1), std::clamp(0.75, 5.0 + min_delta_vx, 5.0 + max_delta_vx), 1e-6);
    EXPECT_NEAR(state.vy(0, 1), std::clamp(0.5, 1.0 + min_delta_vy, 1.0 + max_delta_vy), 1e-6);
    EXPECT_NEAR(state.wz(0, 1), std::clamp(0.1, 6.0 - max_delta_wz, 6.0 + max_delta_wz), 1e-6);

    // Putting them together: updateStateVelocities
    state.reset(1000, 50);
    state.speed.linear.x = -5.0;
    state.speed.linear.y = -1.0;
    state.speed.angular.z = -6.0;
    state.cvx = -0.75 * xt::ones<float>({1000, 50});
    state.cvy = -0.5 * xt::ones<float>({1000, 50});
    state.cwz = -0.1 * xt::ones<float>({1000, 50});
    updateStateVelocities(state);
    EXPECT_NEAR(state.vx(0, 0), -5.0, 1e-6);
    EXPECT_NEAR(state.vy(0, 0), -1.0, 1e-6);
    EXPECT_NEAR(state.wz(0, 0), -6.0, 1e-6);
    EXPECT_NEAR(state.vx(0, 1), std::clamp(-0.75, -5.0 + min_delta_vx, -5.0 + max_delta_vx), 1e-6);
    EXPECT_NEAR(state.vy(0, 1), std::clamp(-0.5, -1.0 - max_delta_vy, -1.0 + max_delta_vy), 1e-6);
    EXPECT_NEAR(state.wz(0, 1), std::clamp(-0.1, -6.0 - max_delta_wz, -6.0 + max_delta_wz), 1e-6);
  }

  geometry_msgs::msg::TwistStamped getControlFromSequenceAsTwistWrapper()
  {
    builtin_interfaces::msg::Time stamp;
    return getControlFromSequenceAsTwist(stamp);
  }

  void integrateStateVelocitiesWrapper(
    models::Trajectories & traj,
    const models::State & state)
  {
    return integrateStateVelocities(traj, state);
  }

  void expectFinalBankCostsMatchFreshEvaluation()
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.pose.orientation.w = 1.0;
    nav_msgs::msg::Path path;
    path.header = pose.header;
    path.poses = {pose, pose};
    path.poses.back().pose.position.x = 1.0;
    prepare(pose, geometry_msgs::msg::Twist{}, path, path.poses.back().pose, nullptr);
    optimize();
    const auto final_costs = costs_;
    // Independently score precisely the final bank, without another noise draw
    // or control update. Gamma is zero, so this must be the complete objective.
    costs_.fill(0.0f);
    critic_manager_.evalTrajectoriesScores(critics_data_);
    ASSERT_GT(xt::amin(costs_)(), 1.0f);  // Ensure the real critic actually ran.
    EXPECT_LT(xt::amax(xt::abs(final_costs - costs_))(), 1e-5f);
  }
};

TEST(OptimizerTests, EachIterationScoresItsCurrentBankWithoutEarlierCosts)
{
  // One iteration is the unchanged control; three exercises repeated recentering.
  for (const int iterations : {1, 3}) {
    SCOPED_TRACE(iterations);
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("bank_cost_node");
    node->declare_parameter("controller_frequency", rclcpp::ParameterValue(20.0));
    node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(32));
    node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(10));
    node->declare_parameter("mppic.iteration_count", rclcpp::ParameterValue(iterations));
    node->declare_parameter("mppic.gamma", rclcpp::ParameterValue(0.0));
    node->declare_parameter("mppic.regenerate_noises", rclcpp::ParameterValue(false));
    node->declare_parameter(
      "mppic.critics", rclcpp::ParameterValue(std::vector<std::string>{"GoalCritic"}));
    node->declare_parameter("mppic.GoalCritic.threshold_to_consider", rclcpp::ParameterValue(2.0));
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "bank_costmap", "", "bank_costmap", true);
    rclcpp_lifecycle::State lifecycle_state;
    ASSERT_EQ(costmap->on_configure(lifecycle_state), nav2_util::CallbackReturn::SUCCESS);
    costmap->pause();
    ParametersHandler parameters(node);
    OptimizerTester optimizer;
    optimizer.initialize(node, "mppic", costmap, &parameters);
    optimizer.expectFinalBankCostsMatchFreshEvaluation();
    optimizer.shutdown();
    costmap->on_cleanup(lifecycle_state);
  }
}

TEST(OptimizerTests, FinalGuardFallbackBudgetRejectsMutationAndPreservesReadback)
{
  for (const bool predeclared : {false, true}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("fallback_parameter_test");
    node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
    const int initial = predeclared ? 8 : 0;
    const std::string name = "mppic.final_guard_fallback_candidates";
    if (predeclared) {
      // Exercise the callback fallback when no read-only descriptor exists.
      node->declare_parameter(name, rclcpp::ParameterValue(initial));
    }
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "dummy_costmap", "", "dummy_costmap", true);
    ParametersHandler parameters(node);
    rclcpp_lifecycle::State state;
    costmap->on_configure(state);
    OptimizerTester optimizer;
    optimizer.initialize(node, "mppic", costmap, &parameters);
    parameters.start();

    EXPECT_EQ(optimizer.finalGuardFallbackBudget(), initial);
    const auto result = node->set_parameter(rclcpp::Parameter(name, predeclared ? 0 : 8));
    EXPECT_FALSE(result.successful);
    EXPECT_EQ(node->get_parameter(name).as_int(), initial);
    EXPECT_EQ(optimizer.finalGuardFallbackBudget(), initial);
    optimizer.shutdown();
  }
}

TEST(OptimizerTests, BasicInitializedFunctions)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.ax_min", rclcpp::ParameterValue(3.0));
  node->declare_parameter("mppic.ay_min", rclcpp::ParameterValue(3.0));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test value of ax_min, ay_min it should be negative
  auto & constraints = optimizer_tester.getControlConstraints();
  EXPECT_EQ(constraints.ax_min, -3.0);
  EXPECT_EQ(constraints.ay_min, -3.0);

  // Should be empty of size batches x time steps
  // and tests getting set params: time_steps, batch_size, controller_frequency
  auto trajs = optimizer_tester.getGeneratedTrajectories();
  EXPECT_EQ(trajs.x.shape(0), 1000u);
  EXPECT_EQ(trajs.x.shape(1), 50u);
  EXPECT_EQ(trajs.x, xt::zeros<float>({1000, 50}));

  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();
  auto traj = optimizer_tester.getOptimizedTrajectory();
  EXPECT_EQ(traj(5, 0), 0.0);  // x
  EXPECT_EQ(traj(5, 1), 0.0);  // y
  EXPECT_EQ(traj(5, 2), 0.0);  // yaw
  EXPECT_EQ(traj.shape(0), 50u);
  EXPECT_EQ(traj.shape(1), 3u);

  optimizer_tester.reset();
  optimizer_tester.shutdown();
}

TEST(OptimizerTests, TestOptimizerMotionModels)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Diff Drive should be non-holonomic
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetDiffModel();

  // Omni Drive should be holonomic
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();

  // // Ackermann should be non-holonomic
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetAckModel();

  // // Rand should fail
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetRandModel();
}

TEST(OptimizerTests, setOffsetTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("mppic.model_dt", rclcpp::ParameterValue(0.1));
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test offsets are properly set based on relationship of model_dt and controller frequency
  // Also tests getting set model_dt parameter.
  EXPECT_THROW(optimizer_tester.setOffsetWrapper(1.0), std::runtime_error);
  EXPECT_NO_THROW(optimizer_tester.setOffsetWrapper(30.0));
  EXPECT_FALSE(optimizer_tester.getShiftControlSequence());
  EXPECT_NO_THROW(optimizer_tester.setOffsetWrapper(10.0));
  EXPECT_TRUE(optimizer_tester.getShiftControlSequence());
}

TEST(OptimizerTests, resetTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Tests resetting the full state of all the functions after filling with garbage
  optimizer_tester.fillOptimizerWithGarbage();
  optimizer_tester.testReset();
}

TEST(OptimizerTests, FallbackTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.retry_attempt_limit", rclcpp::ParameterValue(2));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test fallback logic, also tests getting set param retry_attempt_limit
  // Because retry set to 2, it should attempt soft resets 2x before throwing exception
  // for hard reset
  EXPECT_FALSE(optimizer_tester.fallbackWrapper(false));
  EXPECT_TRUE(optimizer_tester.fallbackWrapper(true));
  EXPECT_TRUE(optimizer_tester.fallbackWrapper(true));
  EXPECT_THROW(optimizer_tester.fallbackWrapper(true), std::runtime_error);
}

TEST(OptimizerTests, PrepareTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.retry_attempt_limit", rclcpp::ParameterValue(2));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test Prepare function to set the state of the robot pose/speed on new cycle
  // Populate the contents with things easily identifiable if correct
  geometry_msgs::msg::PoseStamped pose;
  pose.pose.position.x = 999;
  geometry_msgs::msg::Twist speed;
  speed.linear.y = 4.0;
  nav_msgs::msg::Path path;
  geometry_msgs::msg::Pose goal;
  path.poses.resize(17);

  optimizer_tester.testPrepare(pose, speed, path, goal, nullptr);
}

TEST(OptimizerTests, shiftControlSequenceTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.retry_attempt_limit", rclcpp::ParameterValue(2));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test shiftControlSequence by setting the 2nd value to something unique to neighbors
  auto & sequence = optimizer_tester.grabControlSequence();
  sequence.reset(100);
  sequence.vx(0) = 9999;
  sequence.vx(1) = 6;
  sequence.vx(2) = 888;
  sequence.vy(0) = 9999;
  sequence.vy(1) = 6;
  sequence.vy(2) = 888;
  sequence.wz(0) = 9999;
  sequence.wz(1) = 6;
  sequence.wz(2) = 888;

  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();
  optimizer_tester.shiftControlSequenceWrapper();

  EXPECT_EQ(sequence.vx(0), 6);
  EXPECT_EQ(sequence.vy(0), 6);
  EXPECT_EQ(sequence.wz(0), 6);
  EXPECT_EQ(sequence.vx(1), 888);
  EXPECT_EQ(sequence.vy(1), 888);
  EXPECT_EQ(sequence.wz(1), 888);
  EXPECT_EQ(sequence.vx(2), 0);
  EXPECT_EQ(sequence.vy(2), 0);
  EXPECT_EQ(sequence.wz(2), 0);
}

TEST(OptimizerTests, SpeedLimitTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.retry_attempt_limit", rclcpp::ParameterValue(2));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test Speed limits API
  auto [v_min, v_max] = optimizer_tester.getVelLimits();
  EXPECT_EQ(v_max, 0.5f);
  EXPECT_EQ(v_min, -0.35f);
  optimizer_tester.setSpeedLimit(0, false);
  auto [v_min2, v_max2] = optimizer_tester.getVelLimits();
  EXPECT_EQ(v_max2, 0.5f);
  EXPECT_EQ(v_min2, -0.35f);
  optimizer_tester.setSpeedLimit(50.0, true);
  auto [v_min3, v_max3] = optimizer_tester.getVelLimits();
  EXPECT_NEAR(v_max3, 0.5 / 2.0, 1e-3);
  EXPECT_NEAR(v_min3, -0.35 / 2.0, 1e-3);
  optimizer_tester.setSpeedLimit(0, true);
  auto [v_min4, v_max4] = optimizer_tester.getVelLimits();
  EXPECT_EQ(v_max4, 0.5f);
  EXPECT_EQ(v_min4, -0.35f);
  optimizer_tester.setSpeedLimit(0.75, false);
  auto [v_min5, v_max5] = optimizer_tester.getVelLimits();
  EXPECT_NEAR(v_max5, 0.75, 1e-3);
  EXPECT_NEAR(v_min5, -0.5249, 1e-2);
}

TEST(OptimizerTests, applyControlSequenceConstraintsTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.vx_max", rclcpp::ParameterValue(1.0));
  node->declare_parameter("mppic.vx_min", rclcpp::ParameterValue(-1.0));
  node->declare_parameter("mppic.vy_max", rclcpp::ParameterValue(0.75));
  node->declare_parameter("mppic.wz_max", rclcpp::ParameterValue(2.0));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test constraints being applied to ensure feasibility of trajectories
  // Also tests param get of set vx/vy/wz min/maxes

  // Set model to omni to consider holonomic vy elements
  // Ack is not tested here because `applyConstraints` is covered in detail
  // in motion_models_test.cpp
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();
  auto & sequence = optimizer_tester.grabControlSequence();

  // Test boundary of limits
  sequence.vx = xt::ones<float>({50});
  sequence.vy = 0.75 * xt::ones<float>({50});
  sequence.wz = 2.0 * xt::ones<float>({50});
  optimizer_tester.applyControlSequenceConstraintsWrapper();
  EXPECT_EQ(sequence.vx, xt::ones<float>({50}));
  EXPECT_EQ(sequence.vy, 0.75 * xt::ones<float>({50}));
  EXPECT_EQ(sequence.wz, 2.0 * xt::ones<float>({50}));

  // Test breaking limits sets to maximum
  sequence.vx = 5.0 * xt::ones<float>({50});
  sequence.vy = 5.0 * xt::ones<float>({50});
  sequence.wz = 5.0 * xt::ones<float>({50});
  optimizer_tester.applyControlSequenceConstraintsWrapper();
  EXPECT_EQ(sequence.vx, xt::ones<float>({50}));
  EXPECT_EQ(sequence.vy, 0.75 * xt::ones<float>({50}));
  EXPECT_EQ(sequence.wz, 2.0 * xt::ones<float>({50}));

  // Test breaking limits sets to minimum
  sequence.vx = -5.0 * xt::ones<float>({50});
  sequence.vy = -5.0 * xt::ones<float>({50});
  sequence.wz = -5.0 * xt::ones<float>({50});
  optimizer_tester.applyControlSequenceConstraintsWrapper();
  EXPECT_EQ(sequence.vx, -1.0 * xt::ones<float>({50}));
  EXPECT_EQ(sequence.vy, -0.75 * xt::ones<float>({50}));
  EXPECT_EQ(sequence.wz, -2.0 * xt::ones<float>({50}));
}

TEST(OptimizerTests, updateStateVelocitiesTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.vx_max", rclcpp::ParameterValue(1.0));
  node->declare_parameter("mppic.vx_min", rclcpp::ParameterValue(-1.0));
  node->declare_parameter("mppic.vy_max", rclcpp::ParameterValue(0.60));
  node->declare_parameter("mppic.wz_max", rclcpp::ParameterValue(2.0));
  node->declare_parameter("mppic.ax_max", rclcpp::ParameterValue(3.0));
  node->declare_parameter("mppic.ax_min", rclcpp::ParameterValue(-3.0));
  node->declare_parameter("mppic.ay_max", rclcpp::ParameterValue(3.0));
  node->declare_parameter("mppic.ay_min", rclcpp::ParameterValue(-3.0));
  node->declare_parameter("mppic.az_max", rclcpp::ParameterValue(3.5));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test settings of the state to the initial robot speed to start rollout
  // Set model to omni to consider holonomic vy elements
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();
  optimizer_tester.testupdateStateVels();
}

TEST(OptimizerTests, getControlFromSequenceAsTwistTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  node->declare_parameter("mppic.vx_max", rclcpp::ParameterValue(1.0));
  node->declare_parameter("mppic.vx_min", rclcpp::ParameterValue(-1.0));
  node->declare_parameter("mppic.vy_max", rclcpp::ParameterValue(0.60));
  node->declare_parameter("mppic.wz_max", rclcpp::ParameterValue(2.0));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  // Test conversion of control sequence into a Twist command to execute
  auto & sequence = optimizer_tester.grabControlSequence();
  sequence.vx = 0.25 * xt::ones<float>({10});
  sequence.vy = 0.5 * xt::ones<float>({10});
  sequence.wz = 0.1 * xt::ones<float>({10});

  auto diff_t = optimizer_tester.getControlFromSequenceAsTwistWrapper();
  EXPECT_NEAR(diff_t.twist.linear.x, 0.25, 1e-6);
  EXPECT_NEAR(diff_t.twist.linear.y, 0.0, 1e-6);  // Y should not be populated
  EXPECT_NEAR(diff_t.twist.angular.z, 0.1, 1e-6);

  // Set model to omni to consider holonomic vy elements
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();
  auto omni_t = optimizer_tester.getControlFromSequenceAsTwistWrapper();
  EXPECT_NEAR(omni_t.twist.linear.x, 0.25, 1e-6);
  EXPECT_NEAR(omni_t.twist.linear.y, 0.5, 1e-6);  // Now it should be
  EXPECT_NEAR(omni_t.twist.angular.z, 0.1, 1e-6);
}

TEST(OptimizerTests, integrateStateVelocitiesTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(30.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(1000));
  node->declare_parameter("mppic.model_dt", rclcpp::ParameterValue(0.1));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(50));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);
  optimizer_tester.resetMotionModel();
  optimizer_tester.testSetOmniModel();

  // Test integration of velocities for trajectory rollout poses

  // Give it a couple of easy const traj and check rollout, start from 0
  models::State state;
  state.reset(1000, 50);
  models::Trajectories traj;
  state.vx = 0.1 * xt::ones<float>({1000, 50});
  xt::view(state.vx, xt::all(), 0) = xt::zeros<float>({1000});
  state.vy = xt::zeros<float>({1000, 50});
  state.wz = xt::zeros<float>({1000, 50});

  optimizer_tester.integrateStateVelocitiesWrapper(traj, state);
  EXPECT_EQ(traj.y, xt::zeros<float>({1000, 50}));
  EXPECT_EQ(traj.yaws, xt::zeros<float>({1000, 50}));
  for (unsigned int i = 0; i != traj.x.shape(1); i++) {
    EXPECT_NEAR(traj.x(1, i), i * 0.1 /*vel*/ * 0.1 /*dt*/, 1e-3);
  }

  // Give it a bit of a more complex trajectory to crunch
  state.vy = 0.2 * xt::ones<float>({1000, 50});
  xt::view(state.vy, xt::all(), 0) = xt::zeros<float>({1000});
  optimizer_tester.integrateStateVelocitiesWrapper(traj, state);

  EXPECT_EQ(traj.yaws, xt::zeros<float>({1000, 50}));
  for (unsigned int i = 0; i != traj.x.shape(1); i++) {
    EXPECT_NEAR(traj.x(1, i), i * 0.1 /*vel*/ * 0.1 /*dt*/, 1e-3);
    EXPECT_NEAR(traj.y(1, i), i * 0.2 /*vel*/ * 0.1 /*dt*/, 1e-3);
  }

  // Lets add some angular motion to the mix
  state.vy = xt::zeros<float>({1000, 50});
  state.wz = 0.2 * xt::ones<float>({1000, 50});
  xt::view(state.wz, xt::all(), 0) = xt::zeros<float>({1000});
  optimizer_tester.integrateStateVelocitiesWrapper(traj, state);

  float x = 0;
  float y = 0;
  for (unsigned int i = 1; i != traj.x.shape(1); i++) {
    std::cout << i << std::endl;
    x += (0.1 /*vx*/ * cosf(0.2 /*wz*/ * 0.1 /*model_dt*/ * (i - 1))) * 0.1 /*model_dt*/;
    y += (0.1 /*vx*/ * sinf(0.2 /*wz*/ * 0.1 /*model_dt*/ * (i - 1))) * 0.1 /*model_dt*/;

    EXPECT_NEAR(traj.x(1, i), x, 1e-6);
    EXPECT_NEAR(traj.y(1, i), y, 1e-6);
  }
}

namespace
{

class BatchOutcomeCritic : public mppi::critics::CriticFunction
{
public:
  BatchOutcomeCritic() {enabled_ = true;}
  void initialize() override {}
  void score(mppi::CriticData & data) override
  {
    ++calls;
    data.costs.fill(blocked ? 1000.0f : 7.0f);
    if (blocked) {
      data.fail_flag = true;
    }
    // A feasible critic does not clear a failure owned by the retry lifecycle.
  }
  bool blocked{false};
  int calls{0};
};

class BatchManager : public mppi::CriticManager
{
public:
  BatchManager()
  {
    auto critic = std::make_unique<BatchOutcomeCritic>();
    outcome = critic.get();
    critics_.push_back(std::move(critic));
    critic_names_.push_back("BatchOutcomeCritic");
  }
  BatchOutcomeCritic * outcome;
};

class RetryObserver : public mppi::Optimizer
{
public:
  explicit RetryObserver(mppi::ParametersHandler * parameters)
  {
    settings_.batch_size = 2;
    settings_.time_steps = 3;
    settings_.model_dt = 0.05f;
    settings_.retry_attempt_limit = 1;
    settings_.sampling_std.vx = 0.1f;
    settings_.sampling_std.wz = 0.1f;
    motion_model_ = std::make_shared<mppi::DiffDriveMotionModel>();
    noise_generator_.initialize(settings_, false, "FollowPath", parameters);
    reset();
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "map";
    pose.pose.orientation.w = 1.0;
    nav_msgs::msg::Path path;
    path.header = pose.header;
    path.poses = {pose, pose};
    path.poses.back().pose.position.x = 1.0;
    geometry_msgs::msg::Twist speed;
    prepare(pose, speed, path, path.poses.back().pose, nullptr);
  }
  mppi::CriticData & data() {return critics_data_;}
  bool retry() {return fallback(critics_data_.fail_flag);}
};

}  // namespace

TEST(OptimizerTests, FallbackRetryClearsFailFlagAndScoresRetryBatch)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("retry_lifecycle_node");
  ParametersHandler parameters(node);
  RetryObserver optimizer(&parameters);
  RetryObserver peer_optimizer(&parameters);

  BatchManager manager;
  auto & data = optimizer.data();
  auto & peer_data = peer_optimizer.data();

  // Case 1: Transient blocked batch followed by feasible retry batch must score twice,
  // and must not consume a peer optimizer instance's retry budget.
  manager.outcome->blocked = true;
  manager.evalTrajectoriesScores(data);
  ASSERT_EQ(manager.outcome->calls, 1);
  ASSERT_TRUE(data.fail_flag);
  ASSERT_TRUE(optimizer.retry());
  EXPECT_FALSE(data.fail_flag);
  EXPECT_FLOAT_EQ(data.costs(0), 0.0f);
  manager.evalTrajectoriesScores(peer_data);
  ASSERT_TRUE(peer_data.fail_flag);
  EXPECT_TRUE(peer_optimizer.retry());
  EXPECT_FALSE(peer_data.fail_flag);
  manager.outcome->blocked = false;
  manager.evalTrajectoriesScores(data);
  EXPECT_EQ(manager.outcome->calls, 3);
  EXPECT_FALSE(data.fail_flag);
  EXPECT_FLOAT_EQ(data.costs(0), 7.0f);
  EXPECT_FALSE(optimizer.retry());

  // Case 2: Persistent blocked batch still exhausts retry_attempt_limit and throws NoValidControl.
  manager.outcome->blocked = true;
  manager.evalTrajectoriesScores(data);
  ASSERT_EQ(manager.outcome->calls, 4);
  ASSERT_TRUE(data.fail_flag);
  ASSERT_TRUE(optimizer.retry());
  manager.evalTrajectoriesScores(data);
  EXPECT_EQ(manager.outcome->calls, 5);
  EXPECT_TRUE(data.fail_flag);
  EXPECT_THROW(optimizer.retry(), nav2_core::NoValidControl);
}

TEST(OptimizerTests, EvalControlRecoversViaFallbackRetryWhenWarmStartCollides)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("eval_control_retry_node");
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(10.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(4));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(10));
  node->declare_parameter("mppic.model_dt", rclcpp::ParameterValue(0.1));
  node->declare_parameter("mppic.retry_attempt_limit", rclcpp::ParameterValue(1));
  node->declare_parameter("mppic.gamma", rclcpp::ParameterValue(0.0));
  node->declare_parameter("mppic.vx_std", rclcpp::ParameterValue(0.001));
  node->declare_parameter("mppic.vy_std", rclcpp::ParameterValue(0.001));
  node->declare_parameter("mppic.wz_std", rclcpp::ParameterValue(0.001));
  node->declare_parameter(
    "mppic.critics",
    rclcpp::ParameterValue(std::vector<std::string>{"ObstaclesCritic"}));
  node->declare_parameter("mppic.ObstaclesCritic.consider_footprint", rclcpp::ParameterValue(true));

  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "retry_costmap", "", "retry_costmap", true);
  costmap_ros->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  costmap_ros->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[0.170,0.151],[0.170,-0.151],[-0.156,-0.151],[-0.156,0.151]]")));
  rclcpp_lifecycle::State lstate;
  ASSERT_EQ(costmap_ros->on_configure(lstate), nav2_util::CallbackReturn::SUCCESS);
  costmap_ros->pause();
  costmap_ros->getLayeredCostmap()->resizeMap(200, 200, 0.05, -2.0, -2.0);
  auto * grid = costmap_ros->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 40000, 0);

  // Place a lethal obstacle at x = 0.23 m so a warm-started 0.5 m/s rollout collides on pass 1,
  // while the fallback-reset 0.0 m/s rollout at the origin (0, 0) is collision-free on pass 2.
  unsigned int mx = 0u, my = 0u;
  ASSERT_TRUE(grid->worldToMap(0.23, 0.0, mx, my));
  grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);

  ParametersHandler param_handler(node);
  OptimizerTester optimizer_tester;
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);
  optimizer_tester.fallbackWrapper(false);

  auto & seq = optimizer_tester.grabControlSequence();
  seq.vx.fill(0.5f);
  seq.wz.fill(0.0f);

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist speed;
  nav_msgs::msg::Path plan;
  plan.header = pose.header;
  plan.poses = {pose, pose};
  plan.poses.back().pose.position.x = 1.0;

  geometry_msgs::msg::TwistStamped cmd;
  EXPECT_NO_THROW({
    cmd = optimizer_tester.evalControl(pose, speed, plan, plan.poses.back().pose, nullptr);
  });
  EXPECT_NEAR(cmd.twist.linear.x, 0.0, 1e-2);

  optimizer_tester.shutdown();
  costmap_ros->on_cleanup(lstate);
}
