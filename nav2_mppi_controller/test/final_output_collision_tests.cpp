// Copyright (c) 2026 kaidokert
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
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nav2_core/controller_exceptions.hpp"
#include "nav2_mppi_controller/optimizer.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{
class FinalOutputOptimizer : public mppi::Optimizer
{
public:
  void configure(
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm, bool shift = false,
    unsigned int steps = 25)
  {
    costmap_ros_ = cm;
    costmap_ = cm->getCostmap();
    settings_.time_steps = steps;
    settings_.iteration_count = 1;
    settings_.model_dt = 0.1f;
    settings_.shift_control_sequence = shift;
    auto & c = settings_.constraints;
    c.vx_max = 0.5f;
    c.vx_min = -0.5f;
    c.vy = 0.5f;
    c.wz = 2.0f;
    c.ax_max = c.ay_max = c.az_max = 10.0f;
    c.ax_min = c.ay_min = -10.0f;
    motion_model_ = std::make_shared<mppi::DiffDriveMotionModel>();
    syncModel();
    state_.reset(1, steps);
    state_.pose.pose.orientation.w = 1.0;
    control_sequence_.reset(steps);
    control_history_.fill({0.0f, 0.0f, 0.0f});
  }
  void syncModel() {motion_model_->initialize(settings_.constraints, settings_.model_dt);}
  void sequence(float vx, float wz = 0.0f)
  {
    control_sequence_.vx.fill(vx);
    control_sequence_.wz.fill(wz);
  }
  bool clear() {return controlSequenceCollisionFree();}
  auto emit() {return getValidatedControlFromSequenceAsTwist(builtin_interfaces::msg::Time{});}
  auto filteredEmit() {return filterAndValidateControl(builtin_interfaces::msg::Time{});}
  void bank(const std::vector<mppi::models::Control> & controls, const std::vector<float> & costs)
  {
    ASSERT_EQ(controls.size(), costs.size());
    state_.reset(controls.size(), settings_.time_steps);
    costs_ = xt::zeros<float>({controls.size()});
    for (size_t row = 0; row < controls.size(); ++row) {
      costs_(row) = costs[row];
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(row, j) = controls[row].vx;
        state_.cvy(row, j) = controls[row].vy;
        state_.cwz(row, j) = controls[row].wz;
      }
    }
  }
  void smooth() {mppi::utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);}
  using Optimizer::final_guard_fallback_candidates_;
  using Optimizer::control_sequence_;
  using Optimizer::control_history_;
  using Optimizer::motion_model_;
  using Optimizer::settings_;
  using Optimizer::state_;
};

class FinalOutputTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "final_output_map", "", "final_output_map", true);
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    cm->set_parameter(rclcpp::Parameter("footprint", std::string(
        "[[0.170,0.151],[0.170,-0.151],[-0.156,-0.151],[-0.156,0.151]]")));
    cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.012));
    ASSERT_EQ(cm->on_configure(rclcpp_lifecycle::State{}), nav2_util::CallbackReturn::SUCCESS);
    cm->pause();
  }
  void SetUp() override
  {
    cm->getLayeredCostmap()->resizeMap(200, 200, 0.02, -2.0, -2.0);
    auto * grid = cm->getCostmap();
    std::fill(grid->getCharMap(), grid->getCharMap() + 40000, 0);
    opt.configure(cm);
  }
  static void TearDownTestSuite()
  {
    cm->on_cleanup(rclcpp_lifecycle::State{});
    cm.reset();
  }
  void obstacle(double x, double y, unsigned char cost = nav2_costmap_2d::LETHAL_OBSTACLE)
  {
    unsigned int mx, my;
    ASSERT_TRUE(cm->getCostmap()->worldToMap(x, y, mx, my));
    cm->getCostmap()->setCost(mx, my, cost);
  }
  inline static std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm;
  FinalOutputOptimizer opt;
};

TEST_F(FinalOutputTest, ClearForwardAndReverseAreAccepted)
{
  for (float speed : {0.3f, -0.3f}) {
    opt.sequence(speed);
    EXPECT_FLOAT_EQ(opt.emit().twist.linear.x, speed);
    EXPECT_FLOAT_EQ(opt.control_history_.back().vx, speed);
  }
}

TEST_F(FinalOutputTest, ClearCurvesCanAverageIntoCollidingStraight)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.8, 0.0);
  opt.sequence(0.3f, 0.6f);
  EXPECT_TRUE(opt.clear());
  opt.sequence(0.3f, -0.6f);
  EXPECT_TRUE(opt.clear());
  opt.sequence(0.3f);  // Equal weighted mean of the two clear controls.
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, FinalGuardFallbackIsDisabledByDefault)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.8, 0.0);
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  opt.bank({{0.3f, 0.0f, 0.6f}, {0.3f, 0.0f, -0.6f}}, {1.0f, 2.0f});
  opt.sequence(0.3f);
  EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  EXPECT_FLOAT_EQ(opt.control_history_.back().vx, 0.0f);
}

TEST_F(FinalOutputTest, FinalGuardFallbackDoesNotUseBankWithoutOptimization)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.8, 0.0);
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  opt.final_guard_fallback_candidates_ = 2;
  opt.settings_.iteration_count = 0;
  opt.bank({{0.3f, 0.0f, 0.6f}}, {1.0f});
  opt.sequence(0.3f);
  EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
  EXPECT_FLOAT_EQ(opt.control_history_.back().vx, 0.0f);
}

TEST_F(FinalOutputTest, FinalGuardFallbackIssuesLowestCostValidatedCurve)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.8, 0.0);
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  opt.final_guard_fallback_candidates_ = 2;
  opt.bank({{0.3f, 0.0f, -0.6f}, {0.3f, 0.0f, 0.6f}}, {2.0f, 1.0f});
  opt.sequence(0.3f);
  const auto control = opt.filteredEmit();
  EXPECT_GT(control.twist.linear.x, 0.0);
  EXPECT_GT(control.twist.angular.z, 0.0);
  EXPECT_TRUE(opt.clear());
  EXPECT_FLOAT_EQ(opt.control_history_.back().wz, control.twist.angular.z);
}

TEST_F(FinalOutputTest, FinalGuardFallbackAllBlockedStillStops)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.5, 0.0);
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  opt.final_guard_fallback_candidates_ = 2;
  opt.bank({{0.3f, 0.0f, 0.0f}, {0.4f, 0.0f, 0.0f}}, {1.0f, 2.0f});
  opt.sequence(0.3f);
  EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  for (const auto & control : opt.control_history_) {
    EXPECT_FLOAT_EQ(control.vx, 0.0f);
    EXPECT_FLOAT_EQ(control.wz, 0.0f);
  }
}

TEST_F(FinalOutputTest, FinalGuardFallbackNeverMasksInvalidMeanOrState)
{
  for (bool invalid_state : {false, true}) {
    opt.configure(cm);
    opt.final_guard_fallback_candidates_ = 2;
    opt.bank({{0.3f, 0.0f, 0.6f}}, {1.0f});
    opt.sequence(0.3f);
    if (invalid_state) {
      opt.state_.speed.angular.z = std::numeric_limits<double>::infinity();
    } else {
      opt.control_sequence_.vx(0) = std::numeric_limits<float>::quiet_NaN();
    }
    EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
    EXPECT_FLOAT_EQ(opt.control_history_.back().vx, 0.0f);
  }
}

TEST_F(FinalOutputTest, FinalGuardFallbackRespectsAttemptBudget)
{
  opt.state_.speed.linear.x = 0.3;
  obstacle(0.8, 0.0);
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  opt.final_guard_fallback_candidates_ = 1;
  opt.bank({{0.3f, 0.0f, 0.0f}, {0.3f, 0.0f, 0.6f}}, {1.0f, 2.0f});
  opt.sequence(0.3f);
  EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, FinalGuardFallbackTrialsShareOriginalHistoryAndIssuedOffset)
{
  obstacle(0.8, 0.0);
  for (bool shift : {false, true}) {
    opt.configure(cm, shift);
    opt.state_.speed.linear.x = 0.3;
    opt.control_history_ = {{{0.20f, 0.0f, -0.03f}, {0.23f, 0.0f, 0.01f},
      {0.26f, 0.0f, -0.02f}, {0.29f, 0.0f, 0.03f}}};
    FinalOutputOptimizer expected;
    expected.configure(cm, shift);
    expected.state_.speed = opt.state_.speed;
    expected.control_history_ = opt.control_history_;
    expected.sequence(0.3f, 0.6f);
    const auto reference = expected.filteredEmit();
    opt.final_guard_fallback_candidates_ = 2;
    // First finite row still collides after filtering. The second must be
    // filtered against the original history, not the mean or rejected row.
    opt.bank({{0.3f, 0.0f, 0.0f}, {0.3f, 0.0f, 0.6f}}, {1.0f, 2.0f});
    // Predicted velocities intentionally disagree: fallback must use controls.
    opt.state_.vx.fill(-0.4f);
    opt.state_.wz.fill(-1.0f);
    opt.sequence(0.3f);
    const auto actual = opt.filteredEmit();
    EXPECT_FLOAT_EQ(actual.twist.linear.x, reference.twist.linear.x);
    EXPECT_FLOAT_EQ(actual.twist.angular.z, reference.twist.angular.z);
    for (size_t i = 0; i < opt.control_history_.size(); ++i) {
      EXPECT_FLOAT_EQ(opt.control_history_[i].vx, expected.control_history_[i].vx);
      EXPECT_FLOAT_EQ(opt.control_history_[i].vy, expected.control_history_[i].vy);
      EXPECT_FLOAT_EQ(opt.control_history_[i].wz, expected.control_history_[i].wz);
    }
    for (size_t j = 0; j < opt.settings_.time_steps; ++j) {
      EXPECT_FLOAT_EQ(opt.control_sequence_.vx(j), expected.control_sequence_.vx(j));
      EXPECT_FLOAT_EQ(opt.control_sequence_.wz(j), expected.control_sequence_.wz(j));
    }
    const size_t issued = shift ? 1 : 0;
    EXPECT_FLOAT_EQ(actual.twist.angular.z, opt.control_sequence_.wz(issued));
  }
}

TEST_F(FinalOutputTest, FinalGuardFallbackRechecksAckermannConeAndSkipsNonfiniteRows)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("fallback_ackermann");
  node->declare_parameter("test.AckermannConstraints.min_turning_r", 0.561);
  mppi::ParametersHandler handler(node);
  opt.motion_model_ = std::make_shared<mppi::AckermannMotionModel>(&handler, "test");
  opt.syncModel();
  opt.state_.speed.linear.x = 0.3;
  opt.control_history_.fill({0.3f, 0.0f, 0.0f});
  obstacle(0.8, 0.0);
  opt.final_guard_fallback_candidates_ = 1;
  opt.bank({{0.3f, 0.0f, std::numeric_limits<float>::quiet_NaN()},
      {0.3f, 0.0f, 0.0f}, {0.3f, 0.0f, 1.2f}},
    {0.0f, std::numeric_limits<float>::infinity(), 1.0f});
  opt.sequence(0.3f);
  const auto control = opt.filteredEmit();
  EXPECT_GT(control.twist.angular.z, 0.0);
  EXPECT_LE(std::abs(control.twist.angular.z), std::abs(control.twist.linear.x) / 0.561 + 1e-6);
  EXPECT_TRUE(opt.clear());
}

TEST_F(FinalOutputTest, SmoothingCanMakeSafeStoppedSequenceCollide)
{
  obstacle(0.21, 0.0);
  opt.sequence(0.0f);
  ASSERT_TRUE(opt.clear());
  opt.control_history_.fill({0.5f, 0.0f, 0.0f});
  opt.smooth();
  ASSERT_GT(opt.control_sequence_.vx(0), 0.1f);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, FinalGuardFallbackRejectsRowMadeUnsafeBySmoothing)
{
  obstacle(0.21, 0.0);
  FinalOutputOptimizer candidate;
  candidate.configure(cm);
  candidate.sequence(0.0f);
  ASSERT_TRUE(candidate.clear());  // The sampled row itself is collision-free.
  candidate.control_history_.fill({0.5f, 0.0f, 0.0f});
  candidate.smooth();
  ASSERT_GT(candidate.control_sequence_.vx(0), 0.1f);
  ASSERT_FALSE(candidate.clear());  // Issuing it after SG would not be safe.

  opt.final_guard_fallback_candidates_ = 1;
  opt.control_history_.fill({0.5f, 0.0f, 0.0f});
  opt.bank({{0.0f, 0.0f, 0.0f}}, {1.0f});
  opt.sequence(0.3f);  // A blocked mean forces the fallback attempt.
  ASSERT_FALSE(opt.clear());
  EXPECT_THROW(opt.filteredEmit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  for (const auto & control : opt.control_history_) {
    EXPECT_FLOAT_EQ(control.vx, 0.0f);
    EXPECT_FLOAT_EQ(control.wz, 0.0f);
  }
}

TEST_F(FinalOutputTest, ValidationStartsWithActualShiftedCommand)
{
  obstacle(0.22, 0.0);
  opt.configure(cm, false, 2);
  opt.control_sequence_.vx(0) = -0.5f;
  opt.control_sequence_.vx(1) = 0.5f;
  EXPECT_NO_THROW(opt.emit());
  opt.settings_.shift_control_sequence = true;
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, ForecastIncludesLastControlSlot)
{
  opt.configure(cm, false, 2);
  obstacle(0.22, 0.0);
  opt.control_sequence_.vx(1) = 0.5f;
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, MeasuredVelocityContinuesWhileBraking)
{
  obstacle(0.23, 0.0);
  opt.state_.speed.linear.x = 0.3;
  opt.settings_.constraints.ax_min = -0.3f;
  opt.syncModel();
  opt.sequence(0.0f);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, CornerCollisionWithFreeCenterVetoesAndClearsHistory)
{
  obstacle(0.10, 0.16);
  opt.control_history_.fill({0.1f, 0.0f, 0.1f});
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  EXPECT_EQ(opt.control_history_.back().vx, 0.0f);
}

TEST_F(FinalOutputTest, LethalCenterCannotBeMissedByPerimeter)
{
  obstacle(0.0, 0.0);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, UnknownAndMapBoundaryReject)
{
  obstacle(0.0, 0.0, nav2_costmap_2d::NO_INFORMATION);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  obstacle(0.0, 0.0, nav2_costmap_2d::FREE_SPACE);
  opt.state_.pose.pose.position.x = 1.9;
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, NonfiniteControlsRejectUnderProductionFastMath)
{
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()})
  {
    opt.sequence(value);
    EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  }
}

TEST_F(FinalOutputTest, NonfiniteStateAndInvalidQuaternionReject)
{
  opt.state_.speed.angular.z = std::numeric_limits<double>::infinity();
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  opt.state_.speed.angular.z = 0.0;
  opt.state_.pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  opt.state_.pose.pose.position.x = 0.0;
  opt.state_.pose.pose.orientation.w = 0.0;
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, RecordedTableLegCornerIsRejected)
{
  cm->getLayeredCostmap()->resizeMap(
    60, 60, 0.05000000074505806, -1.1000000003725297, -1.4500000003725289);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 3600, 0);
  grid->setCost(30, 42, nav2_costmap_2d::LETHAL_OBSTACLE);
  opt.state_.pose.pose.position.x = 0.5028052926;
  opt.state_.pose.pose.position.y = 0.4304492474;
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, -2.0596189499);
  opt.state_.pose.pose.orientation = tf2::toMsg(q);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, HolonomicLateralMotionIsChecked)
{
  opt.motion_model_ = std::make_shared<mppi::OmniMotionModel>();
  opt.syncModel();
  opt.control_sequence_.vy.fill(0.2f);
  EXPECT_NO_THROW(opt.emit());
  obstacle(0.0, 0.5);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, MalformedSequenceRejectsBeforeConstraintIndexing)
{
  opt.control_sequence_.vy = xt::zeros<float>({0});
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(FinalOutputTest, AckermannConeIsAppliedAfterSmoothing)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("final_ackermann");
  node->declare_parameter("test.AckermannConstraints.min_turning_r", 0.5);
  mppi::ParametersHandler handler(node);
  opt.motion_model_ = std::make_shared<mppi::AckermannMotionModel>(&handler, "test");
  opt.syncModel();
  opt.sequence(0.05f, 1.0f);
  const auto control = opt.emit();
  EXPECT_LE(std::abs(control.twist.angular.z), 0.100001);
  EXPECT_FLOAT_EQ(opt.control_history_.back().wz, control.twist.angular.z);
}
class RadiusOutputTest : public FinalOutputTest
{
protected:
  static void SetUpTestSuite()
  {
    cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "final_radius_map", "", "final_radius_map", true);
    // Deliberately no inflation layer: geometry must provide the hard check.
    cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
    cm->set_parameter(rclcpp::Parameter("robot_radius", 0.2));
    cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.0));
    ASSERT_EQ(cm->on_configure(rclcpp_lifecycle::State{}), nav2_util::CallbackReturn::SUCCESS);
    cm->pause();
    ASSERT_TRUE(cm->getUseRadius());
  }
};

TEST_F(RadiusOutputTest, ClearBodyIsAcceptedWithoutInflation)
{
  opt.sequence(0.1f);
  EXPECT_NO_THROW(opt.emit());
}

TEST_F(RadiusOutputTest, InteriorObstacleAndUnknownAreRejectedWithFreeCenter)
{
  obstacle(0.1, 0.05);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  obstacle(0.1, 0.05, nav2_costmap_2d::NO_INFORMATION);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(RadiusOutputTest, BodyOutsideMapRejectsEvenWithCenterInside)
{
  opt.state_.pose.pose.position.x = 1.85;
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
}

TEST_F(RadiusOutputTest, OutsideCircleCornerCellDoesNotBlock)
{
  obstacle(0.19, 0.19);
  EXPECT_NO_THROW(opt.emit());
}
}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
