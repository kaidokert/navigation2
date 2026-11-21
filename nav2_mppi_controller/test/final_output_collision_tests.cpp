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
#include <array>
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
  void smooth() {mppi::utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);}
  // Sampled rollouts of the last iteration: one constant (vx, wz) row each, with its total cost.
  void rollouts(const std::vector<std::array<float, 3>> & rows)
  {
    state_.reset(rows.size(), settings_.time_steps);
    costs_ = xt::zeros<float>({rows.size()});
    for (size_t i = 0; i < rows.size(); ++i) {
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(i, j) = rows[i][0];
        state_.cwz(i, j) = rows[i][1];
      }
      costs_(i) = rows[i][2];
    }
  }
  using Optimizer::control_sequence_;
  using Optimizer::control_history_;
  using Optimizer::costs_;
  using Optimizer::fallback_rank_;
  using Optimizer::final_check_fallback_rollouts_;
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

// Final-check fallback (final_check_fallback_rollouts). Scene: the straight mean of two clear
// curves collides with an obstacle ahead (ClearCurvesCanAverageIntoCollidingStraight).
class FallbackTest : public FinalOutputTest
{
protected:
  void SetUp() override
  {
    FinalOutputTest::SetUp();
    opt.state_.speed.linear.x = 0.3;
    obstacle(0.8, 0.0);
    // cost order: straight (collides) 0.0 < right curve 1.0 < left curve 2.0
    opt.rollouts({{{0.3f, 0.0f, 0.0f}}, {{0.3f, 0.6f, 2.0f}}, {{0.3f, -0.6f, 1.0f}}});
    opt.sequence(0.3f);  // the mean: straight, collides
  }
};

TEST_F(FallbackTest, OffByDefaultKeepsTheRejection)
{
  EXPECT_EQ(FinalOutputOptimizer{}.final_check_fallback_rollouts_, 0);
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  EXPECT_EQ(opt.fallback_rank_, -1);
}

TEST_F(FallbackTest, EmitsTheCheapestRolloutThatPassesTheFinalCheck)
{
  opt.final_check_fallback_rollouts_ = 3;
  const auto cmd = opt.emit();
  // the cheapest (straight) collides; the next by cost is the right curve
  EXPECT_FLOAT_EQ(cmd.twist.linear.x, 0.3f);
  EXPECT_FLOAT_EQ(cmd.twist.angular.z, -0.6f);
  EXPECT_EQ(opt.fallback_rank_, 1);
  EXPECT_FLOAT_EQ(opt.control_history_.back().wz, -0.6f);
  // the emitted sequence is the one kept for the next cycle, and it passes the check itself
  EXPECT_FLOAT_EQ(opt.control_sequence_.wz(5), -0.6f);
  EXPECT_TRUE(opt.clear());
}

TEST_F(FallbackTest, TriesOnlyTheConfiguredNumberOfRollouts)
{
  opt.final_check_fallback_rollouts_ = 1;  // only the colliding straight one
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
  EXPECT_EQ(opt.control_history_.back().vx, 0.0f);
  EXPECT_EQ(opt.fallback_rank_, -1);
}

TEST_F(FallbackTest, RejectsWhenNoRolloutPasses)
{
  opt.final_check_fallback_rollouts_ = 3;
  opt.rollouts({{{0.3f, 0.0f, 0.0f}}, {{0.3f, 0.01f, 1.0f}}, {{0.3f, -0.01f, 2.0f}}});
  EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
  EXPECT_EQ(xt::sum(xt::abs(opt.control_sequence_.vx))(), 0.0f);
}

TEST_F(FallbackTest, NonFiniteCostsAreNeverTried)
{
  opt.final_check_fallback_rollouts_ = 3;
  // the only rollout that would pass has a NaN (or infinite) cost: nothing is emitted
  for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    opt.sequence(0.3f);
    opt.rollouts({{{0.3f, 0.0f, 0.0f}}, {{0.3f, 0.01f, 1.0f}}, {{0.3f, -0.6f, bad}}});
    EXPECT_THROW(opt.emit(), nav2_core::NoValidControl);
    EXPECT_EQ(opt.fallback_rank_, -1);
  }
  // with a finite cost the same rollout is emitted
  opt.sequence(0.3f);
  opt.rollouts({{{0.3f, 0.0f, 0.0f}}, {{0.3f, 0.01f, 1.0f}}, {{0.3f, -0.6f, 5.0f}}});
  EXPECT_FLOAT_EQ(opt.emit().twist.angular.z, -0.6f);
  EXPECT_EQ(opt.fallback_rank_, 2);
}

TEST_F(FallbackTest, TheSameConstraintPassAppliesToTheRollout)
{
  opt.final_check_fallback_rollouts_ = 3;
  // the right curve is sampled above vx_max: it is clipped like the mean would be
  opt.rollouts({{{0.3f, 0.0f, 0.0f}}, {{0.9f, -0.6f, 1.0f}}});
  const auto cmd = opt.emit();
  EXPECT_FLOAT_EQ(cmd.twist.linear.x, opt.settings_.constraints.vx_max);
  EXPECT_FLOAT_EQ(cmd.twist.angular.z, -0.6f);
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
