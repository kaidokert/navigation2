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

#include <gtest/gtest.h>
#include <limits>
#include <nav2_costmap_2d/footprint_collision_checker.hpp>
#include "nav2_mppi_controller/optimizer.hpp"
#include "settled_costmap.hpp"
#include "handoff_prefix_fixture.hpp"
#include "early_cusp_fixture.hpp"

namespace
{
class EarlyCuspOptimizer : public mppi::Optimizer
{
public:
  void load()
  {
    const auto & d = early_cusp_fixture::data;
    setRealPathPointCount(d["real_path_point_count"]);
    setActiveCuspDirection(d["active_cusp_direction"]);
    setPathDirection(d["path_direction"]);
    setCuspTolerances(std::make_pair(d["effective_cusp_tolerances"][0].get<double>(),
      d["effective_cusp_tolerances"][1].get<double>()));
    setInitialSteering(d["initial_steering"].get<float>());
    setInitialConditionedSteering(d["initial_conditioned_steering"].get<float>());
    geometry_msgs::msg::PoseStamped robot;robot.header.frame_id = d["frame"];
    robot.header.stamp = rclcpp::Time(int64_t(d["stamp"].get<double>() * 1e9));
    const auto fill = [](auto & p, const auto & a) {
        p.position.x = a[0];p.position.y = a[1];
        p.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
          a[2].template get<double>()));
      };
    fill(robot.pose, d["robot"]);
    geometry_msgs::msg::Twist speed;speed.linear.x = d["speed"][0];speed.angular.z = d["speed"][1];
    geometry_msgs::msg::Pose goal;fill(goal, d["goal"]);
    nav_msgs::msg::Path path;path.header = robot.header;
    for(size_t k = 0; k < d["path_x"]["data"].size(); ++k) {
      auto p = robot;p.pose.position.x = d["path_x"]["data"][k];
      p.pose.position.y = d["path_y"]["data"][k];
      p.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
          d["path_yaw"]["data"][k].get<double>()));
      path.poses.push_back(p);
    }
    prepare(robot, speed, path, goal, nullptr);
    for(size_t k = 0; k < control_sequence_.vx.size(); ++k) {
      control_sequence_.vx(k) = d["control_vx"]["data"][k];
      control_sequence_.wz(k) = d["control_wz"]["data"][k];
    }
    for(size_t k = 0; k < control_history_.size(); ++k) {
      control_history_[k].vx = d["history"][k][0];control_history_[k].wz = d["history"][k][1];
    }
  }
  bool initializeCapturedFamily() {return initializeShortCuspMean();}
  auto emit()
  {
    smoothControlSequence();
    return getControlFromSequenceAsTwist(state_.pose.header.stamp);
  }
  void installMeanPassEmissionFail()
  {
    float remaining = 0.05f;
    for(size_t j = 0; j < control_sequence_.vx.size(); ++j) {
      const float travel = std::min(0.08f, remaining / settings_.model_dt);
      control_sequence_.vx(j) = std::max(0.0f, travel);
      control_sequence_.wz(j) = -0.5f * control_sequence_.vx(j);
      remaining = std::max(0.0f, remaining - control_sequence_.vx(j) * settings_.model_dt);
    }
    applyControlSequenceConstraints();
  }
  void installRetainedStraight()
  {
    state_.pose.pose.position.x = 0;state_.pose.pose.position.y = 0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), 0));
    state_.speed.linear.x = 0.05;state_.speed.angular.z = 0;
    setInitialSteering(0);setInitialConditionedSteering(0);
    setCuspTolerances(std::make_pair(0.03, 0.1));
    path_.x.fill(0.2f);path_.y.fill(0);path_.yaws.fill(0);
    control_sequence_.vx.fill(0.05f);control_sequence_.wz.fill(0);
    mppi::models::ControlSequence tail;tail.reset(81);tail.vx.fill(0.05f);tail.wz.fill(0);
    cusp_completion_ = tail;cusp_completion_attempted_ = true;cusp_completion_shifts_ = 7;
    gentle_short_cusp_required_ = true;
  }
  void wrongFirstGear()
  {
    control_sequence_.vx(1) = -0.001f;control_sequence_.wz(1) = 0;cusp_completion_->vx(1) = -0.001f;
    cusp_completion_->wz(1) = 0;
  }
  void measuredOppositeGear() {state_.speed.linear.x = -0.001;state_.speed.angular.z = 0;}
  bool retainedFeasible() const {return actualCuspStreamFeasible(*cusp_completion_);}
  bool prefixCollisionFree() const {return controlSequenceCollisionFree();}
  void finalGuard() {enforceGentleCuspEmission();}
  void checkFallbackRetention()
  {
    const auto before = *cusp_completion_;const auto shifts = cusp_completion_shifts_;
    fallback(false);ASSERT_TRUE(fallback(true));
    EXPECT_TRUE(gentle_short_cusp_required_);EXPECT_TRUE(cusp_completion_attempted_);
    ASSERT_TRUE(cusp_completion_);EXPECT_EQ(cusp_completion_shifts_, shifts);
    for(size_t j = 0; j < before.vx.size(); ++j) {
      EXPECT_EQ(cusp_completion_->vx(j), before.vx(j));
      EXPECT_EQ(cusp_completion_->wz(j), before.wz(j));
    }
    fallback(false);
  }
  void corruptTail(float value) {cusp_completion_->vx(70) = value;}
  void alignWithInvalidFeedback()
  {
    state_.pose.pose.position.x = 0.2;
    // Public feedback ingestion must reject NaN and clear the old value.
    // The aligned endpoint must not then retire the delivery obligation.
    EXPECT_THROW(setInitialSteering(std::numeric_limits<float>::quiet_NaN()),
        nav2_core::NoValidControl);
  }
  bool meanFeasible() const {return cuspSequenceFeasible(control_sequence_);}
  bool gentleRequired() const {return gentle_short_cusp_required_;}
  auto emitUnfiltered() {return getControlFromSequenceAsTwist(state_.pose.header.stamp);}
  bool emittedTailFeasible() const
  {
    // Independent extraction oracle: next command is index1, not index0.
    auto stream = control_sequence_;
    for(size_t k = 0; k + 1 < stream.vx.size(); ++k) {
      stream.vx(k) = control_sequence_.vx(k + 1);stream.wz(k) = control_sequence_.wz(k + 1);
    }
    stream.vx(stream.vx.size() - 1) = 0;stream.wz(stream.wz.size() - 1) = 0;
    return cuspSequenceFeasible(stream);
  }
};
}  // namespace

TEST(OptimizerTests, RecordedEarlyCuspFindsGentleExecutableContinuationAndRejectsBlockedGrid)
{
  const auto & d = early_cusp_fixture::data;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("early_cusp_family");
  handoff_prefix_fixture::declareParameters(node);
  node->set_parameter(rclcpp::Parameter("mppi.critics", std::vector<std::string>{}));
  auto cm = std::make_shared<SettledTestCostmap>("early_cusp_map", "", "early_cusp_map", false);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame", std::string("odom")));
  cm->set_parameter(rclcpp::Parameter("footprint", d["footprint"].dump()));
  cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid = cm->getCostmap();
  grid->resizeMap(d["grid"]["width"], d["grid"]["height"], d["grid"]["resolution"],
    d["grid"]["origin"][0], d["grid"]["origin"][1]);
  const auto cells = d["grid"]["data"].get<std::vector<unsigned char>>();
  ASSERT_EQ(cells.size(), grid->getSizeInCellsX() * grid->getSizeInCellsY());
  std::copy(cells.begin(), cells.end(), grid->getCharMap());
  mppi::ParametersHandler parameters(node);EarlyCuspOptimizer optimizer;
  optimizer.initialize(node, "mppi", cm, &parameters);optimizer.load();
  EXPECT_TRUE(optimizer.initializeCapturedFamily());
  EXPECT_NO_THROW({
    const auto command = optimizer.emit();
    EXPECT_GT(command.twist.linear.x, 0.0);
    EXPECT_TRUE(optimizer.emittedTailFeasible());
  });
  ASSERT_TRUE(optimizer.gentleRequired());
  // A constrained mean can pass while the actual index-1 stream misses the
  // unchanged endpoint gate. Exercise final extraction, not a helper alone.
  optimizer.installMeanPassEmissionFail();
  ASSERT_TRUE(optimizer.meanFeasible());
  ASSERT_FALSE(optimizer.emittedTailFeasible());
  try {
    const auto repaired = optimizer.emitUnfiltered();
    EXPECT_GT(repaired.twist.linear.x, 0.0);
    EXPECT_TRUE(optimizer.emittedTailFeasible());
  } catch(const nav2_core::FailedToMakeProgress &) {
    // Failing closed is acceptable; returning the invalid stream is not.
  }
  // Same recorded state and exact endpoint gates; only the grid becomes blocked.
  optimizer.reset();optimizer.load();
  std::fill(grid->getCharMap(), grid->getCharMap() + cells.size(),
    nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_FALSE(optimizer.initializeCapturedFamily());
  EXPECT_THROW(optimizer.emit(), nav2_core::FailedToMakeProgress);
  // Synthetic retained-horizon control: its endpoint requires all 81 samples.
  // Every first-41 footprint is clear; a cell reached only by the late tail
  // must still reject. This is guard coverage, separate from recorded repair.
  grid->resizeMap(100, 100, 0.02, -1, -1);
    std::fill(grid->getCharMap(), grid->getCharMap() + 10000, nav2_costmap_2d::FREE_SPACE);
  optimizer.reset();optimizer.load();optimizer.installRetainedStraight();
  ASSERT_TRUE(optimizer.retainedFeasible());
  optimizer.measuredOppositeGear();
  EXPECT_TRUE(optimizer.retainedFeasible());
  EXPECT_NO_THROW(optimizer.finalGuard());
  optimizer.installRetainedStraight();optimizer.wrongFirstGear();
  EXPECT_FALSE(optimizer.retainedFeasible());
  EXPECT_THROW(optimizer.finalGuard(), nav2_core::FailedToMakeProgress);
  optimizer.installRetainedStraight();
  optimizer.checkFallbackRetention();
  optimizer.installRetainedStraight();
  unsigned int mx, my;ASSERT_TRUE(grid->worldToMap(0.44, 0.14, mx, my));
  grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
  ASSERT_TRUE(optimizer.prefixCollisionFree());
  ASSERT_FALSE(optimizer.retainedFeasible());
  EXPECT_THROW(optimizer.finalGuard(), nav2_core::FailedToMakeProgress);
  std::fill(grid->getCharMap(), grid->getCharMap() + 10000, nav2_costmap_2d::FREE_SPACE);
  for(float value : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity()})
  {
    optimizer.installRetainedStraight();optimizer.corruptTail(value);
    EXPECT_FALSE(optimizer.retainedFeasible());
    EXPECT_THROW(optimizer.finalGuard(), nav2_core::FailedToMakeProgress);
  }
  optimizer.installRetainedStraight();optimizer.alignWithInvalidFeedback();
  EXPECT_THROW(optimizer.finalGuard(), nav2_core::FailedToMakeProgress);
  EXPECT_TRUE(optimizer.gentleRequired());
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}
