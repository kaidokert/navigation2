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
#include <cstring>
#include <cstdint>
#include <thread>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <set>
#include "settled_costmap.hpp"

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_mppi_controller/optimizer.hpp"
#include "opposite_feedback_gear_fixture.hpp"
#include "cusp_mean_feasibility_fixture.hpp"
#include "mixed_gear_mean_fixture.hpp"
#include "warm_start_delay_fixture.hpp"
#include "hidden_future_gear_fixture.hpp"
#include "long_prefix_gear_fixture.hpp"
#include "long_entry_fixture.hpp"
#include "handoff_prefix_fixture.hpp"
#include "retained_departure_fixture.hpp"
#include "blocked_forward_entry_fixture.hpp"
#include "handoff_horizon_fixture.hpp"
#include "entry_shift_fixture.hpp"
#include "entry_steering_fixture.hpp"
#include "sampled_failure_warm_fixture.hpp"
#include "effective_gate_fixture.hpp"
#include "exit_completion_fixture.hpp"
#include "nav2_mppi_controller/tools/diagnostic_capture.hpp"

TEST(OptimizerDiagnostics, TargetedFailedBatchCaptureSurvivesEarlierFailure)
{
  using mppi::diagnostic::failedBatchCaptureEligible;
  using mppi::diagnostic::minimumCaptureStamp;
  // Actual first capture and later terminal-filter state from run054108.
  // The one-shot must remain available when the earlier passage fails.
  const auto target = minimumCaptureStamp("69860.0");
  EXPECT_FALSE(failedBatchCaptureEligible(69782.897, 0.0, 0.031939, false, target));
  EXPECT_TRUE(failedBatchCaptureEligible(69873.55, 0.020859, -0.009461, false, target));
  EXPECT_FALSE(failedBatchCaptureEligible(69873.55, 0.020859, -0.009461, true, target));
  EXPECT_TRUE(failedBatchCaptureEligible(69873.55, 0.0, 0.0, true, target));
  EXPECT_TRUE(failedBatchCaptureEligible(69860.0, 0.0, 0.0, true, target));
  EXPECT_TRUE(failedBatchCaptureEligible(69782.897, 0.0, 0.031939, false,
    minimumCaptureStamp(nullptr)));
  for (const char * invalid : {"", "nan", "inf", "-1", "69860oops"}) {
    EXPECT_FALSE(failedBatchCaptureEligible(69873.55, 0.0, 0.0, false,
      minimumCaptureStamp(invalid)));
  }
}

TEST(OptimizerDiagnostics, RecentFailedBatchCaptureIsBoundedAndDoesNotConsumeFirstOnlyPolicy)
{
  mppi::diagnostic::FailedBatchRing ring;
  EXPECT_FALSE(ring.eligible(false, 74075.577));
  EXPECT_FALSE(ring.eligible(true, std::numeric_limits<double>::quiet_NaN()));
  for (unsigned int i = 0; i < 25; ++i) {
    const double stamp = 74075.577 + i;
    ASSERT_TRUE(ring.eligible(true, stamp));
    EXPECT_EQ(ring.take(stamp), i % 8);
    EXPECT_FALSE(ring.eligible(true, stamp));
    EXPECT_FALSE(ring.eligible(true, stamp + 0.999));
  }
  EXPECT_TRUE(ring.eligible(true, 1.0));
  EXPECT_EQ(ring.take(1.0), 1u);
}

TEST(OptimizerDiagnostics, SuccessfulBatchTargetDoesNotSelectEarlierOrShortSegments)
{
  using mppi::diagnostic::successfulBatchCaptureEligible;
  const std::optional<double> start{75685.0};
  EXPECT_FALSE(successfulBatchCaptureEligible(75684.9, 34, 1, start));
  EXPECT_FALSE(successfulBatchCaptureEligible(75692.05, 5, 1, start));
  EXPECT_FALSE(successfulBatchCaptureEligible(75692.05, 34, -1, start));
  EXPECT_FALSE(successfulBatchCaptureEligible(75692.05, 34, 1, std::nullopt));
  EXPECT_FALSE(successfulBatchCaptureEligible(
      std::numeric_limits<double>::quiet_NaN(), 34, 1, start));
  EXPECT_TRUE(successfulBatchCaptureEligible(75692.05, 34, 1, start));
}

TEST(OptimizerDiagnostics, MissingDiagnosticTimestampSerializesAsNullUnderFastMath)
{
  const auto missing = mppi::diagnostic::minimumCaptureStamp(nullptr);
  ASSERT_TRUE(missing);
  EXPECT_FALSE(mppi::diagnostic::isFiniteCaptureValue(*missing));
  EXPECT_FALSE(mppi::diagnostic::isFiniteCaptureValue(
      std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(mppi::diagnostic::isFiniteCaptureValue(
      std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(mppi::diagnostic::isFiniteCaptureValue(132.0));
}

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
  void testRecordedBlockedForwardEntry()
  {
    namespace f = blocked_forward_entry_fixture;
    ASSERT_EQ(f::robot.size(),3u);ASSERT_EQ(f::speed.size(),2u);
    ASSERT_EQ(f::steering.size(),2u);
    ASSERT_EQ(f::cvx.size(),41u);ASSERT_EQ(f::cwz.size(),41u);
    ASSERT_EQ(f::tx.size(),41u);ASSERT_EQ(f::ty.size(),41u);ASSERT_EQ(f::yaw.size(),41u);
    ASSERT_EQ(settings_.time_steps,41u);ASSERT_FLOAT_EQ(settings_.model_dt,0.05f);
    const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    ASSERT_TRUE(model);ASSERT_FLOAT_EQ(model->getMinTurningRadius(),0.401f);
    ASSERT_TRUE(model->predictsSteeringConditioning());
    state_.pose.header.frame_id = "odom";
    state_.pose.pose.position.x=f::robot[0];state_.pose.pose.position.y=f::robot[1];
    state_.pose.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::robot[2]));
    state_.speed.linear.x=f::speed[0];state_.speed.angular.z=f::speed[1];
    setInitialSteering(f::steering[0]);setInitialConditionedSteering(f::steering[1]);
    control_sequence_.vx.fill(0);control_sequence_.wz.fill(0);
    const auto hold=control_sequence_;
    ASSERT_TRUE(controlSequenceCollisionFree());  // Hold on the unchanged real raster.
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,hold.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,hold.wz)));
    std::copy(f::cvx.begin(),f::cvx.end(),control_sequence_.vx.begin());
    std::copy(f::cwz.begin(),f::cwz.end(),control_sequence_.wz.begin());
    models::State predicted;predicted.reset(1,41);predicted.pose=state_.pose;predicted.speed=state_.speed;
    for(size_t j=0;j<41;++j) {predicted.cvx(0,j)=f::cvx[j];predicted.cwz(0,j)=f::cwz[j];}
    updateStateVelocities(predicted);models::Trajectories trajectory;trajectory.reset(1,41);
    integrateStateVelocities(trajectory,predicted);
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
    bool footprint_hit=false;
    for(size_t j=0;j<41;++j) {
      EXPECT_NEAR(trajectory.x(0,j),f::tx[j],1e-6);
      EXPECT_NEAR(trajectory.y(0,j),f::ty[j],1e-6);
      EXPECT_NEAR(trajectory.yaws(0,j),f::yaw[j],1e-6);
      unsigned int mx,my;
      ASSERT_TRUE(costmap_->worldToMap(trajectory.x(0,j),trajectory.y(0,j),mx,my));
      EXPECT_LT(costmap_->getCost(mx,my),nav2_costmap_2d::LETHAL_OBSTACLE);
      const double cost=checker.footprintCostAtPose(trajectory.x(0,j),trajectory.y(0,j),
        trajectory.yaws(0,j),costmap_ros_->getRobotFootprint());
      EXPECT_GE(cost,0);
      if(cost>=nav2_costmap_2d::LETHAL_OBSTACLE) {footprint_hit=true;}
    }
    ASSERT_TRUE(footprint_hit);  // A center-only collision check misses this actual corner.
    const auto before=control_sequence_;
    EXPECT_FALSE(controlSequenceCollisionFree());
    EXPECT_EQ(control_sequence_.vx,before.vx);EXPECT_EQ(control_sequence_.wz,before.wz);
  }


  void testRecordedPostHandoffHorizon(bool delivery = false)
  {
    namespace f = handoff_horizon_fixture;
    // This is the captured POST-handoff mean state, not the preceding guard call.
    ASSERT_EQ(f::robot.size(),3u);ASSERT_EQ(f::speed.size(),2u);
    ASSERT_EQ(f::steering.size(),2u);ASSERT_EQ(f::goal.size(),3u);ASSERT_EQ(f::gates.size(),2u);
    ASSERT_EQ(f::history_vx.size(),4u);ASSERT_EQ(f::history_wz.size(),4u);
    ASSERT_EQ(f::path_x.size(),34u);ASSERT_EQ(f::path_y.size(),34u);ASSERT_EQ(f::path_yaw.size(),34u);
    ASSERT_EQ(f::control_vx.size(),41u);ASSERT_EQ(f::control_wz.size(),41u);
    ASSERT_DOUBLE_EQ(f::gates[0],0.023990586400032043);
    ASSERT_DOUBLE_EQ(f::gates[1],0.10000000149011612);
    ASSERT_EQ(settings_.time_steps,41u);ASSERT_FLOAT_EQ(settings_.model_dt,0.05f);
    ASSERT_EQ(settings_.batch_size,1000u);
    ASSERT_FLOAT_EQ(settings_.constraints.vx_max,0.5f);
    ASSERT_FLOAT_EQ(settings_.constraints.vx_min,-0.3f);
    ASSERT_FLOAT_EQ(settings_.constraints.wz,1.9f);
    ASSERT_FLOAT_EQ(settings_.constraints.ax_max,5.0f);
    ASSERT_FLOAT_EQ(settings_.constraints.ax_min,-5.0f);
    ASSERT_FLOAT_EQ(settings_.constraints.az_max,3.5f);
    const auto model=std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    ASSERT_TRUE(model);ASSERT_FLOAT_EQ(model->getMinTurningRadius(),0.401f);
    ASSERT_TRUE(model->predictsSteeringConditioning());ASSERT_TRUE(model->requiresConditionedSteering());
    state_.pose.header.frame_id="odom";
    state_.pose.pose.position.x=f::robot[0];state_.pose.pose.position.y=f::robot[1];
    state_.pose.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::robot[2]));
    state_.speed.linear.x=f::speed[0];state_.speed.angular.z=f::speed[1];
    setInitialSteering(f::steering[0]);setInitialConditionedSteering(f::steering[1]);
    goal_.position.x=f::goal[0];goal_.position.y=f::goal[1];
    goal_.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::goal[2]));
    path_.reset(f::path_x.size());
    std::copy(f::path_x.begin(),f::path_x.end(),path_.x.begin());
    std::copy(f::path_y.begin(),f::path_y.end(),path_.y.begin());
    std::copy(f::path_yaw.begin(),f::path_yaw.end(),path_.yaws.begin());
    setRealPathPointCount(34);setActiveCuspDirection(1);setPathDirection(1);
    setCuspTolerances(std::make_pair(f::gates[0],f::gates[1]));
    for(size_t k=0;k<control_history_.size();++k) {
      control_history_[k].vx=f::history_vx[k];control_history_[k].wz=f::history_wz[k];
      control_history_[k].vy=0;
    }
    nav_msgs::msg::Path outgoing;outgoing.header.frame_id="odom";
    for(size_t k=0;k<9;++k) {
      geometry_msgs::msg::PoseStamped target;target.header.frame_id="odom";
      target.pose.position.x=f::path_x[k];target.pose.position.y=f::path_y[k];
      target.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::path_yaw[k]));
      outgoing.poses.push_back(target);
    }
    const auto original_path=path_;
    std::copy(f::control_vx.begin(),f::control_vx.end(),control_sequence_.vx.begin());
    std::copy(f::control_wz.begin(),f::control_wz.end(),control_sequence_.wz.begin());
    const auto untouched=control_sequence_;
    ASSERT_TRUE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,
      f::gates[0],f::gates[1],false));
    EXPECT_EQ(control_sequence_.vx,untouched.vx);EXPECT_EQ(control_sequence_.wz,untouched.wz);
    if (delivery) {
      ASSERT_FALSE(initializeShortCuspMean()); // Exact41 failure authorizes81 fallback.
      EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      const auto robot=state_.pose;const auto speed=state_.speed;const auto goal=goal_;
      nav_msgs::msg::Path plan;plan.header.frame_id="odom";
      for(size_t k=0;k<f::path_x.size();++k) {
        geometry_msgs::msg::PoseStamped point;point.header=plan.header;
        point.pose.position.x=f::path_x[k];point.pose.position.y=f::path_y[k];
        point.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::path_yaw[k]));
        plan.poses.push_back(point);
      }
      reset();setRealPathPointCount(34);setActiveCuspDirection(1);setPathDirection(1);
      setCuspTolerances(std::make_pair(f::gates[0],f::gates[1]));
      requestShortCuspSeed();prepare(robot,speed,plan,goal,nullptr);
      ASSERT_GT(control_sequence_.vx(1),0.0f);
      ASSERT_TRUE(long_entry_);EXPECT_EQ(control_sequence_.vx.size(),41u);
      ASSERT_EQ(long_entry_->controls.vx.size(),81u);
      const auto target=long_entry_->target;
      // Modeled execution, not a claim of measured robot parity. Each command
      // is extracted at index1, then advanced with the concrete fixture's native
      // model and equivalent separate conditioner/servo feedback update.
      size_t ticks=0;bool cruised=false;
      for(;ticks<81;++ticks) {
        if(ticks) {prepare(state_.pose,state_.speed,plan,goal,nullptr);}
        if(!long_entry_required_) {break;}
        ASSERT_TRUE(long_entry_);
        EXPECT_EQ(long_entry_->shifts,ticks);
        ASSERT_FALSE(fallback(true,nullptr)); // No reset, family regeneration or delay restart.
        ASSERT_TRUE(long_entry_);EXPECT_EQ(long_entry_->shifts,ticks);
        control_sequence_.vx.fill(0);control_sequence_.wz.fill(0);
        const auto bad_filter=control_sequence_;
        const auto command=getControlFromSequenceAsTwist(state_.pose.header.stamp,nullptr,&bad_filter);
        ASSERT_TRUE(long_entry_);
        EXPECT_FLOAT_EQ(command.twist.linear.x,long_entry_->controls.vx(1));
        EXPECT_FLOAT_EQ(command.twist.angular.z,long_entry_->controls.wz(1));
        size_t emitted_hit=0;auto emitted_tail=long_entry_->controls;
        EXPECT_TRUE(validateLongEntrySequence(emitted_tail,emitted_hit));
        if(command.twist.linear.x>=0.019f) {cruised=true;}
        models::State next;next.reset(1,2);next.pose=state_.pose;next.speed=state_.speed;
        next.cvx(0,0)=command.twist.linear.x;next.cwz(0,0)=command.twist.angular.z;
        updateStateVelocities(next);
        const float old_conditioned=*model->initialConditionedSteering();
        const float old_physical=*model->initialSteering();
        const float speed_abs=std::abs(static_cast<float>(state_.speed.linear.x));
        const float limit=std::min(0.576f,std::atan(0.213f/0.401f));
        const float raw=std::clamp(std::atan(0.213f*next.cwz(0,0)/next.cvx(0,0)),-limit,limit);
        const float alpha=std::min(1.0f,speed_abs/0.05f);
        const float blend=1.0f-std::pow(1.0f-alpha,settings_.model_dt/0.1f);
        const float desired=std::clamp(blend*raw+(1.0f-blend)*old_conditioned,-0.576f,0.576f);
        const float rate=std::min(6.0f,0.5f+10.0f*speed_abs);
        const float conditioned=old_conditioned+std::clamp(desired-old_conditioned,
          -rate*settings_.model_dt,rate*settings_.model_dt);
        const float physical=old_physical+std::clamp(conditioned-old_physical,
          -settings_.model_dt,settings_.model_dt);
        const float native_wz=next.vx(0,1)*std::tan(physical)/0.213f;
        const float response=native_wz*std::abs(next.vx(0,1))/
          std::max(std::abs(next.vx(0,1))+0.093f*std::abs(native_wz),1e-9f);
        EXPECT_NEAR(next.wz(0,1),response,1e-6);
        const double yaw=tf2::getYaw(state_.pose.pose.orientation);
        state_.pose.pose.position.x+=state_.speed.linear.x*std::cos(yaw)*settings_.model_dt;
        state_.pose.pose.position.y+=state_.speed.linear.x*std::sin(yaw)*settings_.model_dt;
        state_.pose.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),
          yaw+state_.speed.angular.z*settings_.model_dt));
        state_.pose.header.stamp=rclcpp::Time(rclcpp::Time(state_.pose.header.stamp).nanoseconds()+
          static_cast<int64_t>(settings_.model_dt*1e9));
        state_.speed.linear.x=next.vx(0,1);state_.speed.angular.z=next.wz(0,1);
        setInitialSteering(std::nullopt);setInitialConditionedSteering(std::nullopt);
        setInitialSteering(physical);setInitialConditionedSteering(conditioned);
        shiftControlSequence();
      }
      ASSERT_LT(ticks,81u);EXPECT_GT(ticks,40u);EXPECT_TRUE(cruised);
      EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      EXPECT_LE(std::hypot(state_.pose.pose.position.x-target.position.x,
        state_.pose.pose.position.y-target.position.y),f::gates[0]);
      EXPECT_LE(std::abs(angles::shortest_angular_distance(tf2::getYaw(state_.pose.pose.orientation),
        tf2::getYaw(target.orientation))),f::gates[1]);
      const auto modeled_aligned_pose=state_.pose;
      const auto fresh=[&]() {
          reset();setInitialSteering(f::steering[0]);setInitialConditionedSteering(f::steering[1]);
          setRealPathPointCount(34);setActiveCuspDirection(1);setPathDirection(1);
          setCuspTolerances(std::make_pair(f::gates[0],f::gates[1]));requestShortCuspSeed();
          prepare(robot,speed,plan,goal,nullptr);
        };
      fresh();ASSERT_TRUE(long_entry_);
      auto shifted_plan=plan;for(auto & point:shifted_plan.poses) {point.pose.position.x+=0.001;}
      EXPECT_THROW(prepare(robot,speed,shifted_plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_); // Same frame/generation cannot conceal changed target coordinates.
      fresh();auto expired=robot;expired.header.stamp=rclcpp::Time(5000000000LL);
      EXPECT_THROW(prepare(expired,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      fresh();auto later=robot;later.header.stamp=rclcpp::Time(1000000000LL);
      ASSERT_NO_THROW(prepare(later,speed,plan,goal,nullptr));
      auto rewound=robot;rewound.header.stamp=rclcpp::Time(500000000LL);
      EXPECT_THROW(prepare(rewound,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      const auto aligned_pose=[&]() {
          auto value=modeled_aligned_pose;value.header.stamp=robot.header.stamp;return value;
        };
      fresh();auto healthy_aligned=aligned_pose();
      // Physical alignment discards the future-tail obligation. This tail must
      // never be emitted, but its invalidity cannot obstruct valid retirement.
      long_entry_->controls.vx.fill(std::numeric_limits<float>::quiet_NaN());
      ASSERT_NO_THROW(prepare(healthy_aligned,speed,plan,goal,nullptr));
      EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      fresh();auto aligned_bad_speed=aligned_pose();
      auto nan_aligned_speed=speed;nan_aligned_speed.linear.x=std::numeric_limits<double>::quiet_NaN();
      EXPECT_THROW(prepare(aligned_bad_speed,nan_aligned_speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);EXPECT_TRUE(long_entry_required_);
      fresh();auto accepted_later=robot;accepted_later.header.stamp=rclcpp::Time(1000000000LL);
      ASSERT_NO_THROW(prepare(accepted_later,speed,plan,goal,nullptr));
      auto aligned_rewind=aligned_pose();aligned_rewind.header.stamp=rclcpp::Time(500000000LL);
      EXPECT_THROW(prepare(aligned_rewind,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);EXPECT_TRUE(long_entry_required_);
      fresh();auto aligned_missing=aligned_pose();setInitialConditionedSteering(std::nullopt);
      EXPECT_THROW(prepare(aligned_missing,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);EXPECT_TRUE(long_entry_required_);
      fresh();const auto valid_footprint=costmap_ros_->getRobotFootprint();
      auto malformed=valid_footprint;malformed[0].x=std::numeric_limits<double>::quiet_NaN();
      costmap_ros_->setRobotFootprint(malformed);
      EXPECT_THROW(prepare(robot,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);costmap_ros_->setRobotFootprint(valid_footprint);
      fresh();setInitialConditionedSteering(std::nullopt);
      EXPECT_THROW(prepare(robot,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      fresh();auto bad_speed=speed;bad_speed.linear.x=std::numeric_limits<double>::quiet_NaN();
      EXPECT_THROW(prepare(robot,bad_speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      fresh();auto wrong_frame=robot;wrong_frame.header.frame_id="map";
      EXPECT_THROW(prepare(wrong_frame,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      fresh();setCuspTolerances(std::make_pair(1e-5,1e-5));
      EXPECT_THROW(prepare(robot,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      fresh();requestPlanEntrySeed();EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      fresh();setActiveCuspDirection(-1);EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      fresh();reset();EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
      fresh();ASSERT_TRUE(long_entry_);
      models::State tail;tail.reset(1,81);tail.pose=state_.pose;tail.speed=state_.speed;
      for(size_t j=0;j<80;++j) {
        tail.cvx(0,j)=long_entry_->controls.vx(j+1);tail.cwz(0,j)=long_entry_->controls.wz(j+1);
      }
      updateStateVelocities(tail);models::Trajectories forecast;forecast.reset(1,81);
      integrateStateVelocities(forecast,tail);
      nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
      const auto footprint=costmap_ros_->getRobotFootprint();
      size_t blocked_cell=22500;unsigned char blocked_old_cost=0;
      for(size_t cell=0;cell<22500;++cell) {
        const auto old_cost=costmap_->getCharMap()[cell];
        if(old_cost>=254) {continue;}
        costmap_->getCharMap()[cell]=254;
        bool early_clear=checker.footprintCostAtPose(robot.pose.position.x,robot.pose.position.y,
          tf2::getYaw(robot.pose.orientation),footprint)<254;
        bool late_hit=false;
        if(early_clear) {
          for(size_t j=41;j<81;++j) {
            if(checker.footprintCostAtPose(forecast.x(0,j),forecast.y(0,j),forecast.yaws(0,j),footprint)>=254)
            {late_hit=true;break;}
          }
        }
        if(early_clear&&late_hit) {
          for(size_t j=0;j<41;++j) {
            if(checker.footprintCostAtPose(forecast.x(0,j),forecast.y(0,j),forecast.yaws(0,j),footprint)>=254)
            {early_clear=false;break;}
          }
        }
        costmap_->getCharMap()[cell]=old_cost;
        if(early_clear&&late_hit) {blocked_cell=cell;blocked_old_cost=old_cost;break;}
      }
      ASSERT_LT(blocked_cell,22500u);
      costmap_->getCharMap()[blocked_cell]=254;
      EXPECT_THROW(prepare(robot,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_); // The first41 steps are clear; the retained tail is not.
      costmap_->getCharMap()[blocked_cell]=blocked_old_cost;
      fresh();std::fill(costmap_->getCharMap(),costmap_->getCharMap()+22500,254);
      EXPECT_THROW(prepare(robot,speed,plan,goal,nullptr),nav2_core::FailedToMakeProgress);
      EXPECT_FALSE(long_entry_);
      return;
    }
    ASSERT_TRUE(initialize_long_cusp_entry_);ASSERT_TRUE(seed_short_cusp_on_handoff_);
    // Historical checkpoint characterization, not a permanent desired behavior.
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_EQ(control_sequence_.vx,untouched.vx);EXPECT_EQ(control_sequence_.wz,untouched.wz);
    EXPECT_EQ(path_.x,original_path.x);EXPECT_EQ(path_.y,original_path.y);
    EXPECT_EQ(path_.yaws,original_path.yaws);
    EXPECT_EQ(critics_data_.real_path_point_count.value(),34u);
    EXPECT_EQ(effective_cusp_tolerances_.value(),std::make_pair(f::gates[0],f::gates[1]));
    EXPECT_EQ(settings_.time_steps,41u);
    // Counterfactual control only: an always-false/disabled initializer must not
    // satisfy this historical reduction. Nominal gates are never a repair.
    setCuspTolerances(std::make_pair(0.05,0.1));
    ASSERT_TRUE(initializeShortCuspMean());
    setCuspTolerances(std::make_pair(f::gates[0],f::gates[1]));
    control_sequence_=untouched;
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_EQ(control_sequence_.vx,untouched.vx);EXPECT_EQ(control_sequence_.wz,untouched.wz);
    EXPECT_EQ(effective_cusp_tolerances_.value(),std::make_pair(f::gates[0],f::gates[1]));

    // Forecast a concrete longer witness with native constraints/model and full
    // padded footprint. Only scratch sequence length changes, never settings.
    models::ControlSequence witness;witness.reset(81);
    for(size_t j=0;j<81;++j) {
      witness.vx(j)=j*settings_.model_dt<1.5f ? 0.003f : 0.02f;
      witness.wz(j)=-witness.vx(j)/0.401f;
    }
    control_sequence_=witness;applyControlSequenceConstraints();witness=control_sequence_;
    control_sequence_=untouched;
    models::State predicted;predicted.reset(1,81);
    predicted.pose=state_.pose;predicted.speed=state_.speed;
    for(size_t j=0;j<81;++j) {
      predicted.cvx(0,j)=witness.vx(j);predicted.cwz(0,j)=witness.wz(j);
    }
    updateStateVelocities(predicted);models::Trajectories trajectory;trajectory.reset(1,81);
    integrateStateVelocities(trajectory,predicted);
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
    const double edge=std::hypot(f::path_x[1]-f::path_x[0],f::path_y[1]-f::path_y[0]);
    bool early_aligned=false,final_aligned=false;
    for(size_t j=0;j<81;++j) {
      const double x=trajectory.x(0,j),y=trajectory.y(0,j),yaw=trajectory.yaws(0,j);
      const double cost=checker.footprintCostAtPose(x,y,yaw,costmap_ros_->getRobotFootprint());
      ASSERT_GE(cost,0);ASSERT_LT(cost,nav2_costmap_2d::LETHAL_OBSTACLE);
      if(std::hypot(x-f::robot[0],y-f::robot[1])+1e-6<edge) {continue;}
      for(size_t k=1;k<9;++k) {
        const bool aligned=std::hypot(x-f::path_x[k],y-f::path_y[k])<=f::gates[0] &&
          std::abs(angles::shortest_angular_distance(yaw,f::path_yaw[k]))<=f::gates[1];
        if(j<41) {early_aligned=early_aligned || aligned;}
        if(j==80) {final_aligned=final_aligned || aligned;}
      }
    }
    EXPECT_FALSE(early_aligned);EXPECT_TRUE(final_aligned);
    EXPECT_NEAR(trajectory.x(0,80),5.08641242980957,1e-6);
    EXPECT_NEAR(trajectory.y(0,80),0.06440215557813644,1e-6);
    EXPECT_NEAR(trajectory.yaws(0,80),-2.713496685028076,1e-6);
    EXPECT_EQ(settings_.time_steps,41u);
    EXPECT_EQ(control_sequence_.vx,untouched.vx);EXPECT_EQ(control_sequence_.wz,untouched.wz);
    // A blocked raster must still reject the unchanged acceptance policy.
    std::fill(costmap_->getCharMap(),costmap_->getCharMap()+22500,254);
    EXPECT_FALSE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,
      f::gates[0],f::gates[1],false));
  }

  void testRecordedRetainedDeparture()
  {
    namespace f = retained_departure_fixture;
    const auto pose = [](const std::vector<double> & p) {
        geometry_msgs::msg::PoseStamped value; value.header.frame_id="odom";
        value.pose.position.x=p[0]; value.pose.position.y=p[1];
        value.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),p[2]));
        return value;
      };
    const auto speed = [](const std::vector<double> & v) {
        geometry_msgs::msg::Twist value; value.linear.x=v[0];value.angular.z=v[1];return value;
      };
    nav_msgs::msg::Path outgoing; outgoing.header.frame_id="odom";
    for(size_t k=0;k<f::path_x.size();++k) {
      outgoing.poses.push_back(pose({f::path_x[k],f::path_y[k],f::path_yaw[k]}));
    }
    const auto restore = [&](const std::vector<double> & p,const std::vector<double> & v,
        const std::vector<double> & steering) {
        state_.pose=pose(p);state_.speed=speed(v);
        setInitialSteering(steering[0]);setInitialConditionedSteering(steering[1]);
      };
    // Reproduce both first-stage prefixes from the actual rejection, rather
    // than merely asserting reachability from arbitrarily shifted poses.
    const auto reverse_prefix = [&](bool carried,const std::vector<double> & expected_pose,
        const std::vector<double> & expected_speed,const std::vector<double> & expected_steering) {
        restore(f::robot0,f::speed0,f::steering0);
        models::State predicted;predicted.reset(1,41);predicted.pose=state_.pose;predicted.speed=state_.speed;
        for(size_t j=0;j<41;++j) {
          predicted.cvx(0,j)=carried ? f::incoming_control_vx[j] : -0.02f;
          predicted.cwz(0,j)=carried ? f::incoming_control_wz[j] : -0.02f/0.401f;
        }
        updateStateVelocities(predicted);models::Trajectories trajectory;trajectory.reset(1,41);
        integrateStateVelocities(trajectory,predicted);
        nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
        for(size_t j=0;j<20;++j) {
          const double cost=checker.footprintCostAtPose(trajectory.x(0,j),trajectory.y(0,j),
            trajectory.yaws(0,j),costmap_ros_->getRobotFootprint());
          EXPECT_GE(cost,0);EXPECT_LT(cost,nav2_costmap_2d::LETHAL_OBSTACLE);
        }
        EXPECT_NEAR(trajectory.x(0,19),expected_pose[0],1e-6);
        EXPECT_NEAR(trajectory.y(0,19),expected_pose[1],1e-6);
        EXPECT_NEAR(trajectory.yaws(0,19),expected_pose[2],1e-6);
        EXPECT_NEAR(predicted.vx(0,20),expected_speed[0],1e-7);
        EXPECT_NEAR(predicted.wz(0,20),expected_speed[1],1e-7);
        // Invert the calibrated response to cross-check physical steering.
        // Conditioned steering is archived separately by the diagnostic
        // recurrence with zero residual; these two settled states coincide.
        const float vx=predicted.vx(0,20),wz=predicted.wz(0,20);
        const float raw_wz=wz*std::abs(vx)/(std::abs(vx)-0.093f*std::abs(wz));
        EXPECT_NEAR(std::atan(0.213f*raw_wz/vx),expected_steering[0],1e-6);
        EXPECT_DOUBLE_EQ(expected_steering[0],expected_steering[1]);
      };
    reverse_prefix(false,f::robot1,f::speed1,f::steering1);
    reverse_prefix(true,f::robot2,f::speed2,f::steering2);
    restore(f::robot0,f::speed0,f::steering0);
    EXPECT_FALSE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    restore(f::robot1,f::speed1,f::steering1);
    EXPECT_TRUE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    restore(f::robot2,f::speed2,f::steering2);
    EXPECT_TRUE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    // Independent explicit member of the unchanged terminal family. The
    // two post-states came from clear 1s native reverse prefixes, not a drive.
    const auto terminal_geometry = [&](const std::vector<double> & p,
        const std::vector<double> & v,const std::vector<double> & steering,bool expected) {
        restore(p,v,steering);
        models::State candidate;candidate.reset(1,81);candidate.pose=state_.pose;candidate.speed=state_.speed;
        float remaining=std::hypot(f::path_x.back()-p[0],f::path_y.back()-p[1]);
        for(size_t j=0;j<81;++j) {
          const float vx=std::min(j*settings_.model_dt<1.5f ? 0.003f : 0.12f,remaining/settings_.model_dt);
          candidate.cvx(0,j)=vx;candidate.cwz(0,j)=-vx/0.401f;
          remaining=std::max(0.0f,remaining-vx*settings_.model_dt);
        }
        updateStateVelocities(candidate);models::Trajectories trajectory;trajectory.reset(1,81);
        integrateStateVelocities(trajectory,candidate);
        nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
        for(size_t j=0;j<81;++j) {
          const double cost=checker.footprintCostAtPose(trajectory.x(0,j),trajectory.y(0,j),
            trajectory.yaws(0,j),costmap_ros_->getRobotFootprint());
          EXPECT_GE(cost,0);EXPECT_LT(cost,nav2_costmap_2d::LETHAL_OBSTACLE);
        }
        const float xy=std::hypot(trajectory.x(0,80)-f::path_x.back(),trajectory.y(0,80)-f::path_y.back());
        const float yaw=std::abs(angles::shortest_angular_distance(trajectory.yaws(0,80),f::path_yaw.back()));
        EXPECT_EQ(xy<=0.05f && yaw<=0.1f,expected);
      };
    terminal_geometry(f::robot0,f::speed0,f::steering0,false);
    terminal_geometry(f::robot1,f::speed1,f::steering1,true);
    terminal_geometry(f::robot2,f::speed2,f::steering2,true);
    restore(f::robot0,f::speed0,f::steering0);
    std::fill(costmap_->getCharMap(),costmap_->getCharMap()+22500,0);
    EXPECT_FALSE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    restore(f::robot1,f::speed1,f::steering1);
    EXPECT_TRUE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    restore(f::robot2,f::speed2,f::steering2);
    EXPECT_TRUE(longCuspDepartureAvailable(state_.pose,state_.speed,outgoing,0.05,0.1,true));
    std::copy(f::grid.begin(),f::grid.end(),costmap_->getCharMap());
    restore(f::robot0,f::speed0,f::steering0);
    control_sequence_.vx.fill(-0.017f);control_sequence_.wz.fill(-0.031f);
    costs_.fill(123.0f);state_.cvx.fill(0.019f);state_.cwz.fill(0.023f);
    critics_data_.furthest_reached_path_point=3;critics_data_.path_pts_valid=std::vector<bool>{true,false,true};
    critics_data_.fail_flag=true;
    const auto before=control_sequence_;const auto costs_before=costs_;
    const auto cvx_before=state_.cvx;const auto cwz_before=state_.cwz;
    const auto valid_before=critics_data_.path_pts_valid;
    const auto cusp=outgoing.poses.front();const auto command=speed(f::command);
    EXPECT_TRUE(retainedDepartureCommandSafe(state_.pose,state_.speed,command,cusp,0.005));
    auto wrong_frame = state_.pose;wrong_frame.header.frame_id = "map";
    auto wrong_cusp = cusp;wrong_cusp.header.frame_id = "map";
    EXPECT_FALSE(retainedDepartureCommandSafe(wrong_frame,state_.speed,command,wrong_cusp,0.005));
    EXPECT_TRUE(retainedDepartureCommandSafe(state_.pose,state_.speed,command,cusp,0.01));
    auto aggressive=command;aggressive.linear.x=-0.3;aggressive.angular.z=-0.3/0.401;
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,aggressive,cusp,0.01));
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,aggressive,cusp,0.005));
    auto forward=command;forward.linear.x=0.02;
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,forward,cusp,0.01));
    EXPECT_FALSE(retainedDepartureCommandSafe(cusp,state_.speed,command,cusp,0.01));
    auto near_cusp=cusp;near_cusp.pose.position.y-=0.003;
    EXPECT_FALSE(retainedDepartureCommandSafe(near_cusp,state_.speed,command,cusp,0.01));
    auto invalid=command;invalid.linear.x=std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,invalid,cusp,0.01));
    setInitialConditionedSteering(std::nullopt);
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,command,cusp,0.005));
    setInitialConditionedSteering(f::steering0[1]);
    // Deterministic late collision: this recorded-grid cell is absent from
    // the initial and first two forecast perimeters, but enters the third.
    // A guard that checks only the current footprint must fail this control.
    auto late_command=command;late_command.linear.x=-0.12;late_command.angular.z=-0.12/0.401;
    ASSERT_TRUE(retainedDepartureCommandSafe(state_.pose,state_.speed,late_command,cusp,0.005));
    models::State late_prediction;late_prediction.reset(1,3);
    late_prediction.pose=state_.pose;late_prediction.speed=state_.speed;
    for(size_t j=0;j<3;++j) {
      late_prediction.cvx(0,j)=late_command.linear.x;
      late_prediction.cwz(0,j)=late_command.angular.z;
    }
    updateStateVelocities(late_prediction);
    const auto old_cell=costmap_->getCost(69,80);
    costmap_->setCost(69,80,nav2_costmap_2d::LETHAL_OBSTACLE);
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> late_checker(costmap_);
    double late_x=state_.pose.pose.position.x,late_y=state_.pose.pose.position.y;
    double late_yaw=tf2::getYaw(state_.pose.pose.orientation);
    ASSERT_LT(late_checker.footprintCostAtPose(late_x,late_y,late_yaw,costmap_ros_->getRobotFootprint()),
      nav2_costmap_2d::LETHAL_OBSTACLE);
    for(size_t j=0;j<3;++j) {
      late_x+=late_prediction.vx(0,j)*std::cos(late_yaw)*settings_.model_dt;
      late_y+=late_prediction.vx(0,j)*std::sin(late_yaw)*settings_.model_dt;
      late_yaw+=late_prediction.wz(0,j)*settings_.model_dt;
      const double cost=late_checker.footprintCostAtPose(late_x,late_y,late_yaw,costmap_ros_->getRobotFootprint());
      if(j<2) {EXPECT_GE(cost,0);EXPECT_LT(cost,nav2_costmap_2d::LETHAL_OBSTACLE);}
      else {EXPECT_EQ(cost,nav2_costmap_2d::LETHAL_OBSTACLE);}
    }
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,late_command,cusp,0.005));
    costmap_->setCost(69,80,old_cell);
    std::fill(costmap_->getCharMap(),costmap_->getCharMap()+22500,nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_FALSE(retainedDepartureCommandSafe(state_.pose,state_.speed,command,cusp,0.01));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,before.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,before.wz)));
    EXPECT_TRUE(xt::all(xt::equal(costs_,costs_before)));
    EXPECT_TRUE(xt::all(xt::equal(state_.cvx,cvx_before)));
    EXPECT_TRUE(xt::all(xt::equal(state_.cwz,cwz_before)));
    EXPECT_EQ(critics_data_.path_pts_valid,valid_before);
    EXPECT_EQ(critics_data_.furthest_reached_path_point.value(),3u);
    EXPECT_TRUE(critics_data_.fail_flag);
  }

  void testRecordedLongPathLocalEntry()
  {
    namespace f = handoff_prefix_fixture;
    // Recorded coordinates are odom-local; certificate validation requires the
    // same explicit identity as the configured costmap and transformed path.
    state_.pose.header.frame_id="odom";
    state_.pose.pose.position.x=f::robot[0]; state_.pose.pose.position.y=f::robot[1];
    state_.pose.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::robot[2]));
    state_.speed.linear.x=f::speed[0]; state_.speed.angular.z=f::speed[1];
    setInitialSteering(f::steering[0]); setInitialConditionedSteering(f::steering[1]);
    goal_.position.x=f::goal[0];goal_.position.y=f::goal[1];
    goal_.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::goal[2]));
    path_.reset(f::path_x.size());
    std::copy(f::path_x.begin(),f::path_x.end(),path_.x.begin());
    std::copy(f::path_y.begin(),f::path_y.end(),path_.y.begin());
    std::copy(f::path_yaw.begin(),f::path_yaw.end(),path_.yaws.begin());
    setRealPathPointCount(36); setActiveCuspDirection(1); setPathDirection(1);
    setCuspTolerances(std::make_pair(0.05,0.10));
    // Independent native oracle: retain full path and check the local first edge.
    const auto local_clear = [&](const models::ControlSequence & sequence) {
      models::State predicted; predicted.reset(1,settings_.time_steps);
      predicted.pose=state_.pose; predicted.speed=state_.speed;
      for(size_t j=0;j<settings_.time_steps;++j) {
        predicted.cvx(0,j)=sequence.vx(j); predicted.cwz(0,j)=sequence.wz(j);
      }
      updateStateVelocities(predicted);
      models::Trajectories trajectory; trajectory.reset(1,settings_.time_steps);
      integrateStateVelocities(trajectory,predicted);
      nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
      for(size_t j=0;j<settings_.time_steps;++j) {
        const auto cost=checker.footprintCostAtPose(trajectory.x(0,j),trajectory.y(0,j),
          trajectory.yaws(0,j),costmap_ros_->getRobotFootprint());
        if(cost<0 || cost>=nav2_costmap_2d::LETHAL_OBSTACLE) {return false;}
      }
      for(size_t j=1;j<settings_.time_steps;++j) {if(sequence.vx(j)<-1e-6f) {return false;}}
      const size_t last=settings_.time_steps-1;
      if(std::hypot(trajectory.x(0,last)-state_.pose.pose.position.x,
        trajectory.y(0,last)-state_.pose.pose.position.y)<
        std::hypot(path_.x(1)-path_.x(0),path_.y(1)-path_.y(0))) {return false;}
      return std::hypot(trajectory.x(0,last)-path_.x(1),trajectory.y(0,last)-path_.y(1))<=0.05 &&
        std::abs(angles::shortest_angular_distance(trajectory.yaws(0,last),path_.yaws(1)))<=0.10;
    };
    models::ControlSequence witness; witness.reset(settings_.time_steps);
    for(size_t j=0;j<settings_.time_steps;++j) {
      witness.vx(j)=j<30 ? 0.01f : 0.12f; witness.wz(j)=-witness.vx(j)/0.401f;
      control_sequence_.vx(j)=f::control_vx[j]; control_sequence_.wz(j)=f::control_wz[j];
    }
    ASSERT_TRUE(local_clear(witness));
    ASSERT_FALSE(local_clear(control_sequence_));
    ASSERT_FALSE(cuspSequenceFeasible(witness));  // Local entry is not full endpoint completion.
    const auto original_path=path_;
    control_sequence_.vx.fill(0);control_sequence_.wz.fill(0);
    const auto untouched=control_sequence_;
    initialize_long_cusp_entry_=false;
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,untouched.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,untouched.wz)));
    initialize_long_cusp_entry_=true;seed_short_cusp_on_handoff_=false;
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,untouched.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,untouched.wz)));
    seed_short_cusp_on_handoff_=true;setActiveCuspDirection(-1);
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,untouched.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,untouched.wz)));
    setActiveCuspDirection(1);
    effective_cusp_tolerances_=std::make_pair(std::numeric_limits<double>::infinity(),0.1);
    EXPECT_FALSE(initializeShortCuspMean());
    setCuspTolerances(std::make_pair(0.05,0.10));
    ASSERT_TRUE(initializeShortCuspMean());
    EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
    EXPECT_TRUE(local_clear(control_sequence_));
    EXPECT_TRUE(xt::all(xt::equal(path_.x,original_path.x)));
    EXPECT_TRUE(xt::all(xt::equal(path_.y,original_path.y)));
    EXPECT_TRUE(xt::all(xt::equal(path_.yaws,original_path.yaws)));
    EXPECT_EQ(critics_data_.real_path_point_count.value(),36u);
    EXPECT_EQ(active_cusp_direction_,1);
    EXPECT_EQ(path_direction_,1);
    EXPECT_EQ(effective_cusp_tolerances_.value(),std::make_pair(0.05,0.10));
    EXPECT_DOUBLE_EQ(goal_.position.x,f::goal[0]);
    nav_msgs::msg::Path full_plan;full_plan.header.frame_id="odom";
    for(size_t k=0;k<36;++k) {
      geometry_msgs::msg::PoseStamped target;target.header=full_plan.header;
      target.pose.position.x=f::path_x[k];target.pose.position.y=f::path_y[k];
      target.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::path_yaw[k]));
      full_plan.poses.push_back(target);
    }
    control_sequence_.vx.fill(0);control_sequence_.wz.fill(0);
    ASSERT_FALSE(local_clear(control_sequence_));
    requestShortCuspSeed();
    prepare(state_.pose,state_.speed,full_plan,goal_,nullptr);
    EXPECT_FALSE(short_cusp_seed_requested_);
    EXPECT_FALSE(long_entry_);EXPECT_FALSE(long_entry_required_);
    EXPECT_TRUE(local_clear(control_sequence_));
    control_sequence_=witness;
    prepare(state_.pose,state_.speed,full_plan,goal_,nullptr);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,witness.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,witness.wz)));
    const auto before_blocked=control_sequence_;
    std::fill(costmap_->getCharMap(),costmap_->getCharMap()+22500,nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_FALSE(initializeShortCuspMean());
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx,before_blocked.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz,before_blocked.wz)));
  }

  models::ControlSequence departureControlsForTest() const {return control_sequence_;}

  OptimizerTester()
  : Optimizer() {}

  void testFilteredCollisionRetention()
  {
    state_.pose.pose.orientation.w = 1.0;
    state_.speed = geometry_msgs::msg::Twist();
    control_sequence_.vx.fill(0.05f);
    control_sequence_.wz.fill(0.0f);
    const auto before = control_sequence_;
    control_sequence_.vx.fill(0.2f);
    applyControlSequenceConstraints();
    const auto filtered = control_sequence_;
    preserve_filtered_collision_feasibility_ = true;
    ASSERT_TRUE(controlSequenceCollisionFree());
    preserveFilteredCollisionFeasibility(&before);
    EXPECT_EQ(control_sequence_.vx, filtered.vx);
    // A wall reached only by the faster filtered sequence.
    unsigned int mx, my;
    for (double y = -0.5; y < 0.5; y += 0.025) {
      ASSERT_TRUE(costmap_->worldToMap(0.6, y, mx, my));
      costmap_->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
    ASSERT_FALSE(controlSequenceCollisionFree());
    preserve_filtered_collision_feasibility_ = false;
    preserveFilteredCollisionFeasibility(&before);
    EXPECT_EQ(control_sequence_.vx, filtered.vx);
    preserve_filtered_collision_feasibility_ = true;
    preserveFilteredCollisionFeasibility(&before);
    EXPECT_TRUE(controlSequenceCollisionFree());
    EXPECT_EQ(control_sequence_.vx, before.vx);
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(),
      nav2_costmap_2d::LETHAL_OBSTACLE);
    control_sequence_ = filtered;
    EXPECT_THROW(preserveFilteredCollisionFeasibility(&before), nav2_core::NoValidControl);
    EXPECT_EQ(control_sequence_.vx, filtered.vx);
    const char * capture = std::getenv("MPPI_FILTER_REJECTION_SNAPSHOT_PATH");
    if (capture && *capture) {
      ASSERT_TRUE(std::filesystem::exists(capture));
      std::ifstream file(capture); std::string text; std::getline(file, text);
      EXPECT_NE(text.find("Exact rejected filter means; never emitted"), std::string::npos);
      EXPECT_NE(text.find("\"before_filter_available\":true"), std::string::npos);
      EXPECT_NE(text.find("\"source_row\":63"), std::string::npos);
      EXPECT_NE(text.find("\"width\":80"), std::string::npos);
    }
  }

  void testLongPathSampleRetention()
  {
    setMotionModel("Ackermann");
    state_.pose.pose.orientation.w = 1.0;
    state_.speed = geometry_msgs::msg::Twist();
    path_.reset(5);
    for (size_t k = 0; k < 5; ++k) {path_.x(k) = 0.5f * k;}
    setRealPathPointCount(5);
    setActiveCuspDirection(1);
    costs_.fill(1e9f);
    state_.cvx.fill(0.05f); state_.cwz.fill(0.0f);
    costs_(0) = 0.0f;
    xt::view(state_.cvx, 0, xt::all()).fill(-0.05f);  // Wrong gear, even though clear.
    costs_(2) = std::numeric_limits<float>::quiet_NaN();
    costs_(3) = 3.0f;
    control_sequence_.vx.fill(0.2f); control_sequence_.wz.fill(0.0f);
    applyControlSequenceConstraints();
    const auto rejected = control_sequence_;
    unsigned int mx, my;
    for (double y = -0.5; y < 0.5; y += 0.025) {
      ASSERT_TRUE(costmap_->worldToMap(0.6, y, mx, my));
      costmap_->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    }
    ASSERT_FALSE(controlSequenceCollisionFree());
    preserve_filtered_collision_feasibility_ = true;
    EXPECT_THROW(preserveFilteredCollisionFeasibility(&rejected), nav2_core::NoValidControl);
    retain_collision_free_long_path_sample_ = true;
    ASSERT_NO_THROW(preserveFilteredCollisionFeasibility(&rejected));
    EXPECT_TRUE(controlSequenceCollisionFree());
    EXPECT_NEAR(control_sequence_.vx(1), 0.05f, 1e-6f);
    EXPECT_EQ(control_sequence_.wz, xt::zeros<float>({settings_.time_steps}));
    state_.cvx(3, 0) = state_.cvx(3, 1) = state_.cvx(3, 2) = 0.0f;
    control_sequence_ = rejected;
    ASSERT_NO_THROW(preserveFilteredCollisionFeasibility(&rejected));
    EXPECT_EQ(control_sequence_.vx(1), 0.0f);
    EXPECT_GT(control_sequence_.vx(4), 0.0f);
    xt::view(state_.cvx, 3, xt::all()).fill(0.0f);
    control_sequence_ = rejected;
    EXPECT_THROW(preserveFilteredCollisionFeasibility(&rejected), nav2_core::NoValidControl);
    xt::view(state_.cvx, 3, xt::all()).fill(0.05f);
    // Never substitute a long-path sample for strict endpoint handling.
    setRealPathPointCount(4);
    control_sequence_ = rejected;
    EXPECT_THROW(preserveFilteredCollisionFeasibility(&rejected), nav2_core::NoValidControl);
    EXPECT_EQ(control_sequence_.vx, rejected.vx);
    setRealPathPointCount(5);
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(),
      nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_THROW(preserveFilteredCollisionFeasibility(&rejected), nav2_core::NoValidControl);
    EXPECT_EQ(control_sequence_.vx, rejected.vx);
  }

  void testSparseOppositeSteeringEntry()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), 0.0));
    state_.speed.linear.x = entry_steering_fixture::speed[0];
    state_.speed.angular.z = entry_steering_fixture::speed[1];
    setInitialSteering(entry_steering_fixture::steering[0]);
    setInitialConditionedSteering(entry_steering_fixture::steering[1]);
    path_.reset(entry_steering_fixture::path_x.size());
    std::copy(entry_steering_fixture::path_x.begin(),
      entry_steering_fixture::path_x.end(), path_.x.begin());
    std::copy(entry_steering_fixture::path_y.begin(),
      entry_steering_fixture::path_y.end(), path_.y.begin());
    std::copy(entry_steering_fixture::path_yaw.begin(),
      entry_steering_fixture::path_yaw.end(), path_.yaws.begin());
    for (size_t k = 0; k < control_history_.size(); ++k) {
      control_history_[k].vx = entry_steering_fixture::history_vx[k];
      control_history_[k].wz = entry_steering_fixture::history_wz[k];
      control_history_[k].vy = 0.0f;
    }
    setRealPathPointCount(4); setActiveCuspDirection(1); setPathDirection(1);
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    condition_control_on_path_direction_ = true;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    initialize_long_cusp_entry_ = false;
    EXPECT_FALSE(initializeShortCuspMean());
    initialize_long_cusp_entry_ = true;
    ASSERT_TRUE(initializeShortCuspMean());
    EXPECT_TRUE(shortCuspSequenceFeasible());
    const auto feasible = control_sequence_;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    ASSERT_FALSE(shortCuspSequenceFeasible());
    enforceShortCuspMeanFeasibility(&feasible);
    EXPECT_TRUE(shortCuspSequenceFeasible());
    const size_t offset = settings_.shift_control_sequence ? 1 : 0;
    for (size_t k = offset; k < settings_.time_steps; ++k) {
      EXPECT_GE(control_sequence_.vx(k), -1e-6f);
    }
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() + costmap_->getSizeInCellsX() *
      costmap_->getSizeInCellsY(), nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_FALSE(initializeShortCuspMean());
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() + costmap_->getSizeInCellsX() *
      costmap_->getSizeInCellsY(), 0);
    initialize_long_cusp_entry_ = false;
    condition_control_on_path_direction_ = false;
  }

  void testLongEntryShiftRepair()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), 0.0));
    state_.speed.linear.x = entry_shift_fixture::speed[0];
    state_.speed.angular.z = entry_shift_fixture::speed[1];
    setInitialSteering(entry_shift_fixture::steering[0]);
    setInitialConditionedSteering(entry_shift_fixture::steering[1]);
    path_.reset(entry_shift_fixture::path_x.size());
    std::copy(entry_shift_fixture::path_x.begin(), entry_shift_fixture::path_x.end(), path_.x.begin());
    std::copy(entry_shift_fixture::path_y.begin(), entry_shift_fixture::path_y.end(), path_.y.begin());
    std::copy(entry_shift_fixture::path_yaw.begin(), entry_shift_fixture::path_yaw.end(), path_.yaws.begin());
    setRealPathPointCount(11); setActiveCuspDirection(-1); setPathDirection(-1);
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    initialize_long_cusp_entry_ = true;
    models::ControlSequence warm; warm.reset(settings_.time_steps);
    for (size_t j = 0; j < settings_.time_steps; ++j) {
      control_sequence_.vx(j) = entry_shift_fixture::emitted_vx[j];
      control_sequence_.wz(j) = entry_shift_fixture::emitted_wz[j];
      warm.vx(j) = entry_shift_fixture::warm_vx[j];
      warm.wz(j) = entry_shift_fixture::warm_wz[j];
    }
    const auto invalid = control_sequence_;
    ASSERT_FALSE(shortCuspSequenceFeasible());
    repair_long_cusp_warm_start_ = false;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, invalid.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, invalid.wz)));
    repair_long_cusp_warm_start_ = true;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(shortCuspSequenceFeasible());
    EXPECT_LT(control_sequence_.vx(1), -0.01f);  // Repair is not a zero-speed hold.
    const auto repaired = control_sequence_;
    // A feasible optimized result has priority over every repair candidate.
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, repaired.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, repaired.wz)));
    const size_t cells = costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY();
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() + cells,
      nav2_costmap_2d::LETHAL_OBSTACLE);
    control_sequence_ = invalid;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, invalid.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, invalid.wz)));
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() + cells, 0);
    // An invalid warm sequence must not manufacture a feasibility certificate.
    warm.vx.fill(0.0f); warm.wz.fill(0.0f);
    control_sequence_ = invalid;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, invalid.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, invalid.wz)));
    repair_long_cusp_warm_start_ = false;
    initialize_long_cusp_entry_ = false;
  }

  void testLongEntryRetention()
  {
    settings_.constraints.ax_max = 5.0f;
    settings_.constraints.ax_min = -5.0f;
    settings_.constraints.vx_max = 0.5f;
    settings_.constraints.vx_min = -0.3f;
    motion_model_->initialize(settings_.constraints, settings_.model_dt);
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), 0.0));
    state_.speed.linear.x = long_entry_fixture::speed[0];
    state_.speed.angular.z = long_entry_fixture::speed[1];
    setInitialSteering(long_entry_fixture::steering[0]);
    setInitialConditionedSteering(long_entry_fixture::steering[1]);
    path_.reset(long_entry_fixture::path_x.size());
    std::copy(long_entry_fixture::path_x.begin(),
      long_entry_fixture::path_x.end(), path_.x.begin());
    std::copy(long_entry_fixture::path_y.begin(),
      long_entry_fixture::path_y.end(), path_.y.begin());
    std::copy(long_entry_fixture::path_yaw.begin(),
      long_entry_fixture::path_yaw.end(), path_.yaws.begin());
    setRealPathPointCount(12); setActiveCuspDirection(-1); setPathDirection(-1);
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    const auto load = [&](bool seed) {
        for (size_t j = 0; j < 81; ++j) {
          control_sequence_.vx(j) = seed ?
            long_entry_fixture::seed_vx[j] : long_entry_fixture::emitted_vx[j];
          control_sequence_.wz(j) = seed ?
            long_entry_fixture::seed_wz[j] : long_entry_fixture::emitted_wz[j];
        }
      };
    load(true); ASSERT_TRUE(shortCuspSequenceFeasible());
    const auto feasible = control_sequence_;
    load(false); ASSERT_FALSE(shortCuspSequenceFeasible());
    const auto invalid = control_sequence_;
    initialize_long_cusp_entry_ = false;
    short_cusp_seed_requested_ = false;
    requestPlanEntrySeed();
    EXPECT_FALSE(short_cusp_seed_requested_);
    enforceShortCuspMeanFeasibility(&feasible);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, invalid.vx)));
    initialize_long_cusp_entry_ = true;
    requestPlanEntrySeed();
    EXPECT_TRUE(short_cusp_seed_requested_);
    short_cusp_seed_requested_ = false;
    enforceShortCuspMeanFeasibility(&feasible);
    EXPECT_TRUE(shortCuspSequenceFeasible());
    load(false);
    enforceShortCuspMeanFeasibility(&invalid);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, invalid.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, invalid.wz)));
    ASSERT_TRUE(initializeShortCuspMean());
    EXPECT_TRUE(shortCuspSequenceFeasible());
    initialize_long_cusp_entry_ = false;
  }

  void testHiddenFutureGearMean()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), hidden_future_gear_fixture::robot_yaw[0]));
    state_.speed.linear.x = hidden_future_gear_fixture::speed[0];
    state_.speed.angular.z = hidden_future_gear_fixture::speed[1];
    setInitialSteering(hidden_future_gear_fixture::steering[0]);
    setInitialConditionedSteering(hidden_future_gear_fixture::steering[1]);
    path_.reset(hidden_future_gear_fixture::path_x.size());
    std::copy(hidden_future_gear_fixture::path_x.begin(), hidden_future_gear_fixture::path_x.end(), path_.x.begin());
    std::copy(hidden_future_gear_fixture::path_y.begin(), hidden_future_gear_fixture::path_y.end(), path_.y.begin());
    std::copy(hidden_future_gear_fixture::path_yaw.begin(), hidden_future_gear_fixture::path_yaw.end(), path_.yaws.begin());
    setRealPathPointCount(3); setActiveCuspDirection(1); setPathDirection(1);
    preserve_short_cusp_feasibility_ = true; seed_short_cusp_on_handoff_ = true;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    control_history_.fill({0.0f, 0.0f, 0.0f});
    for (size_t b = 0; b < settings_.batch_size; ++b) {
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(b, j) = hidden_future_gear_fixture::control_vx[j];
        state_.cwz(b, j) = hidden_future_gear_fixture::control_wz[j];
      }
    }
    // Only commands are bounded; initial physical velocity and steering persist.
    constrainShortCuspSampleDirections();
    EXPECT_GE(xt::amin(state_.cvx)(), 0.0f);
    // Deterministic equal rows retain the recorded tail through native
    // prediction, weighting, smoothing and final command constraints.
    updateStateVelocities(state_);
    costs_.fill(0.0f);
    updateControlSequence();
    smoothControlSequence();
    applyControlSequenceConstraints();
    EXPECT_GE(xt::amin(control_sequence_.vx)(), -1e-6f);
    EXPECT_DOUBLE_EQ(state_.speed.linear.x, hidden_future_gear_fixture::speed[0]);
    EXPECT_DOUBLE_EQ(state_.speed.angular.z, hidden_future_gear_fixture::speed[1]);
  }

  void testLongPrefixGearMean()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), long_prefix_gear_fixture::robot_yaw[0]));
    state_.speed.linear.x = long_prefix_gear_fixture::speed[0];
    state_.speed.angular.z = long_prefix_gear_fixture::speed[1];
    setInitialSteering(long_prefix_gear_fixture::steering[0]);
    setInitialConditionedSteering(long_prefix_gear_fixture::steering[1]);
    path_.reset(long_prefix_gear_fixture::path_x.size());
    std::copy(long_prefix_gear_fixture::path_x.begin(), long_prefix_gear_fixture::path_x.end(), path_.x.begin());
    std::copy(long_prefix_gear_fixture::path_y.begin(), long_prefix_gear_fixture::path_y.end(), path_.y.begin());
    std::copy(long_prefix_gear_fixture::path_yaw.begin(), long_prefix_gear_fixture::path_yaw.end(), path_.yaws.begin());
    setRealPathPointCount(34); setActiveCuspDirection(1); setPathDirection(1);
    preserve_short_cusp_feasibility_ = false; seed_short_cusp_on_handoff_ = false;
    condition_control_on_path_direction_ = true;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    control_history_.fill({0.0f, 0.0f, 0.0f});
    for (size_t b = 0; b < settings_.batch_size; ++b) {
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(b, j) = long_prefix_gear_fixture::control_vx[j];
        state_.cwz(b, j) = long_prefix_gear_fixture::control_wz[j];
      }
    }
    // Only commands are bounded; initial physical velocity and steering persist.
    constrainShortCuspSampleDirections();
    EXPECT_GE(xt::amin(state_.cvx)(), 0.0f);
    // Deterministic equal rows retain the recorded tail through native
    // prediction, weighting, smoothing and final command constraints.
    updateStateVelocities(state_);
    costs_.fill(0.0f);
    updateControlSequence();
    smoothControlSequence();
    applyControlSequenceConstraints();
    EXPECT_GE(xt::amin(control_sequence_.vx)(), -1e-6f);
    EXPECT_DOUBLE_EQ(state_.speed.linear.x, long_prefix_gear_fixture::speed[0]);
    EXPECT_DOUBLE_EQ(state_.speed.angular.z, long_prefix_gear_fixture::speed[1]);
    condition_control_on_path_direction_ = false;
  }

  void testOppositeFeedbackDirectionRequests()
  {
    condition_control_on_path_direction_ = true;
    preserve_short_cusp_feasibility_ = false;
    seed_short_cusp_on_handoff_ = false;
    path_.reset(42); setRealPathPointCount(42); setActiveCuspDirection(1);
    state_.speed.linear.x = opposite_feedback_gear_fixture::speed[0];
    state_.speed.angular.z = opposite_feedback_gear_fixture::speed[1];
    setInitialSteering(opposite_feedback_gear_fixture::steering[0]);
    setInitialConditionedSteering(opposite_feedback_gear_fixture::steering[1]);
    for (size_t b = 0; b < settings_.batch_size; ++b) {
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        state_.cvx(b, j) = opposite_feedback_gear_fixture::control_vx[j];
        state_.cwz(b, j) = opposite_feedback_gear_fixture::control_wz[j];
      }
    }
    EXPECT_EQ(shortCuspControlDirection(), 1);
    constrainShortCuspSampleDirections();
    EXPECT_GE(xt::amin(state_.cvx)(), 0.0f);
    control_sequence_.vx = xt::view(state_.cvx, 0, xt::all());
    control_sequence_.wz = xt::view(state_.cwz, 0, xt::all());
    applyControlSequenceConstraints();
    EXPECT_GE(xt::amin(control_sequence_.vx)(), 0.0f);

    // A deliberately slow acceleration bound proves request conditioning does
    // not replace measured old-gear motion with an instantaneous velocity flip.
    auto constraints = settings_.constraints;
    constraints.ax_max = 0.1f; constraints.ax_min = -0.1f;
    motion_model_->initialize(constraints, settings_.model_dt);
    updateStateVelocities(state_);
    EXPECT_FLOAT_EQ(state_.vx(0, 0), opposite_feedback_gear_fixture::speed[0]);
    EXPECT_LT(state_.vx(0, 1), 0.0f);
    EXPECT_GE(state_.vx(0, 1), state_.vx(0, 0));
    EXPECT_LE(state_.vx(0, 1) - state_.vx(0, 0), 0.005001f);
    EXPECT_DOUBLE_EQ(state_.speed.linear.x, opposite_feedback_gear_fixture::speed[0]);
    motion_model_->initialize(settings_.constraints, settings_.model_dt);
    state_.speed.linear.x = 0.024f; setActiveCuspDirection(-1);
    EXPECT_EQ(shortCuspControlDirection(), -1);
    state_.cvx.fill(0.1f); constrainShortCuspSampleDirections();
    EXPECT_LE(xt::amax(state_.cvx)(), 0.0f);
    condition_control_on_path_direction_ = false;
    EXPECT_EQ(shortCuspControlDirection(), 0);
    condition_control_on_path_direction_ = true; setActiveCuspDirection(0);
    EXPECT_EQ(shortCuspControlDirection(), 0);
    condition_control_on_path_direction_ = false;
  }

  void testShortCuspDirectionScope()
  {
    const auto check = [&](bool enabled, int gear, size_t real, float speed, bool bounded) {
        preserve_short_cusp_feasibility_ = enabled;
        seed_short_cusp_on_handoff_ = true;
        setActiveCuspDirection(gear); setRealPathPointCount(real);
        path_.reset(6); state_.speed.linear.x = speed;
        state_.cvx.fill(gear == -1 ? 0.1f : -0.1f);
        const auto before = state_.cvx;
        constrainShortCuspSampleDirections();
        if (bounded) {
          EXPECT_EQ(xt::sum(xt::fabs(state_.cvx))(), 0.0f);
        } else {
          EXPECT_TRUE(xt::all(xt::equal(state_.cvx, before)));
        }
        EXPECT_FLOAT_EQ(state_.speed.linear.x, speed);
      };
    check(true, 1, 3, 0.0f, true);
    check(true, -1, 3, -0.05f, true);
    check(false, 1, 3, 0.0f, false);
    check(true, 0, 3, 0.0f, false);  // Final segment.
    check(true, 1, 5, 0.0f, false);  // Long prefix.
    check(true, 1, 0, 0.0f, false);
    check(true, 1, 3, -0.05f, false);  // Physical old-gear braking.
    check(true, -1, 3, 0.05f, false);
    seed_short_cusp_on_handoff_ = false;
    EXPECT_EQ(shortCuspControlDirection(), 0);
    condition_control_on_path_direction_ = true;
    check(false, 1, 5, 0.02f, true);
    check(false, -1, 5, -0.02f, true);
    check(false, 1, 5, -0.02f, true);  // Requests, not measured braking velocity.
    check(false, -1, 5, 0.02f, true);
    check(false, 0, 5, 0.0f, false);
    condition_control_on_path_direction_ = false;
  }

  void testRecordedEffectiveCuspGate()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation.w = 1.0;
    state_.speed.linear.x = effective_gate_fixture::speed[0];
    state_.speed.angular.z = effective_gate_fixture::speed[1];
    setInitialSteering(effective_gate_fixture::steering[0]);
    setInitialConditionedSteering(effective_gate_fixture::steering[1]);
    path_.reset(effective_gate_fixture::path_x.size());
    std::copy(effective_gate_fixture::path_x.begin(), effective_gate_fixture::path_x.end(), path_.x.begin());
    std::copy(effective_gate_fixture::path_y.begin(), effective_gate_fixture::path_y.end(), path_.y.begin());
    std::copy(effective_gate_fixture::path_yaw.begin(), effective_gate_fixture::path_yaw.end(), path_.yaws.begin());
    std::copy(effective_gate_fixture::vx.begin(), effective_gate_fixture::vx.end(), control_sequence_.vx.begin());
    std::copy(effective_gate_fixture::wz.begin(), effective_gate_fixture::wz.end(), control_sequence_.wz.begin());
    setRealPathPointCount(1); setActiveCuspDirection(1); setPathDirection(1);
    ASSERT_TRUE(shortCuspSequenceFeasible());  // Legacy raw5cm gate.
    setCuspTolerances(std::make_pair(0.023990616202354431, 0.10));
    EXPECT_FALSE(shortCuspSequenceFeasible());  // Must honor the actual handoff gate.
    setCuspTolerances(std::nullopt);
    EXPECT_TRUE(shortCuspSequenceFeasible());  // Standalone caller remains compatible.
    setCuspTolerances(std::make_pair(0.023990616202354431, 0.10));
    fallback(false);
    ASSERT_TRUE(fallback(true));
    ASSERT_TRUE(effective_cusp_tolerances_.has_value());
    EXPECT_DOUBLE_EQ(cuspTolerances().first, 0.023990616202354431);
    fallback(false);
    setActiveCuspDirection(0);
    EXPECT_FALSE(effective_cusp_tolerances_.has_value());
    setActiveCuspDirection(1);
    EXPECT_DOUBLE_EQ(cuspTolerances().first, 0.05);
    setCuspTolerances(std::make_pair(0.023990616202354431, 0.10));
    reset();
    EXPECT_FALSE(effective_cusp_tolerances_.has_value());
    EXPECT_THROW(setCuspTolerances(std::make_pair(-0.01, 0.10)), nav2_core::InvalidPath);
    EXPECT_THROW(setCuspTolerances(std::make_pair(0.02, 0.0)), nav2_core::InvalidPath);
  }

  void testExitCompletionContinuation()
  {
    state_.pose.pose.orientation.w = 1.0;
    state_.speed.linear.x = exit_completion_fixture::speed[0];
    state_.speed.angular.z = exit_completion_fixture::speed[1];
    setInitialSteering(exit_completion_fixture::steering[0]);
    setInitialConditionedSteering(exit_completion_fixture::steering[1]);
    path_.reset(exit_completion_fixture::path_x.size());
    std::copy(exit_completion_fixture::path_x.begin(), exit_completion_fixture::path_x.end(), path_.x.begin());
    std::copy(exit_completion_fixture::path_y.begin(), exit_completion_fixture::path_y.end(), path_.y.begin());
    std::copy(exit_completion_fixture::path_yaw.begin(), exit_completion_fixture::path_yaw.end(), path_.yaws.begin());
    setRealPathPointCount(3); setActiveCuspDirection(1); setPathDirection(1);
    setCuspTolerances(std::make_pair(0.05, 0.10));
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    const auto rejected = control_sequence_;
    EXPECT_FALSE(tryExtendedShortCuspContinuation(true));  // Default off.
    extend_short_cusp_completion_ = true;
    ASSERT_TRUE(tryExtendedShortCuspContinuation(true));
    ASSERT_TRUE(cusp_completion_.has_value());
    EXPECT_EQ(control_sequence_.vx.size(), 41u);
    EXPECT_EQ(cusp_completion_->vx.size(), 81u);
    EXPECT_FALSE(shortCuspSequenceFeasible());  // Prefix alone does not finish.
    const auto sampled_state = state_;
    const auto sampled_costs = costs_;
    critics_data_.fail_flag = true;
    EXPECT_FALSE(fallback(true, nullptr));  // Retain before a sampled reset.
    EXPECT_FALSE(critics_data_.fail_flag);
    EXPECT_EQ(state_.cvx, sampled_state.cvx);
    EXPECT_EQ(costs_, sampled_costs);
    enforceShortCuspMeanFeasibility();  // Guard must accept the checked full tail.
    ASSERT_TRUE(cusp_completion_.has_value());
    const auto completion = *cusp_completion_;
    shiftControlSequence();
    ASSERT_TRUE(tryExtendedShortCuspContinuation(false));
    EXPECT_FLOAT_EQ(cusp_completion_->vx(49), completion.vx(50));
    EXPECT_EQ(cusp_completion_shifts_, 1u);
    const auto prefix = control_sequence_;
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(), nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_FALSE(tryExtendedShortCuspContinuation(false));
    EXPECT_EQ(control_sequence_.vx, prefix.vx);
    EXPECT_TRUE(fallback(true, nullptr));  // Internal retry must not renew the delay.
    state_ = sampled_state;
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(), 0);
    EXPECT_FALSE(tryExtendedShortCuspContinuation(true));  // Never restart delay after invalidation.
    requestShortCuspSeed();
    ASSERT_TRUE(tryExtendedShortCuspContinuation(true));
    setActiveCuspDirection(0);
    EXPECT_FALSE(tryExtendedShortCuspContinuation(true));
    setActiveCuspDirection(1); setRealPathPointCount(5);
    EXPECT_FALSE(tryExtendedShortCuspContinuation(true));
    setRealPathPointCount(3); requestPlanEntrySeed();
    setCuspTolerances(std::make_pair(0.05, 0.10));
    ASSERT_TRUE(tryExtendedShortCuspContinuation(true));
    cusp_completion_->vx = -xt::fabs(cusp_completion_->vx);
    EXPECT_FALSE(tryExtendedShortCuspContinuation(false));
    EXPECT_EQ(settings_.time_steps, 41u);
    EXPECT_EQ(state_.cvx.shape(1), 41u);
    EXPECT_EQ(generated_trajectories_.x.shape(1), 41u);
    control_sequence_ = rejected;
  }

  void testSampleFailurePreservesWarmStart()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, 0, 1));
    state_.speed.linear.x = sampled_failure_fixture::speed[0];
    state_.speed.angular.z = sampled_failure_fixture::speed[1];
    setInitialSteering(sampled_failure_fixture::steering[0]);
    setInitialConditionedSteering(sampled_failure_fixture::steering[1]);
    path_.reset(sampled_failure_fixture::path_x.size());
    std::copy(sampled_failure_fixture::path_x.begin(), sampled_failure_fixture::path_x.end(), path_.x.begin());
    std::copy(sampled_failure_fixture::path_y.begin(), sampled_failure_fixture::path_y.end(), path_.y.begin());
    std::copy(sampled_failure_fixture::path_yaw.begin(), sampled_failure_fixture::path_yaw.end(), path_.yaws.begin());
    std::copy(sampled_failure_fixture::warm_vx.begin(), sampled_failure_fixture::warm_vx.end(), control_sequence_.vx.begin());
    std::copy(sampled_failure_fixture::warm_wz.begin(), sampled_failure_fixture::warm_wz.end(), control_sequence_.wz.begin());
    setRealPathPointCount(2); setActiveCuspDirection(1); setPathDirection(1);
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    applyControlSequenceConstraints();
    ASSERT_TRUE(shortCuspSequenceFeasible());
    const auto warm = control_sequence_;
    fallback(false);
    critics_data_.fail_flag = true;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    EXPECT_FALSE(fallback(true, &warm));
    EXPECT_FALSE(critics_data_.fail_flag);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, warm.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, warm.wz)));
    EXPECT_TRUE(shortCuspSequenceFeasible());
    fallback(false);
    const auto rejects = [&](const models::ControlSequence * candidate) {
        control_sequence_ = warm;
        EXPECT_TRUE(fallback(true, candidate));
        EXPECT_FLOAT_EQ(xt::sum(xt::fabs(control_sequence_.vx))(), 0.0f);
        fallback(false);
      };
    preserve_short_cusp_feasibility_ = false;
    rejects(&warm);
    preserve_short_cusp_feasibility_ = true;
    rejects(nullptr);
    auto wrong_gear = warm;
    wrong_gear.vx = -xt::fabs(wrong_gear.vx);
    rejects(&wrong_gear);
    setActiveCuspDirection(0);
    rejects(&warm);
    setActiveCuspDirection(1);
    setRealPathPointCount(5);
    rejects(&warm);
    setRealPathPointCount(2);
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(),
      nav2_costmap_2d::LETHAL_OBSTACLE);
    rejects(&warm);
  }

  void testRecordedWarmStartDelay()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), warm_start_fixture::robot_yaw[0]));
    state_.speed.linear.x = warm_start_fixture::speed[0];
    state_.speed.angular.z = warm_start_fixture::speed[1];
    setInitialSteering(warm_start_fixture::steering[0]);
    setInitialConditionedSteering(warm_start_fixture::steering[1]);
    path_.reset(warm_start_fixture::path_x.size());
    std::copy(warm_start_fixture::path_x.begin(), warm_start_fixture::path_x.end(), path_.x.begin());
    std::copy(warm_start_fixture::path_y.begin(), warm_start_fixture::path_y.end(), path_.y.begin());
    std::copy(warm_start_fixture::path_yaw.begin(), warm_start_fixture::path_yaw.end(), path_.yaws.begin());
    std::copy(warm_start_fixture::warm_vx.begin(), warm_start_fixture::warm_vx.end(), control_sequence_.vx.begin());
    std::copy(warm_start_fixture::warm_wz.begin(), warm_start_fixture::warm_wz.end(), control_sequence_.wz.begin());
    setRealPathPointCount(4); setActiveCuspDirection(1); setPathDirection(1);
    ASSERT_TRUE(shortCuspSequenceFeasible());
    const auto warm = control_sequence_;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    ASSERT_FALSE(shortCuspSequenceFeasible());
    seed_short_cusp_on_handoff_ = true; preserve_short_cusp_feasibility_ = true;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, warm.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, warm.wz)));
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    const auto command = getControlFromSequenceAsTwist(builtin_interfaces::msg::Time(), &warm);
    EXPECT_FLOAT_EQ(command.twist.linear.x, warm.vx(1));
    EXPECT_FLOAT_EQ(command.twist.angular.z, warm.wz(1));
    EXPECT_FLOAT_EQ(control_history_.back().vx, command.twist.linear.x);
    EXPECT_FLOAT_EQ(control_history_.back().wz, command.twist.angular.z);
    auto wrong_gear = warm;
    wrong_gear.vx = -xt::fabs(wrong_gear.vx);
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    enforceShortCuspMeanFeasibility(&wrong_gear);
    EXPECT_TRUE(shortCuspSequenceFeasible());
    EXPECT_GE(xt::amin(control_sequence_.vx)(), -1e-6f);
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    preserve_short_cusp_feasibility_ = false;
    enforceShortCuspMeanFeasibility(&warm);
    EXPECT_EQ(xt::sum(xt::fabs(control_sequence_.vx))(), 0.0f);
    preserve_short_cusp_feasibility_ = true;
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(), nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_THROW(enforceShortCuspMeanFeasibility(&warm), nav2_core::FailedToMakeProgress);
    EXPECT_EQ(xt::sum(xt::fabs(control_sequence_.vx))(), 0.0f);
  }

  void testRecordedMixedGearMean()
  {
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), -2.2830160689876235));
    state_.speed.linear.x = 0.0;
    state_.speed.angular.z = -0.0032322449110119035;
    setInitialSteering(-0.10846415162086487f);
    setInitialConditionedSteering(-0.10847083479166031f);
    path_.reset(mixed_gear_fixture::path_x.size());
    std::copy(mixed_gear_fixture::path_x.begin(), mixed_gear_fixture::path_x.end(), path_.x.begin());
    std::copy(mixed_gear_fixture::path_y.begin(), mixed_gear_fixture::path_y.end(), path_.y.begin());
    std::copy(mixed_gear_fixture::path_yaw.begin(), mixed_gear_fixture::path_yaw.end(), path_.yaws.begin());
    std::copy(mixed_gear_fixture::vx.begin(), mixed_gear_fixture::vx.end(), control_sequence_.vx.begin());
    std::copy(mixed_gear_fixture::wz.begin(), mixed_gear_fixture::wz.end(), control_sequence_.wz.begin());
    setRealPathPointCount(3); setActiveCuspDirection(-1); setPathDirection(-1);
    EXPECT_GT(xt::amax(control_sequence_.vx)(), 0.07f);
    EXPECT_FALSE(shortCuspSequenceFeasible());
  }

  void testRecordedEmittedMeanGuard()
  {
    state_.pose.pose.position.x = 0.28660268570916814;
    state_.pose.pose.position.y = -0.07859553870288444;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), -0.5596634286881496));
    state_.speed.linear.x = 0.04095176460971594;
    state_.speed.angular.z = -0.06710568015687907;
    setInitialSteering(-0.4580419659614563f);
    setInitialConditionedSteering(-0.4383373260498047f);
    path_.reset(6);
    path_.x = {0.2894269824f, 0.2509178817f, 0.2107887417f,
      0.1706596017f, 0.1305304468f, 0.0904012993f};
    path_.y = {-0.0849037841f, -0.0562809072f, -0.0264538825f,
      0.0033731416f, 0.0332001671f, 0.0630271882f};
    path_.yaws = {-0.5955505371f, -0.6828174591f, -0.6391837001f,
      -0.6391837001f, -0.6391837001f, -0.6391837001f};
    setRealPathPointCount(2); setActiveCuspDirection(-1); setPathDirection(-1);
    goal_.position.x = 0.226939383; goal_.position.y = 0.096776931;
    goal_.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), -1.380948995));
    critics_data_.motion_model = motion_model_;
    const auto load_mean = [&](bool seed) {
        for (size_t j = 0; j < 81; ++j) {
          control_sequence_.vx(j) = seed ? cusp_mean_fixture::seed_vx[j] :
            cusp_mean_fixture::emitted_vx[j];
          control_sequence_.wz(j) = seed ? cusp_mean_fixture::seed_wz[j] :
            cusp_mean_fixture::emitted_wz[j];
        }
      };
    load_mean(true);
    EXPECT_TRUE(shortCuspSequenceFeasible());
    load_mean(false);
    EXPECT_FALSE(shortCuspSequenceFeasible());
    seed_short_cusp_on_handoff_ = true;
    preserve_short_cusp_feasibility_ = false;
    const auto old_mean = control_sequence_;
    enforceShortCuspMeanFeasibility();
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, old_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, old_mean.wz)));

    preserve_short_cusp_feasibility_ = true;
    state_.cvx.fill(0.123f); generated_trajectories_.x.fill(0.456f);
    costs_.fill(0.789f); critics_data_.furthest_reached_path_point = 2;
    enforceShortCuspMeanFeasibility();
    EXPECT_TRUE(shortCuspSequenceFeasible());
    EXPECT_TRUE(xt::all(xt::equal(state_.cvx, 0.123f)));
    EXPECT_TRUE(xt::all(xt::equal(generated_trajectories_.x, 0.456f)));
    EXPECT_TRUE(xt::all(xt::equal(costs_, 0.789f)));
    EXPECT_EQ(critics_data_.furthest_reached_path_point.value(), 2u);
    const auto valid_mean = control_sequence_;
    enforceShortCuspMeanFeasibility();
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, valid_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, valid_mean.wz)));
    load_mean(false);
    const auto command = getControlFromSequenceAsTwist(builtin_interfaces::msg::Time());
    EXPECT_TRUE(shortCuspSequenceFeasible());
    EXPECT_FLOAT_EQ(command.twist.linear.x, control_sequence_.vx(1));
    EXPECT_FLOAT_EQ(command.twist.angular.z, control_sequence_.wz(1));
    EXPECT_FLOAT_EQ(control_history_.back().vx, command.twist.linear.x);
    EXPECT_FLOAT_EQ(control_history_.back().wz, command.twist.angular.z);

    // Diagnostic capture must neither rewrite the selected mean nor disturb
    // the Gaussian batch. Only active short-cusp emissions belong in the file.
    const auto trace = std::filesystem::temp_directory_path() /
      ("office_cusp_mean_" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count()) + ".jsonl");
    const auto captured_mean = control_sequence_;
    ASSERT_EQ(setenv("MPPI_CUSP_MEAN_TRACE_PATH", trace.c_str(), 1), 0);
    const auto traced = getControlFromSequenceAsTwist(builtin_interfaces::msg::Time());
    EXPECT_FLOAT_EQ(traced.twist.linear.x, command.twist.linear.x);
    EXPECT_FLOAT_EQ(traced.twist.angular.z, command.twist.angular.z);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, captured_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, captured_mean.wz)));
    EXPECT_TRUE(xt::all(xt::equal(state_.cvx, 0.123f)));
    std::ifstream recorded(trace);
    ASSERT_TRUE(recorded.good());
    std::string text, line;
    while (std::getline(recorded, line)) {text += line + '\n';}
    EXPECT_NE(text.find("\"command_offset\":1"), std::string::npos);
    EXPECT_NE(text.find("\"initial_conditioned_steering\":"), std::string::npos);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
    if (const char * receipt = std::getenv("MPPI_CUSP_MEAN_TRACE_TEST_OUTPUT")) {
      std::ofstream copy(receipt); copy << text;
      ASSERT_TRUE(copy.good());
    }
    const auto bytes = std::filesystem::file_size(trace);
    setActiveCuspDirection(0);
    getControlFromSequenceAsTwist(builtin_interfaces::msg::Time());
    EXPECT_EQ(std::filesystem::file_size(trace), bytes);
    // Explicit all-path mode captures final and long approaches without
    // changing the control sequence or touching the sampled trajectories.
    ASSERT_EQ(setenv("MPPI_EMITTED_MEAN_TRACE_ALL_PATHS", "1", 1), 0);
    getControlFromSequenceAsTwist(builtin_interfaces::msg::Time());
    EXPECT_GT(std::filesystem::file_size(trace), bytes);
    const auto final_bytes = std::filesystem::file_size(trace);
    setActiveCuspDirection(-1); setRealPathPointCount(6);
    getControlFromSequenceAsTwist(builtin_interfaces::msg::Time());
    EXPECT_GT(std::filesystem::file_size(trace), final_bytes);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, captured_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, captured_mean.wz)));
    EXPECT_TRUE(xt::all(xt::equal(generated_trajectories_.x, 0.456f)));
    EXPECT_TRUE(xt::all(xt::equal(state_.cvx, 0.123f)));
    ASSERT_EQ(unsetenv("MPPI_EMITTED_MEAN_TRACE_ALL_PATHS"), 0);
    setRealPathPointCount(2);
    setActiveCuspDirection(-1);
    ASSERT_EQ(setenv("MPPI_CUSP_MEAN_TRACE_PATH", "/missing-office-trace-dir/mean.jsonl", 1), 0);
    EXPECT_NO_THROW(getControlFromSequenceAsTwist(builtin_interfaces::msg::Time()));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, captured_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, captured_mean.wz)));
    ASSERT_EQ(unsetenv("MPPI_CUSP_MEAN_TRACE_PATH"), 0);
    recorded.close();
    std::filesystem::remove(trace);

    load_mean(false); setActiveCuspDirection(0);  // Final goal excluded.
    enforceShortCuspMeanFeasibility();
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, old_mean.vx)));
    setActiveCuspDirection(-1); setRealPathPointCount(5);  // Long prefix excluded.
    enforceShortCuspMeanFeasibility();
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.wz, old_mean.wz)));
    setRealPathPointCount(2);
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(), nav2_costmap_2d::LETHAL_OBSTACLE);
    EXPECT_THROW(enforceShortCuspMeanFeasibility(), nav2_core::FailedToMakeProgress);
    EXPECT_TRUE(xt::all(xt::equal(control_sequence_.vx, old_mean.vx)));
    EXPECT_TRUE(xt::all(xt::equal(costs_, 0.789f)));
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(), nav2_costmap_2d::NO_INFORMATION);
    EXPECT_THROW(enforceShortCuspMeanFeasibility(), nav2_core::FailedToMakeProgress);
  }

  bool recordedShortCuspSeed(bool enabled, int gear, size_t real = 2,
    float yaw_offset = 0.0f, bool closer_handoff = false)
  {
    seed_short_cusp_on_handoff_ = enabled;
    const float rotation = gear > 0 ? M_PI : 0.0f;
    state_.pose.pose.position.x = 0.2923928677;
    state_.pose.pose.position.y = -0.0799737399;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
        tf2::Vector3(0, 0, 1), -0.5762864309 + rotation));
    state_.speed.linear.x = -gear * 0.0389661513;
    state_.speed.angular.z = -0.0782486952;
    setInitialSteering(gear * 0.4301109612f);
    setInitialConditionedSteering(gear * 0.4273190498f);
    path_.reset(6);
    path_.x = {0.29182267f, 0.25382188f, 0.21422242f, 0.17462295f, 0.13502349f, 0.09542403f};
    path_.y = {-0.08320937f, -0.05391501f, -0.02338826f, 0.00713849f, 0.03766524f, 0.06819199f};
    path_.yaws.fill(1.5f);  // Deliberately wrong preview; target must be real cusp.
    path_.yaws(0) = -0.61310279f + rotation;
    path_.yaws(1) = -0.70036972f + rotation + yaw_offset;
    if (closer_handoff) {
      // 180605 reaches the same shape only3.73cm from the cusp. A fixed
      // centimetre budget grid entered the gate then stopped0.6mm outside it.
      state_.pose.pose.position.x = 0.2874718063;
      state_.pose.pose.position.y = -0.0831910087;
      state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(
          tf2::Vector3(0, 0, 1), -0.5792950201 + rotation));
      state_.speed.linear.x = -gear * 0.0416766641;
      state_.speed.angular.z = -0.0692321209;
      setInitialSteering(gear * 0.4580996931f);
      setInitialConditionedSteering(gear * 0.4580996931f);
      path_.x(1) = 0.2564393878f; path_.y(1) = -0.0624313727f;
      path_.yaws(1) = -0.7039966583f + rotation;
    }
    setRealPathPointCount(real); setActiveCuspDirection(gear);
    goal_.position.x = 5.0; goal_.orientation.w = 1.0;
    critics_data_.motion_model = motion_model_;
    control_sequence_.vx.fill(0.0f); control_sequence_.wz.fill(0.0f);
    const bool found = initializeShortCuspMean();
    if (found) {
      for (size_t j = 0; j < settings_.time_steps; ++j) {
        EXPECT_GE(gear * control_sequence_.vx(j), 0.0f);
        EXPECT_LE(std::abs(control_sequence_.wz(j)),
          std::abs(control_sequence_.vx(j)) / 0.401f + 1e-6f);
      }
      // Forecast the selected mean using the same measured steering model.
      for (size_t row = 0; row < settings_.batch_size; ++row) {
        for (size_t j = 0; j < settings_.time_steps; ++j) {
          state_.cvx(row, j) = control_sequence_.vx(j);
          state_.cwz(row, j) = control_sequence_.wz(j);
        }
      }
      updateStateVelocities(state_); integrateStateVelocities(generated_trajectories_, state_);
      const size_t j = settings_.time_steps - 1;
      EXPECT_LE(std::hypot(generated_trajectories_.x(0, j) - path_.x(real - 1),
        generated_trajectories_.y(0, j) - path_.y(real - 1)), 0.02f);
      EXPECT_LE(std::abs(angles::shortest_angular_distance(
        generated_trajectories_.yaws(0, j), path_.yaws(real - 1))), 0.04f);
    } else {
      EXPECT_EQ(xt::sum(xt::fabs(control_sequence_.vx))(), 0.0f);
      EXPECT_EQ(xt::sum(xt::fabs(control_sequence_.wz))(), 0.0f);
    }
    EXPECT_FALSE(critics_data_.fail_flag);
    EXPECT_FALSE(critics_data_.furthest_reached_path_point.has_value());
    EXPECT_FALSE(critics_data_.path_pts_valid.has_value());
    EXPECT_EQ(xt::sum(costs_)(), 0.0f);
    return found;
  }

  xt::xtensor<float, 2> captureFinalCollisionForTest(
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> cm)
  {
    costmap_ros_ = cm; costmap_ = cm->getCostmap();
    settings_.time_steps = 4; settings_.model_dt = 0.05;
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
    state_.reset(2, 4); state_.pose.pose.position.x = 1.0;
    state_.pose.pose.position.y = 1.0; state_.pose.pose.orientation.w = 1.0;
    state_.pose.header.stamp.sec = 123;
    control_sequence_.reset(4); control_sequence_.vx.fill(0.1);
    generated_trajectories_.reset(2, 4); costs_ = xt::zeros<float>({2});
    const auto before = control_sequence_.vx;
    auto result = getOptimizedTrajectory();
    EXPECT_EQ(control_sequence_.vx, before);
    return result;
  }

  void checkCuspCriticMetadata(int gear)
  {
    setActiveCuspDirection(gear);
    EXPECT_EQ(critics_data_.active_cusp_direction, gear);
  }

  bool conditionForTest(bool enabled, int gear, size_t real, float a, float b,
    xt::xtensor<float, 1> & weights, float forward_cost = 0.0f, int cusp_gear = 0) {
    condition_control_on_path_direction_ = enabled;
    settings_.shift_control_sequence = true;settings_.temperature = 0.3f;
    path_.reset(2);path_.x = {0.0f, 0.1f * gear};path_.y.fill(0);path_.yaws.fill(0);
    critics_data_.real_path_point_count = real;state_.reset(2,2);
    setActiveCuspDirection(cusp_gear);
    state_.cvx(0,0) = -a;state_.cvx(1,0) = -b;
    state_.cvx(0,1) = a;state_.cvx(1,1) = b;
    costs_ = {forward_cost,0.0f};
    return conditionControlExponents(weights);
  }

  bool terminalConditionForTest(bool enabled, int gear, float along,
    xt::xtensor<float, 1> & weights)
  {
    condition_terminal_path_direction_ = enabled;
    setPathDirection(gear);
    state_.pose.pose.position.x = -along;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation.w = 1.0;
    return conditionForTest(true, 1, 1, 0.02f, -0.02f, weights);
  }

  bool filteredConditionForTest(int gear, xt::xtensor<float, 1> & weights) {
    condition_control_on_path_direction_ = true;
    smooth_control_sequence_ = true;
    settings_.shift_control_sequence = true;settings_.temperature = 0.3f;
    path_.reset(2);path_.x = {0.0f, 0.1f * gear};path_.y.fill(0);path_.yaws.fill(0);
    critics_data_.real_path_point_count = 2;state_.reset(2,41);
    for (auto & h : control_history_) {h = {0.0f,0.0f,0.0f};}
    xt::view(state_.cvx, 0, xt::all()).fill(-0.02f * gear);
    state_.cvx(0,1) = 0.0373f * gear;
    xt::view(state_.cvx, 1, xt::all()).fill(0.01f * gear);
    costs_ = {0.0f,0.0f};
    return conditionControlExponents(weights);
  }

  bool brakingConditionForTest(bool enabled, int gear, float measured,
    float candidate, xt::xtensor<float, 1> & weights)
  {
    condition_control_on_path_direction_ = true;
    allow_path_direction_braking_ = enabled;
    smooth_control_sequence_ = true;
    settings_.shift_control_sequence = true;settings_.temperature = 0.3f;
    path_.reset(2);path_.x = {0.0f, .1f * gear};path_.y.fill(0);path_.yaws.fill(0);
    critics_data_.real_path_point_count = 2;state_.reset(2,41);
    state_.speed.linear.x = measured;
    for (auto & h : control_history_) {h = {0.0f,0.0f,0.0f};}
    xt::view(state_.cvx, 0, xt::all()).fill(candidate);
    xt::view(state_.cvx, 1, xt::all()).fill(.01f * gear);
    costs_ = {0.0f,1.0f};
    return conditionControlExponents(weights);
  }

  bool carriedGearHistoryForTest(int gear, bool clear_history,
    xt::xtensor<float, 1> & weights)
  {
    condition_control_on_path_direction_ = true;
    smooth_control_sequence_ = true;
    settings_.shift_control_sequence = true;settings_.temperature = 0.3f;
    path_.reset(2);path_.x = {0.0f, 0.1f * gear};
    path_.y.fill(0);path_.yaws.fill(0);
    critics_data_.real_path_point_count = 2;state_.reset(2,41);
    for (auto & h : control_history_) {
      h = {clear_history ? 0.0f : -0.2f * gear, 0.0f, 0.0f};
    }
    state_.cvx.fill(0.01f * gear);
    costs_ = {0.0f,0.0f};
    return conditionControlExponents(weights);
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

  geometry_msgs::msg::TwistStamped recoverWrongWarmGear(float physical_speed)
  {
    control_sequence_.vx.fill(0.3f);
    control_sequence_.wz.fill(0.0f);
    for (auto & h : control_history_) {h = {-0.003f,0.0f,0.0f};}
    setRealPathPointCount(2);
    setActiveCuspDirection(-1);
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.orientation.w = 1.0;pose.pose.position.x = 1.25;
    geometry_msgs::msg::Twist speed;speed.linear.x = physical_speed;
    nav_msgs::msg::Path plan;plan.poses = {pose,pose};
    plan.poses[1].pose.position.x -= 0.1;
    auto command = evalControl(pose,speed,plan,plan.poses.back().pose,nullptr);
    EXPECT_DOUBLE_EQ(state_.pose.pose.position.x,1.25);
    EXPECT_DOUBLE_EQ(state_.speed.linear.x,physical_speed);
    EXPECT_NEAR(settings_.constraints.vx_max,0.1f,1e-6);
    return command;
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

  void seedAndSmoothAckermannCancellation()
  {
    // Each input pair is feasible at R=.401, but smoothing signed speeds
    // independently of yaw rate creates an infeasible near-zero-speed pair.
    control_sequence_.reset(41);
    for (size_t i = 0; i < 41; ++i) {
      control_sequence_.vx(i) = i % 2 ? -0.1f : 0.1f;
      control_sequence_.wz(i) = 0.2f;
    }
    for (size_t i = 0; i < control_history_.size(); ++i) {
      control_history_[i].vx = i % 2 ? -0.1f : 0.1f;
      control_history_[i].vy = 0.0f;
      control_history_[i].wz = 0.2f;
    }
    utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);
  }

  void setCommandOffset(bool shifted) {settings_.shift_control_sequence = shifted;}
  const models::Control & lastCommandHistory() const {return control_history_.back();}

  void testFreshRolloutsDiscardHistoricalCollisionCosts()
  {
    // Candidate indices are reused after the control distribution changes.
    // A prior collision at index 0 must not veto a new, cost-free rollout.
    settings_.gamma = 0.0f;
    settings_.iteration_count = 2;
    state_.pose.pose.orientation.w = 1.0;
    costs_.fill(0.0f);
    costs_(0) = 1.0e9f;
    optimize();  // Empty critic list: the current rollout costs are all zero.
    EXPECT_EQ(costs_, xt::zeros<float>({settings_.batch_size}));
  }

  void testFailedCriticBatchDoesNotUpdateDistribution()
  {
    state_.pose.pose.orientation.w = 1.0;
    settings_.iteration_count = 3;
    control_sequence_.vx.fill(0.15f);
    control_sequence_.wz.fill(0.02f);
    const auto before = control_sequence_;
    std::fill(costmap_->getCharMap(), costmap_->getCharMap() +
      costmap_->getSizeInCellsX() * costmap_->getSizeInCellsY(),
      nav2_costmap_2d::LETHAL_OBSTACLE);
    optimize();
    EXPECT_TRUE(critics_data_.fail_flag);
    // Every sample collided. No probability distribution may be fitted to it,
    // and subsequent iterations must not overwrite the failure with zero costs.
    EXPECT_EQ(control_sequence_.vx, before.vx);
    EXPECT_EQ(control_sequence_.wz, before.wz);
    EXPECT_GT(static_cast<float>(xt::amin(costs_)()), 0.0f);
  }

  void integrateStateVelocitiesWrapper(
    models::Trajectories & traj,
    const models::State & state)
  {
    return integrateStateVelocities(traj, state);
  }

  void testCalibratedTrajectoryDisplay()
  {
    state_.pose.pose.orientation.w = 1.0;
    control_sequence_.vx.fill(0.08f);
    control_sequence_.wz.fill(0.08f / 0.401f);
    auto trajectory = getOptimizedTrajectory();
    EXPECT_NEAR(trajectory(10, 2), 11.0f * settings_.model_dt *
      0.08f / (0.401f + 0.094f), 1e-6f);
    EXPECT_NEAR(control_sequence_.wz(10), 0.08f / 0.401f, 1e-6f);
  }

  void testSmoothingPolicy(bool enabled)
  {
    // Rounded recorded D20 sequence: forward now, reverse later. This checks
    // parameter ingestion and command extraction, not a synthetic mission pass.
    control_sequence_.vx.fill(-0.2f);
    control_sequence_.wz.fill(-0.18f);
    control_sequence_.vx(0) = 0.1162f;
    control_sequence_.vx(1) = 0.1086f;
    control_sequence_.vx(2) = -0.0720f;
    control_sequence_.vx(3) = -0.1962f;
    control_sequence_.vx(4) = -0.2124f;
    auto expected = control_sequence_;
    auto history = control_history_;
    if (enabled) {
      utils::savitskyGolayFilter(expected, history, settings_);
    }
    smoothControlSequence();
    EXPECT_EQ(control_sequence_.vx, expected.vx);
    EXPECT_EQ(control_sequence_.wz, expected.wz);
    auto command = getControlFromSequenceAsTwistWrapper();
    EXPECT_LE(std::fabs(command.twist.angular.z),
      std::fabs(command.twist.linear.x) / 0.401f + 1e-6f);
    EXPECT_NEAR(lastCommandHistory().vx, command.twist.linear.x, 1e-6f);
    if (!enabled) {
      EXPECT_NEAR(command.twist.linear.x, 0.1086f, 1e-6f);
    }
  }

  void testMotorDeadbandFloor()
  {
    motor_deadband_floor_ = 0.050f;
    motor_deadband_approach_radius_ = 0.15f;
    motor_deadband_terminal_radius_ = 0.04f;

    // 1a. Intermediate forward cusp within approach radius (10 cm away) -> floored forward
    setActiveCuspDirection(1);
    path_.reset(2);
    path_.x = {0.0f, 0.10f};
    path_.y.fill(0.0f);
    path_.yaws.fill(0.0f);
    setRealPathPointCount(2);
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion::getIdentity());
    control_sequence_.vx.fill(0.010f);
    control_sequence_.wz.fill(0.0f);
    auto cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_GE(cmd.twist.linear.x, 0.050f);

    // 1b. Intermediate reverse cusp within approach radius (10 cm away) -> floored reverse
    setActiveCuspDirection(-1);
    path_.reset(2);
    path_.x = {0.0f, -0.10f};
    path_.y.fill(0.0f);
    path_.yaws.fill(0.0f);
    setRealPathPointCount(2);
    control_sequence_.vx.fill(-0.010f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_LE(cmd.twist.linear.x, -0.050f);

    // 1c. Intermediate cusp inside terminal radius (2 cm away <= 4 cm radius) -> floor released
    setActiveCuspDirection(-1);
    path_.reset(2);
    path_.x = {0.0f, -0.02f};
    path_.y.fill(0.0f);
    path_.yaws.fill(0.0f);
    setRealPathPointCount(2);
    control_sequence_.vx.fill(-0.010f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, -0.010f, 1e-4);

    // 1d. Intermediate cusp outside approach radius (30 cm away > 15 cm radius) -> floor NOT applied
    setActiveCuspDirection(1);
    path_.reset(2);
    path_.x = {0.0f, 0.30f};
    path_.y.fill(0.0f);
    path_.yaws.fill(0.0f);
    setRealPathPointCount(2);
    control_sequence_.vx.fill(0.010f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, 0.010f, 1e-4);

    // 2. Final segment (active_cusp_direction_ == 0) within approach radius (10 cm away) floored forward
    active_cusp_direction_ = 0;
    path_direction_ = 1;
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    goal_.position.x = 0.10;
    goal_.position.y = 0.0;
    control_sequence_.vx.fill(0.015f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_GE(cmd.twist.linear.x, 0.050f);

    // 3. Final segment reverse within approach radius floored reverse
    path_direction_ = -1;
    control_sequence_.vx.fill(-0.015f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_LE(cmd.twist.linear.x, -0.050f);

    // 4. Final segment inside terminal radius (2 cm away <= 4 cm radius) -> floor released to decelerate/stop
    path_direction_ = 1;
    state_.pose.pose.position.x = 0.08;
    state_.pose.pose.position.y = 0.0;
    goal_.position.x = 0.10;
    control_sequence_.vx.fill(0.015f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, 0.015f, 1e-4);

    // 5. Final segment outside approach radius (30 cm away > 15 cm radius) -> floor NOT applied (normal MPPI)
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    goal_.position.x = 0.30;
    control_sequence_.vx.fill(0.020f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, 0.020f, 1e-4);

    // 6. Final segment within approach radius with negative noise -> directional floor forces positive approach (no chatter)
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    goal_.position.x = 0.10; // 10 cm <= 15 cm approach radius
    control_sequence_.vx.fill(-0.005f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_GE(cmd.twist.linear.x, 0.050f);

    // 7. Final segment within approach radius, but heading misaligned to path (> 30 deg) -> floor NOT applied (let MPPI steer)
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    path_.reset(2);
    path_.x = {0.0f, 0.10f};
    path_.y.fill(0.0f);
    path_.yaws.fill(0.0f);
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, 1.0); // ~57 deg misaligned from path heading (0.0)
    state_.pose.pose.orientation = tf2::toMsg(q);
    goal_.position.x = 0.10;
    goal_.position.y = 0.0;
    control_sequence_.vx.fill(0.015f);
    control_sequence_.wz.fill(0.0f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, 0.015f, 1e-4);

    // 8. Final segment with reverse path_direction_ (-1) but goal is in front and MPPI commands forward creep (+0.020f)
    // -> effective_direction aligns forward, floored to +0.050f (never inverted to -0.050f)
    path_direction_ = -1;
    state_.pose.pose.position.x = 0.0;
    state_.pose.pose.position.y = 0.0;
    state_.pose.pose.orientation = geometry_msgs::msg::Quaternion(); // yaw = 0.0
    path_.yaws.fill(0.0f);
    goal_.position.x = 0.10; // in front
    goal_.position.y = 0.0;
    control_sequence_.vx.fill(0.020f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_GE(cmd.twist.linear.x, 0.050f);

    // 9. Final segment with reverse path_direction_ (-1) and MPPI commands deliberate forward maneuver (+0.180f)
    // -> preserved without modification (never inverted to -0.050f)
    control_sequence_.vx.fill(0.180f);
    cmd = getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(cmd.twist.linear.x, 0.180f, 1e-4);
  }
};

TEST(OptimizerTests, ExitCuspContinuationPreservesHorizonAndNeverRestartsDelay)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("exit_completion");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  node->declare_parameter("mppic.batch_size", 8);
  node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("mppic.AckermannConstraints.response_radius_offset", 0.093);
  node->declare_parameter("mppic.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("mppic.AckermannConstraints.steering_actuator_rate", 1.0);
  node->declare_parameter("mppic.critics", std::vector<std::string>{"PathFollowCritic"});
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>("exit_completion_map", "", "exit_completion_map", true);
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.025, -1.0, -1.0);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.testExitCompletionContinuation();
  optimizer.shutdown();
}

TEST(OptimizerTests, ControlSmoothingPolicyKeepsLegacyDefaultAndPhysicalBounds)
{
  for (bool enabled : {true, false}) {
    auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("smoothing_policy");
    node->declare_parameter("controller_frequency", 20.0);
    node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
    node->declare_parameter("mppic.AckermannConstraints.min_turning_r", 0.401);
    if (!enabled) {node->declare_parameter("mppic.smooth_control_sequence", false);}
    auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
      "smoothing_costmap", "", "smoothing_costmap", true);
    ParametersHandler parameters(node);
    rclcpp_lifecycle::State lifecycle;
    costmap->on_configure(lifecycle);
    OptimizerTester optimizer;
    optimizer.initialize(node, "mppic", costmap, &parameters);
    optimizer.testSmoothingPolicy(enabled);
    optimizer.shutdown();
  }
}

TEST(OptimizerTests, CalibratedTrajectoryDisplayPreservesCommands)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("calibrated_display");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.response_radius_offset", 0.094);
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "calibrated_costmap", "", "calibrated_costmap", true);
  ParametersHandler parameters(node);
  rclcpp_lifecycle::State lifecycle;
  costmap->on_configure(lifecycle);
  OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", costmap, &parameters);
  optimizer.testCalibratedTrajectoryDisplay();
  optimizer.shutdown();
}

TEST(OptimizerTests, OppositeFeedbackConditionsRequestsWithoutTeleportingVelocity)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("opposite_feedback_gear");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  node->declare_parameter("mppic.batch_size", 8);
  node->declare_parameter("mppic.motion_model", "Ackermann");
  node->declare_parameter("mppic.critics", std::vector<std::string>{"PathFollowCritic"});
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "opposite_feedback_map", "", "opposite_feedback_map", true);
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.testOppositeFeedbackDirectionRequests();
  optimizer.shutdown();
}

TEST(OptimizerTests, ShortCuspMeanUsesPhysicalPredictionAndPreservesNegativeControls)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("short_cusp_seed");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 81);
  node->declare_parameter("mppic.batch_size", 64);
  node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("mppic.AckermannConstraints.response_radius_offset", 0.093);
  node->declare_parameter("mppic.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("mppic.AckermannConstraints.steering_actuator_rate", 1.0);
  node->declare_parameter("mppic.inversion_xy_tolerance", 0.02);
  node->declare_parameter("mppic.inversion_yaw_tolerance", 0.04);
  node->declare_parameter("mppic.critics", std::vector<std::string>{"PathAlignCritic"});
  node->declare_parameter("mppic.PathAlignCritic.cusp_terminal_heading_weight", 30.0);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "short_cusp_seed_map", "", "short_cusp_seed_map", true);
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  EXPECT_FALSE(optimizer.recordedShortCuspSeed(false, -1));
  EXPECT_TRUE(optimizer.recordedShortCuspSeed(true, -1));
  EXPECT_TRUE(optimizer.recordedShortCuspSeed(true, 1));
  EXPECT_TRUE(optimizer.recordedShortCuspSeed(true, -1, 2, 0.0f, true));
  EXPECT_FALSE(optimizer.recordedShortCuspSeed(true, 0));
  EXPECT_FALSE(optimizer.recordedShortCuspSeed(true, -1, 5));
  EXPECT_FALSE(optimizer.recordedShortCuspSeed(true, -1, 2, 1.0f));
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, nav2_costmap_2d::LETHAL_OBSTACLE);
  EXPECT_FALSE(optimizer.recordedShortCuspSeed(true, -1));
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  optimizer.testRecordedEmittedMeanGuard();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  optimizer.testRecordedMixedGearMean();
  optimizer.testRecordedWarmStartDelay();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  optimizer.testHiddenFutureGearMean();
  optimizer.testLongPrefixGearMean();
  optimizer.testShortCuspDirectionScope();
  optimizer.testLongEntryRetention();
  optimizer.testLongEntryShiftRepair();
  optimizer.testSparseOppositeSteeringEntry();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  optimizer.testSampleFailurePreservesWarmStart();
  optimizer.shutdown();
}

TEST(OptimizerTests, RecordedEmittedMeanMustHonorEffectiveCuspGate)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("effective_gate");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  node->declare_parameter("mppic.batch_size", 1);
  node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("mppic.AckermannConstraints.response_radius_offset", 0.093);
  node->declare_parameter("mppic.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("mppic.AckermannConstraints.steering_actuator_rate", 1.0);
  node->declare_parameter("mppic.inversion_xy_tolerance", 0.05);
  node->declare_parameter("mppic.inversion_yaw_tolerance", 0.10);
  node->declare_parameter("mppic.critics", std::vector<std::string>{});
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>("effective_gate_map", "", "effective_gate_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.testRecordedEffectiveCuspGate();
  optimizer.shutdown();
  cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, FreshRolloutsDiscardHistoricalCollisionCosts)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("fresh_rollout_costs");
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(20.0));
  node->declare_parameter("mppic.model_dt", rclcpp::ParameterValue(0.05));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "fresh_costmap", "", "fresh_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", costmap_ros, &param_handler);
  optimizer.testFreshRolloutsDiscardHistoricalCollisionCosts();
  optimizer.shutdown();
}

TEST(OptimizerTests, FailedCriticBatchDoesNotUpdateDistribution)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("failed_batch_update");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.batch_size", 16);
  node->declare_parameter("mppic.time_steps", 41);
  node->declare_parameter("mppic.critics", std::vector<std::string>{"ObstaclesCritic"});
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "failed_batch_costmap", "", "failed_batch_costmap", true);
  ParametersHandler handler(node);
  rclcpp_lifecycle::State state;
  cm->on_configure(state);
  OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &handler);
  optimizer.testFailedCriticBatchDoesNotUpdateDistribution();
  optimizer.shutdown();
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

TEST(OptimizerTests, AckermannCommandRemainsFeasibleAfterSmoothing)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("command_cone_test");
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(20.0));
  node->declare_parameter("mppic.model_dt", rclcpp::ParameterValue(0.05));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(41));
  node->declare_parameter("mppic.vx_max", rclcpp::ParameterValue(0.5));
  node->declare_parameter("mppic.vx_min", rclcpp::ParameterValue(-0.3));
  node->declare_parameter("mppic.motion_model", rclcpp::ParameterValue("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.min_turning_r", rclcpp::ParameterValue(0.401));
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler parameters(node);
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);
  OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", costmap, &parameters);

  for (bool shifted : {false, true}) {
    optimizer.setCommandOffset(shifted);
    optimizer.seedAndSmoothAckermannCancellation();
    auto & sequence = optimizer.grabControlSequence();
    const size_t index = shifted ? 1 : 0;
    const float speed = sequence.vx(index);
    // Confirm the filter actually generated the defect being tested.
    ASSERT_GT(std::abs(sequence.wz(index)), std::abs(speed) / 0.401f);
    auto command = optimizer.getControlFromSequenceAsTwistWrapper();
    EXPECT_NEAR(command.twist.linear.x, speed, 1e-7);
    EXPECT_LE(std::abs(command.twist.angular.z), std::abs(speed) / 0.401 + 1e-6);
    EXPECT_GT(command.twist.angular.z, 0.0);
    EXPECT_NEAR(optimizer.lastCommandHistory().vx, command.twist.linear.x, 1e-7);
    EXPECT_NEAR(optimizer.lastCommandHistory().wz, command.twist.angular.z, 1e-7);

    // Actual terminal observations, forward/reverse, stopped and feasible cases.
    for (float vx : {0.0f, 0.000195f, -0.000195f, 0.1f, -0.1f}) {
      for (float wz : {-0.00692f, 0.0f, 0.00692f}) {
        sequence.vx.fill(vx);
        sequence.wz.fill(wz);
        command = optimizer.getControlFromSequenceAsTwistWrapper();
        EXPECT_NEAR(command.twist.linear.x, vx, 1e-7);
        EXPECT_LE(std::abs(command.twist.angular.z), std::abs(vx) / 0.401 + 1e-6);
        if (std::abs(wz) <= std::abs(vx) / 0.401f) {
          EXPECT_NEAR(command.twist.angular.z, wz, 1e-7);
        }
      }
    }
    for (float vx : {-0.33f, 0.52f}) {
      sequence.vx.fill(vx);
      sequence.wz.fill(0.0f);
      command = optimizer.getControlFromSequenceAsTwistWrapper();
      EXPECT_NEAR(command.twist.linear.x, vx < 0 ? -0.3 : 0.5, 1e-6);
    }
  }
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

TEST(OptimizerTests, DirectionConditioningIsOptInAndUsesCommandOffset) {
 OptimizerTester o;xt::xtensor<float,1> w={0.2f,0.8f};
 EXPECT_FALSE(o.conditionForTest(false,1,2,0.02f,-0.02f,w));
 EXPECT_FLOAT_EQ(w(0),0.2f);EXPECT_FLOAT_EQ(w(1),0.8f);
 EXPECT_TRUE(o.conditionForTest(true,1,2,0.02f,-0.02f,w));
 EXPECT_FLOAT_EQ(w(0),1.0f);EXPECT_FLOAT_EQ(w(1),0.0f);
 EXPECT_TRUE(o.conditionForTest(true,-1,2,0.02f,-0.02f,w));
 EXPECT_FLOAT_EQ(w(0),0.0f);EXPECT_FLOAT_EQ(w(1),1.0f);
}
TEST(OptimizerTests, DirectionConditioningIgnoresGhostsAndFailsWithoutSamples) {
 OptimizerTester o;xt::xtensor<float,1> w={0.2f,0.8f};
 EXPECT_FALSE(o.conditionForTest(true,1,1,0.02f,-0.02f,w));
 EXPECT_FLOAT_EQ(w(0),0.2f);EXPECT_FLOAT_EQ(w(1),0.8f);
 EXPECT_THROW(o.conditionForTest(true,1,2,-0.02f,-0.01f,w),nav2_core::NoValidControl);
}
TEST(OptimizerTests, DirectionConditioningRenormalizesUnderflowedEligibleWeights) {
 OptimizerTester o;xt::xtensor<float,1> w={0.0f,1.0f};
 EXPECT_TRUE(o.conditionForTest(true,1,2,0.02f,-0.02f,w,10000.0f));
 EXPECT_FLOAT_EQ(w(0),1.0f);EXPECT_FLOAT_EQ(w(1),0.0f);
}

TEST(OptimizerTests, DirectionConditioningRejectsNonFiniteEligibleCostWithFastMath) {
 OptimizerTester o;xt::xtensor<float,1> w={0.0f,1.0f};
 uint32_t bits=0x7fc00000u;float nan;std::memcpy(&nan,&bits,sizeof(nan));
 EXPECT_THROW(o.conditionForTest(true,1,2,0.02f,-0.02f,w,nan),nav2_core::NoValidControl);
}

TEST(OptimizerTests, DirectionBrakingIsOptInAndRequiresMeasuredDeceleration) {
  for (int gear : {-1, 1}) {
    OptimizerTester o;xt::xtensor<float,1> w={.5f,.5f};
    EXPECT_TRUE(o.brakingConditionForTest(false,gear,-.06f*gear,-.01f*gear,w));
    EXPECT_FLOAT_EQ(w(0),0.0f);
    EXPECT_TRUE(o.brakingConditionForTest(true,gear,-.06f*gear,-.01f*gear,w));
    EXPECT_GT(w(0),.9f);
    for (float measured : {0.0f,.06f*gear}) {
      EXPECT_TRUE(o.brakingConditionForTest(true,gear,measured,-.01f*gear,w));
      EXPECT_FLOAT_EQ(w(0),0.0f);
    }
    EXPECT_TRUE(o.brakingConditionForTest(true,gear,-.06f*gear,-.12f*gear,w));
    EXPECT_FLOAT_EQ(w(0),0.0f);
  }
}

TEST(OptimizerTests, DirectionConditioningRejectsTemporalFilterCancellation) {
 OptimizerTester o;xt::xtensor<float,1> w={0.5f,0.5f};
 EXPECT_TRUE(o.filteredConditionForTest(1,w));
 EXPECT_FLOAT_EQ(w(0),0.0f);EXPECT_FLOAT_EQ(w(1),1.0f);
 EXPECT_TRUE(o.filteredConditionForTest(-1,w));
 EXPECT_FLOAT_EQ(w(0),0.0f);EXPECT_FLOAT_EQ(w(1),1.0f);
}

TEST(OptimizerTests, OldGearHistoryCanRejectEveryOtherwiseCorrectNewGearSample) {
  // A sufficient synthetic mechanism, not a reconstruction of live history.
  // Three past samples contribute 107/231 of the shifted SG stencil: old
  // -0.2 m/s commands outweigh new +0.01 m/s controls in either gear.
  OptimizerTester o;xt::xtensor<float,1> w={0.5f,0.5f};
  for (int gear : {-1,1}) {
    EXPECT_THROW(o.carriedGearHistoryForTest(gear,false,w),nav2_core::NoValidControl);
    EXPECT_TRUE(o.carriedGearHistoryForTest(gear,true,w));
    EXPECT_FLOAT_EQ(w(0),1.0f);EXPECT_FLOAT_EQ(w(1),1.0f);
  }
}

TEST(OptimizerTests, PrunedActiveCuspRetainsGearButFinalSingletonDoesNot) {
  OptimizerTester o;xt::xtensor<float,1> w={0.5f,0.5f};
  // Recorded failures leave a real singleton plus ghosts. Opposite commands
  // cancel unless the original active cusp's gear survives pruning. Ghost
  // geometry deliberately points forward while the real segment is reverse.
  EXPECT_TRUE(o.conditionForTest(true,1,1,0.02f,-0.02f,w,0.0f,-1));
  EXPECT_FLOAT_EQ(w(0),0.0f);EXPECT_FLOAT_EQ(w(1),1.0f);
  w={0.5f,0.5f};
  EXPECT_FALSE(o.conditionForTest(true,1,1,0.02f,-0.02f,w));
  EXPECT_FLOAT_EQ(w(0),0.5f);EXPECT_FLOAT_EQ(w(1),0.5f);
}

TEST(OptimizerTests, FinalSingletonGearIsOptInAndReleasesAfterOvershoot)
{
  OptimizerTester o;
  for (int gear : {-1, 1}) {
    xt::xtensor<float, 1> w = {0.5f, 0.5f};
    EXPECT_FALSE(o.terminalConditionForTest(false, gear, .06f * gear, w));
    EXPECT_FLOAT_EQ(w(0), .5f); EXPECT_FLOAT_EQ(w(1), .5f);
    EXPECT_TRUE(o.terminalConditionForTest(true, gear, .06f * gear, w));
    EXPECT_FLOAT_EQ(w(0), gear == 1 ? 1.0f : 0.0f);
    EXPECT_FLOAT_EQ(w(1), gear == -1 ? 1.0f : 0.0f);
    for (float along : {-.06f * gear, 0.0f, .004f * gear}) {
      w = {.5f, .5f};
      EXPECT_FALSE(o.terminalConditionForTest(true, gear, along, w));
      EXPECT_FLOAT_EQ(w(0), .5f); EXPECT_FLOAT_EQ(w(1), .5f);
    }
  }
  xt::xtensor<float, 1> w = {.5f, .5f};
  EXPECT_FALSE(o.terminalConditionForTest(true, 0, .06f, w));
}

TEST(OptimizerTests, WrongWarmGearRetriesWithinExistingBudgetWithoutResettingPlant)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("gear_retry_test");
  node->declare_parameter("controller_frequency",20.0);
  node->declare_parameter("mppic.model_dt",0.05);
  node->declare_parameter("mppic.time_steps",41);
  node->declare_parameter("mppic.vx_std",0.01);
  node->declare_parameter("mppic.gamma",0.0);
  node->declare_parameter("mppic.condition_control_on_path_direction",true);
  node->declare_parameter("mppic.retry_attempt_limit",1);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>("retry_costmap","","retry_costmap",true);
  ParametersHandler parameters(node);rclcpp_lifecycle::State state;cm->on_configure(state);
  OptimizerTester optimizer;optimizer.initialize(node,"mppic",cm,&parameters);
  optimizer.setSpeedLimit(0.1,false);
  EXPECT_NO_THROW({
    auto cmd = optimizer.recoverWrongWarmGear(0.0f);
    EXPECT_LT(cmd.twist.linear.x,0.0);
    EXPECT_LE(std::abs(cmd.twist.linear.x),0.1);
  });
  // Forward physical velocity cannot reverse within one acceleration-limited
  // step. Even a fresh batch must fail; never reset the measured plant speed.
  EXPECT_THROW(optimizer.recoverWrongWarmGear(0.5f),nav2_core::NoValidControl);
}

TEST(OptimizerTests, ActiveCuspMetadataReachesCriticsAndClearsOnFinalSegment)
{
  OptimizerTester optimizer;
  for (int gear : {0, -1, 1, 0}) {optimizer.checkCuspCriticMetadata(gear);}
}

TEST(OptimizerTests, ExactFinalCollisionCaptureIsOptionalAndDoesNotAlterCommands)
{
  const char * output = std::getenv("MPPI_FINAL_COLLISION_SNAPSHOT_PATH");
  if (!output) {GTEST_SKIP() << "Run this case in an isolated process with capture enabled";}
  ASSERT_FALSE(std::filesystem::exists(output));
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "final_capture_map", "", "final_capture_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.050,-0.151],[0.276,-0.151],[0.276,0.151],[-0.050,0.151]]")));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  OptimizerTester optimizer;
  optimizer.captureFinalCollisionForTest(cm);
  EXPECT_FALSE(std::filesystem::exists(output));
  unsigned int mx, my;
  for (double y = 0.5; y < 1.5; y += 0.025) {
    ASSERT_TRUE(grid->worldToMap(1.275, y, mx, my));
    grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
  }
  const auto trajectory = optimizer.captureFinalCollisionForTest(cm);
  ASSERT_TRUE(std::filesystem::exists(output));
  const auto read = [&]() {
      std::ifstream file(output);
      std::string text; std::getline(file, text); return text;
    };
  const auto first = read();
  EXPECT_NE(first.find("\"stamp_ns\":123000000000"), std::string::npos);
  EXPECT_NE(first.find("\"collision_index\":0"), std::string::npos);
  const auto second_trajectory = optimizer.captureFinalCollisionForTest(cm);
  EXPECT_EQ(trajectory, second_trajectory);
  EXPECT_EQ(first, read());
  cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, FilterRetentionChecksBothMeansAndPreservesFreeSpaceFiltering)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("filter_retention");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "filter_retention_map", "", "filter_retention_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.050,-0.151],[0.276,-0.151],[0.276,0.151],[-0.050,0.151]]")));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.testFilteredCollisionRetention();
  optimizer.shutdown(); cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, LongPathSampleRetentionPreservesDirectionAndShortEndpointFailure)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("long_sample_retention");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "long_sample_map", "", "long_sample_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.050,-0.151],[0.276,-0.151],[0.276,0.151],[-0.050,0.151]]")));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  std::fill(cm->getCostmap()->getCharMap(), cm->getCostmap()->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.testLongPathSampleRetention();
  optimizer.shutdown(); cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, FailedBatchCaptureOccursBeforeControlUpdateAndPreservesFailure)
{
  const char * output = std::getenv("MPPI_STALL_SNAPSHOT_PATH");
  const char * enabled = std::getenv("MPPI_SNAPSHOT_ON_FIRST_FAILED_BATCH");
  if (!output || !enabled || std::string(enabled) != "1") {
    GTEST_SKIP() << "Run in an isolated process with failed-batch capture enabled";
  }
  const std::string sidecar = std::string(output) + ".failed_batch.json";
  ASSERT_FALSE(std::filesystem::exists(sidecar));
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("failed_batch_capture");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.batch_size", 16);
  node->declare_parameter("mppic.time_steps", 21);
  node->declare_parameter("mppic.regenerate_noises", false);
  node->declare_parameter("mppic.retry_attempt_limit", 1);
  node->declare_parameter("mppic.critics", std::vector<std::string>{"ObstaclesCritic"});
  auto cm = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "failed_batch_capture_map", "", "failed_batch_capture_map", true);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  cm->getLayeredCostmap()->resizeMap(80, 80, 0.05, -1.0, -1.0);
  auto * grid = cm->getCostmap();
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "odom"; pose.header.stamp.sec = 123;
  pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist speed;
  nav_msgs::msg::Path plan;
  plan.header = pose.header;
  for (size_t k = 0; k < 3; ++k) {
    auto point = pose; point.pose.position.x = 0.1 * k; plan.poses.push_back(point);
  }
  const auto goal = plan.poses.back().pose;
  EXPECT_NO_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr));
  EXPECT_FALSE(std::filesystem::exists(sidecar));
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, nav2_costmap_2d::LETHAL_OBSTACLE);
  const char * after_text = std::getenv("MPPI_FAILED_BATCH_AFTER_STAMP");
  if (after_text) {
    const auto after = mppi::diagnostic::minimumCaptureStamp(after_text);
    ASSERT_TRUE(after && std::isfinite(*after) && *after > 123.0);
    EXPECT_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr), nav2_core::NoValidControl);
    EXPECT_FALSE(std::filesystem::exists(sidecar));
    pose.header.stamp.sec = static_cast<int32_t>(std::ceil(*after)) + 1;
  }
  const char * stationary_only = std::getenv("MPPI_FAILED_BATCH_STATIONARY_ONLY");
  if (stationary_only && std::string(stationary_only) == "1") {
    // Filtered failures must still propagate, without consuming the one-shot.
    speed.linear.x = 0.1;
    EXPECT_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr), nav2_core::NoValidControl);
    EXPECT_FALSE(std::filesystem::exists(sidecar));
    speed.linear.x = 0.0;
    speed.angular.z = 0.1;
    EXPECT_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr), nav2_core::NoValidControl);
    EXPECT_FALSE(std::filesystem::exists(sidecar));
    speed.angular.z = 0.0;
  }
  EXPECT_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr), nav2_core::NoValidControl);
  ASSERT_TRUE(std::filesystem::exists(sidecar));
  std::ifstream file(sidecar);
  std::string text; std::getline(file, text);
  EXPECT_NE(text.find("\"capture_reason\":\"first_failed_critic_batch\""), std::string::npos);
  EXPECT_NE(text.find("\"fail_flag\":true"), std::string::npos);
  EXPECT_NE(text.find("\"unbounded_available\":true"), std::string::npos);
  EXPECT_NE(text.find("\"control_update_completed\":false"), std::string::npos);
  EXPECT_EQ(text.find("\"control_vx_after\""), std::string::npos);
  const char * recent_option = std::getenv("MPPI_FAILED_BATCH_RECENT_RING");
  if (recent_option && std::string(recent_option) == "1") {
    const int32_t first_stamp = pose.header.stamp.sec;
    for (int32_t i = 1; i <= 20; ++i) {
      pose.header.stamp.sec = first_stamp + i;
      EXPECT_THROW(optimizer.evalControl(pose, speed, plan, goal, nullptr),
        nav2_core::NoValidControl);
    }
    std::ifstream original(sidecar);
    std::string unchanged; std::getline(original, unchanged);
    EXPECT_EQ(unchanged, text);
    for (unsigned int slot = 0; slot < 8; ++slot) {
      const auto recent = std::string(output) + ".failed_batch_recent_" +
        std::to_string(slot) + ".json";
      ASSERT_TRUE(std::filesystem::exists(recent));
      std::ifstream input(recent);
      std::string recorded; std::getline(input, recorded);
      EXPECT_NE(recorded.find("\"capture_reason\":\"recent_failed_critic_batch\""),
        std::string::npos);
      EXPECT_NE(recorded.find("\"control_update_completed\":false"), std::string::npos);
    }
    const auto latest = std::string(output) + ".failed_batch_recent_4.json";
    std::ifstream input(latest);
    std::string recorded; std::getline(input, recorded);
    EXPECT_NE(recorded.find("\"stamp\":" + std::to_string(first_stamp + 20) + ","),
      std::string::npos);
  }
  const char * successful_after = std::getenv("MPPI_SUCCESSFUL_BATCH_AFTER_STAMP");
  if (successful_after) {
    const auto minimum = mppi::diagnostic::minimumCaptureStamp(successful_after);
    ASSERT_TRUE(minimum);
    const std::string successful = std::string(output) + ".successful_batch.json";
    ASSERT_FALSE(std::filesystem::exists(successful));
    std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
    plan.poses.clear();
    for (size_t k = 0; k < 30; ++k) {
      auto point = pose; point.pose.position.x = 0.03 * k; plan.poses.push_back(point);
    }
    optimizer.setRealPathPointCount(30);
    optimizer.setActiveCuspDirection(1);
    pose.header.stamp.sec = static_cast<int32_t>(std::ceil(*minimum)) + 2;
    EXPECT_NO_THROW(optimizer.evalControl(pose, speed, plan, plan.poses.back().pose, nullptr));
    ASSERT_TRUE(std::filesystem::exists(successful));
    std::ifstream success_file(successful);
    std::string success_text; std::getline(success_file, success_text);
    EXPECT_NE(success_text.find("targeted_successful_critic_batch"), std::string::npos);
    EXPECT_NE(success_text.find("\"fail_flag\":false"), std::string::npos);
    EXPECT_NE(success_text.find("\"control_vx_after\""), std::string::npos);
    pose.header.stamp.sec += 1;
    EXPECT_NO_THROW(optimizer.evalControl(pose, speed, plan, plan.poses.back().pose, nullptr));
    std::ifstream repeated(successful);
    std::string repeated_text; std::getline(repeated, repeated_text);
    EXPECT_EQ(repeated_text, success_text);
    std::ifstream preserved(sidecar);
    std::string failed_text; std::getline(preserved, failed_text);
    EXPECT_EQ(failed_text, text);
  }
  optimizer.shutdown();
  cm->on_cleanup(lifecycle);
}


TEST(OptimizerTests, DepartureRejectionCapturesActualLockedInputsWithoutChangingPolicy)
{
  const auto file = std::filesystem::temp_directory_path() / "office_departure_rejection_test.jsonl";
  for (int k = 0; k < 8; ++k) {std::filesystem::remove(file.string() + "." + std::to_string(k) + ".json");}
  const std::string path = file.string();
  unsetenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH");
  unsetenv("MPPI_CUSP_MEAN_TRACE_PATH");
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("departure_capture");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("mppic.model_dt", 0.05);
  node->declare_parameter("mppic.time_steps", 41);
  node->declare_parameter("mppic.batch_size", 8);
  node->declare_parameter("mppic.motion_model", std::string("Ackermann"));
  node->declare_parameter("mppic.AckermannConstraints.min_turning_r", 0.401);
  node->declare_parameter("mppic.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("mppic.AckermannConstraints.steering_actuator_rate", 1.0);
  auto cm = std::make_shared<SettledTestCostmap>("departure_capture_map", "", "departure_capture_map", false);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string("[[-0.05,-0.151],[0.276,-0.151],[0.276,0.151],[-0.05,0.151]]")));
  rclcpp_lifecycle::State lifecycle; cm->on_configure(lifecycle);
  auto * grid = cm->getCostmap(); grid->resizeMap(80, 80, 0.05, -1.0, -1.0);
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  ParametersHandler parameters(node); OptimizerTester optimizer;
  optimizer.initialize(node, "mppic", cm, &parameters);
  optimizer.setInitialSteering(0.0f); optimizer.setInitialConditionedSteering(0.0f);
  geometry_msgs::msg::PoseStamped pose; pose.header.frame_id = "odom";
  pose.header.stamp.sec = 2136; pose.header.stamp.nanosec = 500000000;
  pose.pose.position.x = 0.3; pose.pose.position.y = 0.4; pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist speed; speed.linear.x = -0.02;
  nav_msgs::msg::Path outgoing; outgoing.header = pose.header;
  for (double dx : {0.0, 0.1, 0.2}) {
    auto target = pose; target.pose.position.x += dx; outgoing.poses.push_back(target);
  }
  const auto before = optimizer.departureControlsForTest();
  // Default disabled: a rejected footprint still rejects and creates no output.
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 254);
  EXPECT_FALSE(optimizer.longCuspDepartureAvailable(pose, speed, outgoing, 0.05, 0.1, true));
  EXPECT_FALSE(std::filesystem::exists(path + ".0.json"));
  // Failed I/O must not escape through the controller or alter its decision.
  setenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH", "/nonexistent_office_capture_parent/out", 1);
  auto invalid_stamp = pose; invalid_stamp.header.stamp.sec = -1;
  EXPECT_NO_THROW(EXPECT_FALSE(optimizer.longCuspDepartureAvailable(
    invalid_stamp, speed, outgoing, 0.05, 0.1, true)));
  EXPECT_FALSE(optimizer.longCuspDepartureAvailable(pose, speed, outgoing, 0.05, 0.1, true));
  pose.header.stamp.sec += 1;
  // Mean-path fallback is the actual existing diagnostic opt-in in Compose.
  unsetenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH");
  setenv("MPPI_CUSP_MEAN_TRACE_PATH", path.c_str(), 1);
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 0);
  EXPECT_TRUE(optimizer.longCuspDepartureAvailable(pose, speed, outgoing, 0.05, 0.1, true));
  auto impossible = outgoing;
  impossible.poses.back().pose.position.x += 10.0;
  EXPECT_FALSE(optimizer.longCuspDepartureAvailable(pose, speed, impossible, 0.05, 0.1, true));
  const auto fallback = path + ".departure_rejection.1.json";
  ASSERT_TRUE(std::filesystem::exists(fallback));
  {
    std::ifstream stream(fallback); nlohmann::json row; stream >> row;
    EXPECT_EQ(row.at("reason"), "candidate_family_exhausted");
    EXPECT_EQ(row.at("initial_footprint_cost"), 0.0);
    EXPECT_DOUBLE_EQ(row.at("departure_prefix").back()[0], 10.5);
    EXPECT_DOUBLE_EQ(row.at("initial_steering"), 0.0);
    EXPECT_DOUBLE_EQ(row.at("initial_conditioned_steering"), 0.0);
  }
  std::filesystem::remove(fallback);
  unsetenv("MPPI_CUSP_MEAN_TRACE_PATH");
  setenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH", path.c_str(), 1);
  // Oversized diagnostics are skipped, without overriding collision rejection.
  pose.header.stamp.sec += 1;
  grid->resizeMap(501, 501, 0.05, -1.0, -1.0);
  std::fill(grid->getCharMap(), grid->getCharMap() + 251001, 254);
  EXPECT_FALSE(optimizer.longCuspDepartureAvailable(pose, speed, outgoing, 0.05, 0.1, true));
  EXPECT_FALSE(std::filesystem::exists(path + ".2.json"));
  grid->resizeMap(80, 80, 0.05, -1.0, -1.0);
  std::fill(grid->getCharMap(), grid->getCharMap() + 6400, 254);
  for (int k = 0; k < 10; ++k) {
    pose.header.stamp.sec += 1;
    EXPECT_FALSE(optimizer.longCuspDepartureAvailable(pose, speed, outgoing, 0.05, 0.1, true));
  }
  size_t count = 0; std::set<int64_t> retained;
  for (int k = 0; k < 8; ++k) {
    const auto slot = path + "." + std::to_string(k) + ".json";
    ASSERT_TRUE(std::filesystem::exists(slot)); ++count;
    std::ifstream stream(slot); nlohmann::json row; stream >> row;
    retained.insert(row.at("stamp_ns").get<int64_t>());
    EXPECT_EQ(row.at("reason"), "initial_footprint");
    EXPECT_EQ(row.at("frame"), "odom");
    EXPECT_DOUBLE_EQ(row.at("robot")[0], 0.3); EXPECT_DOUBLE_EQ(row.at("robot")[1], 0.4);
    EXPECT_DOUBLE_EQ(row.at("speed")[0], -0.02);
    EXPECT_EQ(row.at("departure_prefix").size(), 3u);
    EXPECT_EQ(row.at("grid").at("data").size(), 6400u);
    EXPECT_EQ(row.at("grid").at("data")[123], 254);
    EXPECT_EQ(row.at("time_steps"), 41);
    EXPECT_GT(row.at("footprint").size(), 2u);
    EXPECT_TRUE(row.at("terminal_endpoint"));
    std::filesystem::remove(slot);
  }
  EXPECT_EQ(count, 8u);
  std::set<int64_t> expected;
  for (int sec = 2141; sec <= 2148; ++sec) {expected.insert(sec * 1000000000LL + 500000000);}
  EXPECT_EQ(retained, expected);
  EXPECT_EQ(optimizer.departureControlsForTest().vx, before.vx);
  EXPECT_EQ(optimizer.departureControlsForTest().wz, before.wz);
  unsetenv("MPPI_DEPARTURE_REJECTION_TRACE_PATH");
  optimizer.shutdown(); cm->on_cleanup(lifecycle); std::filesystem::remove(file);
}


TEST(OptimizerTests, RecordedPostHandoffGuardHorizonExceedsInitializerHorizon)
{
  namespace f=handoff_horizon_fixture;
  ASSERT_EQ(f::grid.size(),22500u);ASSERT_EQ(f::origin.size(),2u);
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("post_handoff_horizon");
  f::declareParameters(node);
  auto cm=std::make_shared<SettledTestCostmap>("post_handoff_horizon_map","","post_handoff_horizon_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string("[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid=cm->getCostmap();grid->resizeMap(150,150,0.02,f::origin[0],f::origin[1]);
  std::copy(f::grid.begin(),f::grid.end(),grid->getCharMap());
  ParametersHandler parameters(node);OptimizerTester optimizer;
  optimizer.initialize(node,"mppi",cm,&parameters);
  optimizer.testRecordedPostHandoffHorizon();
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}
TEST(OptimizerTests, RecordedLongHandoffDeliversRetainedExecutablePrefix)
{
  namespace f=handoff_horizon_fixture;
  ASSERT_EQ(f::grid.size(),22500u);ASSERT_EQ(f::origin.size(),2u);
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("post_handoff_horizon");
  f::declareParameters(node);
  auto cm=std::make_shared<SettledTestCostmap>("post_handoff_horizon_map","","post_handoff_horizon_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string("[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  cm->set_parameter(rclcpp::Parameter("global_frame",std::string("odom")));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid=cm->getCostmap();grid->resizeMap(150,150,0.02,f::origin[0],f::origin[1]);
  std::copy(f::grid.begin(),f::grid.end(),grid->getCharMap());
  ParametersHandler parameters(node);OptimizerTester optimizer;
  optimizer.initialize(node,"mppi",cm,&parameters);
  optimizer.testRecordedPostHandoffHorizon(true);
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}


TEST(OptimizerTests, RecordedLongPathHandoffSeedsLocalEntryWithoutChangingEndpoint)
{
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("long_prefix_entry");
  handoff_prefix_fixture::declareParameters(node);
  auto cm=std::make_shared<SettledTestCostmap>("long_prefix_entry_map","","long_prefix_entry_map",false);
  cm->set_parameter(rclcpp::Parameter("global_frame",std::string("odom")));
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string("[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid=cm->getCostmap();grid->resizeMap(150,150,0.02,handoff_prefix_fixture::origin[0],handoff_prefix_fixture::origin[1]);
  std::copy(handoff_prefix_fixture::grid.begin(),handoff_prefix_fixture::grid.end(),grid->getCharMap());
  ParametersHandler parameters(node);OptimizerTester optimizer;
  optimizer.initialize(node,"mppi",cm,&parameters);
  optimizer.testRecordedLongPathLocalEntry();
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, RecordedShortDepartureRejectsEarlyButClearReversePrefixesEnableSameFamily)
{
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("retained_departure");
  handoff_prefix_fixture::declareParameters(node);
  auto cm=std::make_shared<SettledTestCostmap>("retained_departure_map","","retained_departure_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame",std::string("odom")));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string("[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid=cm->getCostmap();grid->resizeMap(150,150,0.02,retained_departure_fixture::origin[0],retained_departure_fixture::origin[1]);
  std::copy(retained_departure_fixture::grid.begin(),retained_departure_fixture::grid.end(),grid->getCharMap());
  ParametersHandler parameters(node);OptimizerTester optimizer;
  optimizer.initialize(node,"mppi",cm,&parameters);
  optimizer.testRecordedRetainedDeparture();
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, RecordedBlockedForwardEntryRetainsClearHoldAndRejectsCornerCut)
{
  ASSERT_EQ(blocked_forward_entry_fixture::origin.size(),2u);
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("blocked_forward_entry");
  handoff_prefix_fixture::declareParameters(node);
  auto cm=std::make_shared<SettledTestCostmap>("blocked_forward_map","","blocked_forward_map",false);
  ASSERT_DOUBLE_EQ(node->get_parameter("mppi.AckermannConstraints.steering_actuator_rate").as_double(),1.0);
  ASSERT_DOUBLE_EQ(node->get_parameter("mppi.AckermannConstraints.response_radius_offset").as_double(),0.093);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame",std::string("odom")));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string("[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto * grid=cm->getCostmap();grid->resizeMap(150,150,0.02,blocked_forward_entry_fixture::origin[0],blocked_forward_entry_fixture::origin[1]);
  ASSERT_EQ(blocked_forward_entry_fixture::grid.size(),22500u);
  std::copy(blocked_forward_entry_fixture::grid.begin(),blocked_forward_entry_fixture::grid.end(),grid->getCharMap());
  ParametersHandler parameters(node);OptimizerTester optimizer;
  optimizer.initialize(node,"mppi",cm,&parameters);optimizer.testRecordedBlockedForwardEntry();
  optimizer.shutdown();cm->on_cleanup(lifecycle);
}

TEST(OptimizerTests, motorDeadbandFloorTests)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("deadband_test_node");
  OptimizerTester optimizer_tester;
  node->declare_parameter("controller_frequency", rclcpp::ParameterValue(20.0));
  node->declare_parameter("mppic.batch_size", rclcpp::ParameterValue(100));
  node->declare_parameter("mppic.time_steps", rclcpp::ParameterValue(20));
  node->declare_parameter("mppic.vx_max", rclcpp::ParameterValue(1.0));
  node->declare_parameter("mppic.vx_min", rclcpp::ParameterValue(-1.0));
  node->declare_parameter("mppic.wz_max", rclcpp::ParameterValue(2.0));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);
  optimizer_tester.initialize(node, "mppic", costmap_ros, &param_handler);

  optimizer_tester.testMotorDeadbandFloor();
}

