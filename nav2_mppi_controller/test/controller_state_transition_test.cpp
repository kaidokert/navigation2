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

#include "gtest/gtest.h"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <limits>

#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>

#include "nav2_mppi_controller/controller.hpp"

#include "utils/utils.hpp"
#include "settled_costmap.hpp"
#include "handoff_prefix_fixture.hpp"
#include "retained_departure_fixture.hpp"
#include "handoff_horizon_fixture.hpp"
#include "certificate_tf_fixture.hpp"

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

// Test-only member access avoids adding an API for mutating optimizer history.
class OptimizerMemoryAccess : public mppi::Optimizer
{
public:
  static auto sequenceMember() {return &OptimizerMemoryAccess::control_sequence_;}
  static auto historyMember() {return &OptimizerMemoryAccess::control_history_;}
  static auto modelMember() {return &OptimizerMemoryAccess::motion_model_;}
  static auto prepareMember() {return &OptimizerMemoryAccess::prepare;}
  static auto entryMember() {return &OptimizerMemoryAccess::long_entry_;}
  static auto pathMember() {return &OptimizerMemoryAccess::path_;}
};

class HandoffMemoryController : public nav2_mppi_controller::MPPIController
{
public:
  using MPPIController::receiveSteeringFeedback;
  using MPPIController::receiveConditionedSteering;
  using MPPIController::updateSteeringFeedback;
  bool hasLongEntryCertificate() const
  {return (optimizer_.*OptimizerMemoryAccess::entryMember()).has_value();}
  size_t longEntryShifts() const
  {return (optimizer_.*OptimizerMemoryAccess::entryMember())->shifts;}
  geometry_msgs::msg::Pose longEntryTarget() const
  {return (optimizer_.*OptimizerMemoryAccess::entryMember())->target;}
  size_t realPathPointCount() const {return path_handler_.getRealPathPointCount();}
  size_t entrySourceID() const {return (optimizer_.*OptimizerMemoryAccess::entryMember())->source_id;}
  uint64_t entryPlanGeneration() const {return (optimizer_.*OptimizerMemoryAccess::entryMember())->plan_generation;}
  auto pathBindingForTest() const {return path_handler_.getPathBinding();}
  auto transformForTest(const geometry_msgs::msg::PoseStamped & robot) {return path_handler_.transformPath(robot);}
  void prepareBindingForTest(const std::optional<mppi::PathBinding> & binding,
    const geometry_msgs::msg::PoseStamped & robot,const geometry_msgs::msg::Twist & speed,
    const nav_msgs::msg::Path & local,const geometry_msgs::msg::Pose & goal)
  {
    optimizer_.setPathBinding(binding);optimizer_.setRealPathPointCount(local.poses.size());
    (optimizer_.*OptimizerMemoryAccess::prepareMember())(robot,speed,local,goal,nullptr);
  }
  size_t targetCoordinateMatches(const geometry_msgs::msg::Pose & target) const
  {
    const auto & path=optimizer_.*OptimizerMemoryAccess::pathMember();size_t matches=0;
    for(size_t k=0;k<std::min(path.x.size(),realPathPointCount());++k) {
      if(std::hypot(path.x(k)-target.position.x,path.y(k)-target.position.y)<=1e-6 &&
        std::abs(angles::shortest_angular_distance(path.yaws(k),tf2::getYaw(target.orientation)))<=1e-6) {++matches;}
    }
    return matches;
  }
  void installCapturedLongEntry(const geometry_msgs::msg::PoseStamped & robot,
    const geometry_msgs::msg::Twist & speed,const nav_msgs::msg::Path & local)
  {
    namespace f=handoff_horizon_fixture;
    path_handler_.transformPath(robot);
    optimizer_.setPathBinding(path_handler_.getPathBinding());
    optimizer_.setRealPathPointCount(34);optimizer_.setActiveCuspDirection(1);
    optimizer_.setPathDirection(1);
    optimizer_.setCuspTolerances(std::make_pair(f::gates[0],f::gates[1]));
    optimizer_.requestShortCuspSeed();
    geometry_msgs::msg::Pose goal;goal.position.x=f::goal[0];goal.position.y=f::goal[1];
    goal.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),f::goal[2]));
    (optimizer_.*OptimizerMemoryAccess::prepareMember())(robot,speed,local,goal,nullptr);
  }
  void installLocalEntryForTest(const geometry_msgs::msg::PoseStamped & robot,
    const geometry_msgs::msg::Twist & speed,const nav_msgs::msg::Path & local,
    const geometry_msgs::msg::Pose & goal,double xy,double yaw)
  {
    path_handler_.transformPath(robot);optimizer_.setPathBinding(path_handler_.getPathBinding());
    optimizer_.setRealPathPointCount(local.poses.size());optimizer_.setActiveCuspDirection(1);
    optimizer_.setPathDirection(1);optimizer_.setCuspTolerances(std::make_pair(xy,yaw));
    optimizer_.requestShortCuspSeed();
    (optimizer_.*OptimizerMemoryAccess::prepareMember())(robot,speed,local,goal,nullptr);
  }
  void expireConditionedFeedback()
  {
    conditioned_steering_receipt_ = parent_.lock()->now() - rclcpp::Duration::from_seconds(0.31);
  }
  float predictStandstillSteering()
  {
    mppi::models::State state;
    state.reset(1, 2);
    state.cvx.fill(0.05f);state.cwz.fill(-0.1f);
    (optimizer_.*OptimizerMemoryAccess::modelMember())->predict(state);
    return state.wz(0, 1);
  }
  void seedPreviousGear(float speed)
  {
    seedCommand(speed, 0.0f);
  }
  void seedCommand(float speed, float yaw_rate)
  {
    auto & sequence = optimizer_.*OptimizerMemoryAccess::sequenceMember();
    sequence.vx.fill(speed);
    sequence.vy.fill(0.0f);
    sequence.wz.fill(yaw_rate);
    auto & history = optimizer_.*OptimizerMemoryAccess::historyMember();
    for (auto & command : history) {
      command = {speed, 0.0f, yaw_rate};
    }
  }
};



TEST(ControllerStateTransitionTest, RecordedLongEntrySurvivesObservedLocalizationTransform)
{
  const auto & d=certificate_tf_fixture::data;
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("recorded_certificate_tf");
  handoff_horizon_fixture::declareParameters(node);
  node->set_parameter(rclcpp::Parameter("mppi.iteration_count",0));
  node->set_parameter(rclcpp::Parameter("mppi.critics",std::vector<std::string>{}));
  auto cm=std::make_shared<SettledTestCostmap>("recorded_tf_map","","recorded_tf_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame","odom"));
  cm->set_parameter(rclcpp::Parameter("footprint",d["footprint"].dump()));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  auto grid=d["grid"]["data"].get<std::vector<unsigned char>>();
  cm->getCostmap()->resizeMap(d["grid"]["width"],d["grid"]["height"],d["grid"]["resolution"],d["grid"]["origin"][0],d["grid"]["origin"][1]);
  ASSERT_EQ(grid.size(),cm->getCostmap()->getSizeInCellsX()*cm->getCostmap()->getSizeInCellsY());
  std::copy(grid.begin(),grid.end(),cm->getCostmap()->getCharMap());
  HandoffMemoryController controller;controller.configure(node,"mppi",cm->getTfBuffer(),cm);controller.activate();
  const auto pose=[&](const nlohmann::json & a,double stamp) {
    geometry_msgs::msg::PoseStamped p;p.header.frame_id="odom";p.header.stamp=rclcpp::Time(int64_t(stamp*1e9));
    p.pose.position.x=a[0];p.pose.position.y=a[1];
    p.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),a[2].get<double>()));return p;
  };
  const auto feedback=[&](double physical,double conditioned) {
    auto joint=std::make_shared<sensor_msgs::msg::JointState>();joint->header.stamp=node->now();
    joint->name={"front_steering_joint"};joint->position={physical};controller.receiveSteeringFeedback(joint);
    auto target=std::make_shared<std_msgs::msg::Float64>();target->data=conditioned;controller.receiveConditionedSteering(target);
    controller.updateSteeringFeedback();
  };
  const auto transform=[&](const nlohmann::json & a,double stamp) {
    geometry_msgs::msg::TransformStamped t;t.header.frame_id="map";t.child_frame_id="odom";
    t.header.stamp=rclcpp::Time(int64_t(stamp*1e9));t.transform.translation.x=a[0];t.transform.translation.y=a[1];
    t.transform.rotation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),a[2].get<double>()));
    ASSERT_TRUE(cm->getTfBuffer()->setTransform(t,"recorded_capture",false));
  };
  transform(d["old_tf"],60.15);transform(d["old_tf"],60.25);
  auto robot=pose(d["robot"],60.25);geometry_msgs::msg::Twist speed;
  speed.linear.x=d["speed"][0];speed.angular.z=d["speed"][1];
  nav_msgs::msg::Path local;local.header=robot.header;
  for(size_t k=0;k<d["path_x"].size();++k) {
    nlohmann::json a={d["path_x"][k],d["path_y"][k],d["path_yaw"][k]};local.poses.push_back(pose(a,60.25));
  }
  nav_msgs::msg::Path canonical;canonical.header.frame_id="map";
  // Actual recorded global plan, beginning at source point29 corresponding to
  // captured local44[0]; downstream branch/cusps are recorded, not invented.
  for(const auto & a:d["canonical_full_suffix"]) {
    auto p=pose(a,60.25);p.header.frame_id="map";canonical.poses.push_back(p);
  }
  controller.setPlan(canonical);
  feedback(d["initial_steering"],d["initial_conditioned_steering"]);
  auto goal=pose(d["goal"],60.25).pose;
  controller.installLocalEntryForTest(robot,speed,local,goal,d["effective_cusp_tolerances"][0],d["effective_cusp_tolerances"][1]);
  ASSERT_TRUE(controller.hasLongEntryCertificate());
  const auto first=controller.computeVelocityCommands(robot,speed,{});
  EXPECT_GT(first.twist.linear.x,0.0);ASSERT_TRUE(controller.hasLongEntryCertificate());
  const auto original_target=controller.longEntryTarget();
  const auto source_id=controller.entrySourceID();const auto generation=controller.entryPlanGeneration();
  const auto original_shifts=controller.longEntryShifts();
  ASSERT_EQ(controller.targetCoordinateMatches(original_target),1u);
  transform(d["new_tf"],60.35);
  robot=pose(d["second_robot"],60.306);speed.linear.x=d["second_speed"][0];speed.angular.z=d["second_speed"][1];
  feedback(d["second_feedback"][0],d["second_feedback"][1]);
  // Desired behavior: unchanged canonical target survives a harmless localization
  // update, with the remaining tail still revalidated against the captured grid.
  geometry_msgs::msg::TwistStamped second;
  ASSERT_NO_THROW(second=controller.computeVelocityCommands(robot,speed,{}));
  EXPECT_GT(second.twist.linear.x,0.0);ASSERT_TRUE(controller.hasLongEntryCertificate());
  EXPECT_EQ(controller.longEntryShifts(),original_shifts+1);
  EXPECT_EQ(controller.entrySourceID(),source_id);EXPECT_EQ(controller.entryPlanGeneration(),generation);
  EXPECT_EQ(controller.targetCoordinateMatches(original_target),0u);
  EXPECT_EQ(controller.targetCoordinateMatches(controller.longEntryTarget()),1u);
  const auto first_robot=pose(d["robot"],60.25);geometry_msgs::msg::Twist first_speed;
  first_speed.linear.x=d["speed"][0];first_speed.angular.z=d["speed"][1];
  const auto fresh=[&]() {
    std::copy(grid.begin(),grid.end(),cm->getCostmap()->getCharMap());
    controller.setPlan(canonical);feedback(d["initial_steering"],d["initial_conditioned_steering"]);
    controller.installLocalEntryForTest(first_robot,first_speed,local,goal,d["effective_cusp_tolerances"][0],d["effective_cusp_tolerances"][1]);
  };
  fresh();auto binding=controller.pathBindingForTest();ASSERT_TRUE(binding);
  // Pruning is allowed to remove the selected point from the local prefix; the
  // SAME immutable source point remains available within its active segment.
  auto pruned_binding=*binding;auto pruned_local=local;
  const auto selected=controller.entrySourceID();
  const auto cut=std::find(pruned_binding.real_ids.begin(),pruned_binding.real_ids.end(),selected);
  ASSERT_NE(cut,pruned_binding.real_ids.end());
  const size_t count=std::distance(pruned_binding.real_ids.begin(),cut)+1;
  pruned_binding.real_ids.erase(pruned_binding.real_ids.begin(),pruned_binding.real_ids.begin()+count);
  pruned_local.poses.erase(pruned_local.poses.begin(),pruned_local.poses.begin()+count);
  ASSERT_GT(pruned_local.poses.size(),4u);
  ASSERT_NO_THROW(controller.prepareBindingForTest(pruned_binding,first_robot,first_speed,pruned_local,goal));
  EXPECT_EQ(controller.entrySourceID(),selected);EXPECT_TRUE(controller.hasLongEntryCertificate());
  fresh();binding=controller.pathBindingForTest();
  auto bad=*binding;++bad.generation;
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();bad=*controller.pathBindingForTest();++bad.segment;
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();bad=*controller.pathBindingForTest();bad.real_ids[1]=bad.real_ids[2];
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();bad=*controller.pathBindingForTest();bad.real_ids.push_back(bad.real_ids.back()+1);
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();bad=*controller.pathBindingForTest();bad.canonical_to_local.transform.rotation.w=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();bad=*controller.pathBindingForTest();bad.canonical_to_local.header.stamp=rclcpp::Time(1);
  EXPECT_THROW(controller.prepareBindingForTest(bad,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();EXPECT_THROW(controller.prepareBindingForTest(std::nullopt,first_robot,first_speed,local,goal),nav2_core::FailedToMakeProgress);
  fresh();const auto old_generation=controller.entryPlanGeneration();
  controller.setPlan(canonical);EXPECT_FALSE(controller.hasLongEntryCertificate());
  controller.installLocalEntryForTest(first_robot,first_speed,local,goal,d["effective_cusp_tolerances"][0],d["effective_cusp_tolerances"][1]);
  EXPECT_NE(controller.entryPlanGeneration(),old_generation);
  controller.reset();EXPECT_FALSE(controller.hasLongEntryCertificate());
  fresh();std::fill(cm->getCostmap()->getCharMap(),cm->getCostmap()->getCharMap()+grid.size(),254);
  feedback(d["second_feedback"][0],d["second_feedback"][1]);
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::FailedToMakeProgress);
  std::copy(grid.begin(),grid.end(),cm->getCostmap()->getCharMap());
  auto unknown=canonical;unknown.header.frame_id="unrecorded_frame";
  for(auto & point:unknown.poses) {point.header.frame_id=unknown.header.frame_id;}
  controller.setPlan(unknown);
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::ControllerTFError);
  // A valid TF quaternion with nonplanar roll must be rejected by PathHandler
  // before transforming path points into costmap coordinates.
  geometry_msgs::msg::TransformStamped rolled;
  rolled.header.frame_id="odom";rolled.child_frame_id="rolled_plan";
  rolled.transform.rotation.x=std::sin(0.1);rolled.transform.rotation.w=std::cos(0.1);
  ASSERT_TRUE(cm->getTfBuffer()->setTransform(rolled,"native_regression",true));
  auto nonplanar=canonical;nonplanar.header.frame_id="rolled_plan";
  for(auto & point:nonplanar.poses) {point.header.frame_id=nonplanar.header.frame_id;}
  controller.setPlan(nonplanar);
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::ControllerTFError);
  controller.deactivate();controller.cleanup();cm->on_cleanup(lifecycle);
}

TEST(ControllerStateTransitionTest, CanonicalSourceIDsSurvivePruningAndBothHandoffs)
{
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("source_id_pruning");
  handoff_horizon_fixture::declareParameters(node);
  node->set_parameter(rclcpp::Parameter("mppi.bound_handoff_by_next_segment",false));
  auto cm=std::make_shared<SettledTestCostmap>("source_id_map","","source_id_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame","odom"));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  cm->getCostmap()->resizeMap(100,100,0.05,-2.5,-2.5);
  mppi::ParametersHandler parameters(node);mppi::PathHandler path;
  path.initialize(node,"mppi",cm,cm->getTfBuffer(),&parameters);
  nav_msgs::msg::Path full;full.header.frame_id="odom";
  for(double x:{0.0,0.1,0.2,0.1,0.0,-0.1,0.0,0.1,0.2}) {
    geometry_msgs::msg::PoseStamped point;point.header=full.header;
    point.pose.position.x=x;point.pose.orientation.w=1;full.poses.push_back(point);
  }
  path.setPath(full);
  const auto at=[&](double x) {
    geometry_msgs::msg::PoseStamped robot;robot.header=full.header;robot.header.stamp=node->now();
    robot.pose.position.x=x;robot.pose.orientation.w=1;return path.transformPath(robot);
  };
  at(0.1);auto a=path.getPathBinding();ASSERT_TRUE(a);ASSERT_FALSE(a->real_ids.empty());
  EXPECT_EQ(a->real_ids.front(),1u);const auto generation=a->generation;
  at(0.2);auto b=path.getPathBinding();ASSERT_TRUE(b);
  EXPECT_EQ(b->segment_begin,2u);EXPECT_EQ(b->real_ids.front(),2u);
  EXPECT_EQ(b->segment,a->segment+1);EXPECT_EQ(b->generation,generation);
  const auto reverse=at(0.1);auto c=path.getPathBinding();ASSERT_TRUE(c);
  EXPECT_EQ(c->real_ids.front(),3u);EXPECT_EQ(c->segment_begin,2u);
  EXPECT_EQ(c->real_ids.size(),path.getRealPathPointCount());
  EXPECT_LE(c->real_ids.size(),reverse.poses.size());
  EXPECT_DOUBLE_EQ(c->canonical_plan->poses[0].pose.position.x,0.0);
  EXPECT_DOUBLE_EQ(c->canonical_plan->poses[4].pose.position.x,0.0); // Distinct IDs at same coordinate.
  at(-0.1);auto e=path.getPathBinding();ASSERT_TRUE(e);
  EXPECT_EQ(e->segment_begin,5u);EXPECT_EQ(e->real_ids.front(),5u);
  EXPECT_EQ(e->segment,b->segment+1);
  at(0.0);auto f=path.getPathBinding();ASSERT_TRUE(f);EXPECT_EQ(f->real_ids.front(),6u);
  path.setPath(full);at(0.1);auto next=path.getPathBinding();ASSERT_TRUE(next);
  EXPECT_GT(next->generation,generation);EXPECT_EQ(next->real_ids.front(),1u);
  cm->on_cleanup(lifecycle);
}

TEST(ControllerStateTransitionTest, LongEntryCertificateSurvivesActualFeedbackRefreshAndPathPruning)
{
  namespace f=handoff_horizon_fixture;
  ASSERT_EQ(f::robot.size(),3u);ASSERT_EQ(f::speed.size(),2u);ASSERT_EQ(f::steering.size(),2u);
  ASSERT_EQ(f::path_x.size(),34u);ASSERT_EQ(f::path_y.size(),34u);ASSERT_EQ(f::path_yaw.size(),34u);
  ASSERT_EQ(f::origin.size(),2u);ASSERT_EQ(f::grid.size(),22500u);
  auto node=std::make_shared<rclcpp_lifecycle::LifecycleNode>("certificate_controller_refresh");
  f::declareParameters(node);
  // Isolate deterministic delivery from the randomized Gaussian update. Actual
  // controller refresh, PathHandler, final constraints and collision veto run.
  node->set_parameter(rclcpp::Parameter("mppi.iteration_count",0));
  // No critic scores are evaluated with zero iterations. Avoid loading unused
  // critic DSOs here: unloading them also pollutes later costmap-layer factories
  // in this process. Native certificate/final footprint checks still run.
  node->set_parameter(rclcpp::Parameter("mppi.critics",std::vector<std::string>{}));
  auto cm=std::make_shared<SettledTestCostmap>("certificate_refresh_map","","certificate_refresh_map",false);
  cm->set_parameter(rclcpp::Parameter("plugins",std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame","odom"));
  cm->set_parameter(rclcpp::Parameter("footprint",std::string(
    "[[-0.06000000052154064,-0.1609999928623438],[0.2859999928623438,-0.1609999928623438],[0.2859999928623438,0.1609999928623438],[-0.06000000052154064,0.1609999928623438]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding",0.0));
  rclcpp_lifecycle::State lifecycle;cm->on_configure(lifecycle);
  cm->getCostmap()->resizeMap(150,150,0.02,f::origin[0],f::origin[1]);
  std::copy(f::grid.begin(),f::grid.end(),cm->getCostmap()->getCharMap());
  HandoffMemoryController controller;controller.configure(node,"mppi",cm->getTfBuffer(),cm);
  controller.activate();
  const auto make_pose=[&](double x,double y,double yaw) {
      geometry_msgs::msg::PoseStamped p;p.header.frame_id="odom";p.header.stamp=node->now();
      p.pose.position.x=x;p.pose.position.y=y;
      p.pose.orientation=tf2::toMsg(tf2::Quaternion(tf2::Vector3(0,0,1),yaw));return p;
    };
  auto robot=make_pose(f::robot[0],f::robot[1],f::robot[2]);
  geometry_msgs::msg::Twist speed;speed.linear.x=f::speed[0];speed.angular.z=f::speed[1];
  nav_msgs::msg::Path local;local.header=robot.header;
  for(size_t k=0;k<34;++k) {local.poses.push_back(make_pose(f::path_x[k],f::path_y[k],f::path_yaw[k]));}
  nav_msgs::msg::Path full=local;
  // Synthetic preceding SAME-gear point and downstream reverse edge exercise
  // genuine closest-point pruning/active-cusp dispatch. Only outgoing34 and
  // physical state/raster are recorded; this is not an original global-plan replay.
  auto preceding=local.poses.front();preceding.pose.position.x-=f::path_x[1]-f::path_x[0];
  preceding.pose.position.y-=f::path_y[1]-f::path_y[0];
  full.poses.insert(full.poses.begin(),preceding);
  full.poses.push_back(local.poses[32]);
  controller.setPlan(full);
  const auto fresh_feedback=[&]() {
      auto physical=std::make_shared<sensor_msgs::msg::JointState>();physical->header.stamp=node->now();
      physical->name={"front_steering_joint"};physical->position={f::steering[0]};
      controller.receiveSteeringFeedback(physical);
      auto conditioned=std::make_shared<std_msgs::msg::Float64>();conditioned->data=f::steering[1];
      controller.receiveConditionedSteering(conditioned);return physical;
    };
  fresh_feedback();controller.updateSteeringFeedback();
  controller.installCapturedLongEntry(robot,speed,local);
  ASSERT_TRUE(controller.hasLongEntryCertificate());const auto target=controller.longEntryTarget();
  // These are the REAL provisional-clear + reinstatement calls, not direct
  // optimizer setter stand-ins. The eager-clear regression loses the proof here.
  for(size_t refresh=0;refresh<2;++refresh) {
    fresh_feedback();controller.updateSteeringFeedback();
    EXPECT_TRUE(controller.hasLongEntryCertificate()) << "Healthy controller feedback refresh erased certificate";
  }
  if(!controller.hasLongEntryCertificate()) {
    controller.deactivate();controller.cleanup();cm->on_cleanup(lifecycle);return;
  }
  for(size_t tick=0;tick<2;++tick) {
    fresh_feedback();robot.header.stamp=node->now();
    const auto command=controller.computeVelocityCommands(robot,speed,{});
    EXPECT_GT(command.twist.linear.x,0.0);
    ASSERT_TRUE(controller.hasLongEntryCertificate());
    EXPECT_EQ(controller.longEntryShifts(),tick+1);
    EXPECT_EQ(controller.realPathPointCount(),34u); // Preceding point was pruned.
    EXPECT_DOUBLE_EQ(controller.longEntryTarget().position.x,target.position.x);
    EXPECT_DOUBLE_EQ(controller.longEntryTarget().position.y,target.position.y);
  }
  const auto reinstall=[&]() {
      fresh_feedback();controller.updateSteeringFeedback();robot.header.stamp=node->now();
      controller.installCapturedLongEntry(robot,speed,local);
    };
  reinstall();controller.receiveSteeringFeedback({});
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::NoValidControl);
  reinstall();auto stale=fresh_feedback();
  stale->header.stamp=node->now()-rclcpp::Duration::from_seconds(0.31);
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::NoValidControl);
  reinstall();auto invalid=fresh_feedback();invalid->position[0]=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::NoValidControl);
  reinstall();controller.expireConditionedFeedback();
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::NoValidControl);
  reinstall();auto invalid_target=std::make_shared<std_msgs::msg::Float64>();
  invalid_target->data=std::numeric_limits<double>::quiet_NaN();
  controller.receiveConditionedSteering(invalid_target);
  EXPECT_THROW(controller.computeVelocityCommands(robot,speed,{}),nav2_core::NoValidControl);
  // Synthetic map->odom identity first, then a changed transform with no new
  // plan request: the SAME canonical source target is freshly transformed,
  // preserving identity while retaining full native-tail collision validation.
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id="odom";transform.child_frame_id="certificate_map";
  transform.header.stamp=node->now();transform.transform.rotation.w=1.0;
  ASSERT_TRUE(cm->getTfBuffer()->setTransform(transform,"certificate_test",true));
  auto map_plan=full;map_plan.header.frame_id="certificate_map";
  for(auto & point:map_plan.poses) {point.header.frame_id="certificate_map";}
  controller.setPlan(map_plan);reinstall();fresh_feedback();robot.header.stamp=node->now();
  EXPECT_GT(controller.computeVelocityCommands(robot,speed,{}).twist.linear.x,0.0);
  ASSERT_TRUE(controller.hasLongEntryCertificate());
  transform.transform.translation.x=0.001;transform.header.stamp=node->now();
  ASSERT_TRUE(cm->getTfBuffer()->setTransform(transform,"certificate_test",true));
  fresh_feedback();robot.header.stamp=node->now();
  EXPECT_GT(controller.computeVelocityCommands(robot,speed,{}).twist.linear.x,0.0);
  EXPECT_TRUE(controller.hasLongEntryCertificate());
  controller.deactivate();controller.cleanup();cm->on_cleanup(lifecycle);
}

void exerciseRetainedApproachFinalCommand(bool recorded_collision)
{
  namespace f = retained_departure_fixture;
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("retained_command");
  handoff_prefix_fixture::declareParameters(node);
  node->set_parameter(rclcpp::Parameter("mppi.iteration_count", 0));
  node->set_parameter(rclcpp::Parameter("mppi.critics", std::vector<std::string>{}));
  for (const auto * key : {"mppi.preserve_short_cusp_feasibility",
      "mppi.preserve_filtered_collision_feasibility", "mppi.seed_short_cusp_on_handoff",
      "mppi.initialize_long_cusp_entry", "mppi.smooth_control_sequence",
      "mppi.extend_short_cusp_completion", "mppi.retain_collision_free_long_path_sample",
      "mppi.repair_long_cusp_warm_start"})
  {
    node->set_parameter(rclcpp::Parameter(key, false));
  }
  // Isolate the final-command wiring, not MPPI's randomized optimization or
  // other feasibility fallbacks. The real departure predictor remains enabled.
  auto cm = std::make_shared<SettledTestCostmap>("retained_command_map", "", "retained_command_map", false);
  cm->set_parameter(rclcpp::Parameter("plugins", std::vector<std::string>{}));
  cm->set_parameter(rclcpp::Parameter("global_frame", "odom"));
  cm->set_parameter(rclcpp::Parameter("footprint", std::string(
      "[[-0.06,-0.161],[0.286,-0.161],[0.286,0.161],[-0.06,0.161]]")));
  cm->set_parameter(rclcpp::Parameter("footprint_padding", 0.0));
  rclcpp_lifecycle::State lifecycle;
  cm->on_configure(lifecycle);
  cm->getCostmap()->resizeMap(150, 150, 0.02, f::origin[0], f::origin[1]);
  std::copy(f::grid.begin(), f::grid.end(), cm->getCostmap()->getCharMap());
  // Clearing this raster is a labelled synthetic control: it isolates the
  // retained-command envelope from the independent full-horizon collision veto.
  if (!recorded_collision) {
    std::fill_n(cm->getCostmap()->getCharMap(), 22500, nav2_costmap_2d::FREE_SPACE);
  }
  HandoffMemoryController controller;
  controller.configure(node, "mppi", cm->getTfBuffer(), cm);
  controller.activate();
  auto feedback = std::make_shared<sensor_msgs::msg::JointState>();
  feedback->header.stamp = node->now();
  feedback->name = {"front_steering_joint"};feedback->position = {f::steering0[0]};
  controller.receiveSteeringFeedback(feedback);
  auto target = std::make_shared<std_msgs::msg::Float64>();target->data = f::steering0[1];
  controller.receiveConditionedSteering(target);
  const auto pose = [&](double x, double y, double yaw) {
      geometry_msgs::msg::PoseStamped value;value.header.frame_id = "odom";
      value.header.stamp = node->now();value.pose.position.x = x;value.pose.position.y = y;
      tf2::Quaternion q;q.setRPY(0, 0, yaw);value.pose.orientation = tf2::toMsg(q);
      return value;
    };
  nav_msgs::msg::Path plan;plan.header.frame_id = "odom";
  // Synthetic incoming straight edge isolates dispatch. Outgoing prefix, state
  // and raster are captured; this is not an exact replay of the full planner path.
  for (double distance : {0.2, 0.1}) {
    plan.poses.push_back(pose(f::path_x[0] + distance * std::cos(f::path_yaw[0]),
      f::path_y[0] + distance * std::sin(f::path_yaw[0]), f::path_yaw[0]));
  }
  for (size_t k = 0; k < f::path_x.size(); ++k) {
    plan.poses.push_back(pose(f::path_x[k], f::path_y[k], f::path_yaw[k]));
  }
  plan.poses.push_back(plan.poses[plan.poses.size() - 2]); // Internal forward cusp.
  controller.setPlan(plan);
  auto robot = pose(f::robot0[0], f::robot0[1], f::robot0[2]);
  geometry_msgs::msg::Twist speed;speed.linear.x = f::speed0[0];speed.angular.z = f::speed0[1];
  controller.seedCommand(f::command[0], f::command[1]);
  EXPECT_LT(controller.computeVelocityCommands(robot, speed, {}).twist.linear.x, 0.0);
  // A later optimizer result can differ from the safe carried sequence. It must
  // not be returned merely because PathHandler allowed a bounded approach.
  controller.seedCommand(-0.3f, -0.3f / 0.401f);
  robot.header.stamp = node->now();
  if (recorded_collision) {
    try {
      controller.computeVelocityCommands(robot, speed, {});
      ADD_FAILURE() << "Colliding retained forecast escaped the final veto";
    } catch (const nav2_core::NoValidControl & error) {
      EXPECT_NE(std::string(error.what()).find("Final MPPI control sequence is not collision-free"),
        std::string::npos);
    }
  } else {
    try {
      controller.computeVelocityCommands(robot, speed, {});
      ADD_FAILURE() << "Unsafe final command escaped the retained approach envelope";
    } catch (const nav2_core::FailedToMakeProgress & error) {
      EXPECT_NE(std::string(error.what()).find("approach command exceeds"), std::string::npos);
    }
  }
  controller.deactivate();controller.cleanup();cm->on_cleanup(lifecycle);
}

TEST(ControllerStateTransitionTest, RetainedApproachChecksFinalCommandBeforeReturningIt)
{
  exerciseRetainedApproachFinalCommand(false);
}

TEST(ControllerStateTransitionTest, RetainedApproachCollisionVetoPrecedesEnvelope)
{
  exerciseRetainedApproachFinalCommand(true);
}

void exerciseHandoffMemory(bool clear_history, bool steering_feedback = false)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("handoff_memory");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("handoff_memory.model_dt", 0.05);
  node->declare_parameter("handoff_memory.time_steps", 41);
  // No optimization: isolate carried history from randomized critic updates.
  node->declare_parameter("handoff_memory.iteration_count", 0);
  node->declare_parameter("handoff_memory.critics", std::vector<std::string>{});
  node->declare_parameter("handoff_memory.motion_model", std::string("Ackermann"));
  node->declare_parameter("handoff_memory.AckermannConstraints.predict_steering_conditioning",
    steering_feedback);
  node->declare_parameter("handoff_memory.enforce_path_inversion", true);
  if (clear_history) {
    node->declare_parameter("handoff_memory.reset_on_path_handoff", true);
  }
  node->declare_parameter("handoff_memory.inversion_xy_tolerance", 0.01);
  node->declare_parameter("handoff_memory.ghost_max_extension", 0.0);
  auto costmap_ros = getDummyCostmapRos();
  costmap_ros->getCostmap()->resizeMap(100, 100, 0.05, -2.5, -2.5);
  std::fill_n(costmap_ros->getCostmap()->getCharMap(), 10000, nav2_costmap_2d::FREE_SPACE);
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  HandoffMemoryController controller;
  controller.configure(node, "handoff_memory", tf, costmap_ros);
  controller.activate();
  if (steering_feedback) {
    auto feedback = std::make_shared<sensor_msgs::msg::JointState>();
    feedback->header.stamp = node->now();
    feedback->name = {"front_steering_joint"};
    feedback->position = {0.3};
    controller.receiveSteeringFeedback(feedback);
  }
  nav_msgs::msg::Path path;
  path.header.frame_id = costmap_ros->getGlobalFrameID();
  for (double x : {0.0, 0.048, 0.096, 0.048, 0.096, 0.144, 0.192}) {
    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = x;
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);
  }
  controller.setPlan(path);
  auto pose = path.poses.front();
  geometry_msgs::msg::Twist stopped;
  controller.seedPreviousGear(0.2f);
  EXPECT_GT(controller.computeVelocityCommands(pose, stopped, {}).twist.linear.x, 0.15);
  pose.pose.position.x = 0.096;
  controller.seedPreviousGear(0.2f);
  EXPECT_NEAR(controller.computeVelocityCommands(pose, stopped, {}).twist.linear.x,
    clear_history ? 0.0 : 0.2, 1e-6)
    << "Old forward controls and smoothing history leaked into the reverse segment";
  if (steering_feedback) {
    EXPECT_NEAR(controller.predictStandstillSteering(), 0.05f * std::tan(0.3f) / 0.213f, 1e-6);
  }
  controller.seedPreviousGear(-0.2f);
  EXPECT_LT(controller.computeVelocityCommands(pose, stopped, {}).twist.linear.x, -0.15)
    << "History must remain usable between genuine handoffs";
  pose.pose.position.x = 0.048;
  controller.seedPreviousGear(-0.2f);
  EXPECT_NEAR(controller.computeVelocityCommands(pose, stopped, {}).twist.linear.x,
    clear_history ? 0.0 : -0.2, 1e-6)
    << "Old reverse history leaked into the next forward segment";
  if (steering_feedback) {
    EXPECT_NEAR(controller.predictStandstillSteering(), 0.05f * std::tan(0.3f) / 0.213f, 1e-6);
  }
  controller.deactivate();
  controller.cleanup();
}

TEST(ControllerStateTransitionTest, PreviousGearHistoryDoesNotSurviveCuspHandoff)
{
  exerciseHandoffMemory(true);
}

TEST(ControllerStateTransitionTest, DefaultPreservesLegacyHandoffMemoryForComparison)
{
  exerciseHandoffMemory(false);
}

TEST(ControllerStateTransitionTest, MeasuredSteeringIsRestoredAfterHandoffReset)
{
  exerciseHandoffMemory(true, true);
}

TEST(ControllerStateTransitionTest, SteeringFeedbackFailsClosedOnMissingStaleOrInvalidData)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("steering_feedback_validation");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("test.model_dt", 0.05);
  node->declare_parameter("test.motion_model", std::string("Ackermann"));
  node->declare_parameter("test.critics", std::vector<std::string>{});
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  auto costmap_ros = getDummyCostmapRos();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  HandoffMemoryController controller;
  controller.configure(node, "test", tf, costmap_ros);
  controller.activate();
  EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
  auto fresh = [&]() {
      auto message = std::make_shared<sensor_msgs::msg::JointState>();
      message->header.stamp = node->now();
      message->name = {"rear_left_wheel_joint", "front_steering_joint"};
      message->position = {5.0, 0.3};
      return message;
    };
  auto accept = [&](sensor_msgs::msg::JointState::SharedPtr message, bool valid) {
      controller.receiveSteeringFeedback(message);
      if (valid) {EXPECT_NO_THROW(controller.updateSteeringFeedback());} else {
        EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
      }
    };
  accept(fresh(), true);
  auto message = fresh();message->name.clear();accept(message, false);
  message = fresh();message->position.resize(1);accept(message, false);
  message = fresh();message->name.push_back("front_steering_joint");accept(message, false);
  message = fresh();message->header.stamp = rclcpp::Time(0);accept(message, false);
  message = fresh();message->header.stamp.sec = -1;accept(message, false);
  message = fresh();message->header.stamp.nanosec = 1000000000u;accept(message, false);
  message = fresh();message->header.stamp = node->now() - rclcpp::Duration::from_seconds(0.31);
  accept(message, false);
  message = fresh();message->header.stamp = node->now() + rclcpp::Duration::from_seconds(1.0);
  accept(message, false);
  message = fresh();message->position[1] = std::numeric_limits<double>::quiet_NaN();
  accept(message, false);
  message = fresh();message->position[1] = std::numeric_limits<double>::infinity();
  accept(message, false);
  message = fresh();message->position[1] = 0.6;accept(message, false);
  accept(fresh(), true);
  controller.deactivate();
  EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
  controller.cleanup();
}

TEST(ControllerStateTransitionTest, ServoPredictionRequiresFreshSeparateTargetFeedback)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("conditioned_feedback_validation");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("test.model_dt", 0.05);
  node->declare_parameter("test.motion_model", std::string("Ackermann"));
  node->declare_parameter("test.critics", std::vector<std::string>{});
  node->declare_parameter("test.AckermannConstraints.predict_steering_conditioning", true);
  node->declare_parameter("test.AckermannConstraints.steering_actuator_rate", 1.0);
  node->declare_parameter("test.AckermannConstraints.conditioned_steering_topic",
    std::string("/test_conditioned_target"));
  auto costmap_ros = getDummyCostmapRos();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  HandoffMemoryController controller;
  controller.configure(node, "test", tf, costmap_ros);
  controller.activate();
  auto feedback = std::make_shared<sensor_msgs::msg::JointState>();
  feedback->header.stamp = node->now();
  feedback->name = {"front_steering_joint"};feedback->position = {0.0};
  controller.receiveSteeringFeedback(feedback);
  EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
  auto target = std::make_shared<std_msgs::msg::Float64>();target->data = 0.3;
  controller.receiveConditionedSteering(target);
  EXPECT_NO_THROW(controller.updateSteeringFeedback());
  EXPECT_NEAR(controller.predictStandstillSteering(), 0.05f * std::tan(0.05f) / 0.213f, 1e-6);
  controller.expireConditionedFeedback();
  EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
  EXPECT_THROW(controller.predictStandstillSteering(), nav2_core::NoValidControl);
  for (double invalid : {0.6, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    target = std::make_shared<std_msgs::msg::Float64>();target->data = invalid;
    controller.receiveConditionedSteering(target);
    EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
    EXPECT_THROW(controller.predictStandstillSteering(), nav2_core::NoValidControl);
  }
  target = std::make_shared<std_msgs::msg::Float64>();target->data = 0.3;
  controller.receiveConditionedSteering(target);
  EXPECT_NO_THROW(controller.updateSteeringFeedback());
  controller.deactivate();
  EXPECT_THROW(controller.updateSteeringFeedback(), nav2_core::NoValidControl);
  controller.cleanup();
}

// Tests basic transition from configure->active->process->deactive->cleanup

TEST(ControllerStateTransitionTest, ControllerNotFail)
{
  const bool visualize = true;
  TestCostmapSettings costmap_settings{};

  // Node Options
  rclcpp::NodeOptions options;
  std::vector<rclcpp::Parameter> params;
  setUpControllerParams(visualize, params);
  options.parameter_overrides(params);

  auto node = getDummyNode(options);
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap_ros = getDummyCostmapRos(costmap_settings);
  costmap_ros->setRobotFootprint(getDummySquareFootprint(0.01));

  auto controller = getDummyController(node, tf_buffer, costmap_ros);

  TestPose start_pose = costmap_settings.getCenterPose();
  const double path_step = costmap_settings.resolution;
  TestPathSettings path_settings{start_pose, 8, path_step, path_step};

  // evalControl args
  auto pose = getDummyPointStamped(node, start_pose);
  auto velocity = getDummyTwist();
  auto path = getIncrementalDummyPath(node, path_settings);
  path.header.frame_id = costmap_ros->getGlobalFrameID();
  pose.header.frame_id = costmap_ros->getGlobalFrameID();

  controller->setPlan(path);

  EXPECT_NO_THROW(controller->computeVelocityCommands(pose, velocity, {}));

  controller->setSpeedLimit(0.5, true);
  controller->setSpeedLimit(0.5, false);
  controller->setSpeedLimit(1.0, true);
  controller->deactivate();
  controller->cleanup();
}

void exerciseStandstillPresteer(int obstacle_case)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("presteer_test_node");
  node->declare_parameter("controller_frequency", 20.0);
  node->declare_parameter("test.model_dt", 0.05);
  node->declare_parameter("test.motion_model", std::string("Ackermann"));
  node->declare_parameter("test.critics", std::vector<std::string>{});
  node->declare_parameter("test.iteration_count", 0);
  node->declare_parameter("test.enforce_path_inversion", true);
  node->declare_parameter("test.inversion_xy_tolerance", 0.20);
  node->declare_parameter("test.inversion_yaw_tolerance", 0.35);
  node->declare_parameter("test.enable_presteer_on_handoff", true);
  node->declare_parameter("test.presteer_creep_speed", 0.038);
  node->declare_parameter("test.presteer_timeout", 1.0);
  node->declare_parameter("test.presteer_angle_tolerance", 0.05);

  auto costmap_ros = getDummyCostmapRos();
  const double origin = obstacle_case == 2 ? 0.0 : -2.5;
  costmap_ros->getCostmap()->resizeMap(500, 500, 0.01, origin, origin);
  std::fill_n(costmap_ros->getCostmap()->getCharMap(), 250000, nav2_costmap_2d::FREE_SPACE);
  if (obstacle_case == 1 || obstacle_case == 3) {
    unsigned int mx, my;
    const double obstacle_x = obstacle_case == 1 ? 0.30 : 0.17;
    ASSERT_TRUE(costmap_ros->getCostmap()->worldToMap(obstacle_x, 0.0, mx, my));
    if (obstacle_case == 1) {
      costmap_ros->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    } else {
      // A wall outside the current body, but within the held-creep forecast.
      for (unsigned int y = 0; y < 500; ++y) {
        costmap_ros->getCostmap()->setCost(mx, y, nav2_costmap_2d::LETHAL_OBSTACLE);
      }
    }
  }
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  HandoffMemoryController controller;
  controller.configure(node, "test", tf, costmap_ros);
  controller.activate();

  // Provide initial steering feedback: front_steering_joint at -0.40 rad (-23 deg)
  auto feedback = std::make_shared<sensor_msgs::msg::JointState>();
  feedback->header.stamp = node->now();
  feedback->name = {"front_steering_joint"};
  feedback->position = {-0.40};
  controller.receiveSteeringFeedback(feedback);

  // Two-segment plan with a cusp at (0.30, 0.0):
  // Segment 1: Forward from (0,0) to (0.30, 0)
  // Segment 2: Reverse curving left from (0.30, 0) to (0.10, 0.15)
  nav_msgs::msg::Path plan;
  plan.header.frame_id = costmap_ros->getGlobalFrameID();
  plan.poses.push_back(getDummyPointStamped(node, TestPose{0.0, 0.0}));
  plan.poses.push_back(getDummyPointStamped(node, TestPose{0.30, 0.0}));
  // Reverse segment (poses going backwards with left turn)
  plan.poses.push_back(getDummyPointStamped(node, TestPose{0.20, 0.05}));
  plan.poses.push_back(getDummyPointStamped(node, TestPose{0.10, 0.12}));
  for (auto & pose : plan.poses) {pose.header = plan.header;}
  controller.setPlan(plan);

  // Robot reaches cusp at (0.30, 0.0)
  auto robot_pose = getDummyPointStamped(node, TestPose{0.30, 0.0});
  robot_pose.header = plan.header;
  auto velocity = getDummyTwist();

  const auto expect_presteer_veto = [&]() {
      try {
        controller.computeVelocityCommands(robot_pose, velocity, nullptr);
        ADD_FAILURE() << "Unsafe presteer command escaped final forecast validation";
      } catch (const nav2_core::NoValidControl & error) {
        EXPECT_NE(std::string(error.what()).find("Presteer command forecast is not collision-free"),
          std::string::npos);
        EXPECT_TRUE(controller.isPresteering());
      }
    };
  if (obstacle_case > 0 && obstacle_case < 4) {
    expect_presteer_veto();
    controller.deactivate();
    controller.cleanup();
    return;
  }

  // On first call reaching cusp: handoff fires and pre-steer engages because target steering != -0.40 rad
  auto cmd = controller.computeVelocityCommands(robot_pose, velocity, nullptr);
  EXPECT_TRUE(controller.isPresteering());
  // Presteer commands sub-deadband reverse creep (-0.038 m/s)
  EXPECT_NEAR(cmd.twist.linear.x, -0.038, 1e-4);
  EXPECT_NE(cmd.twist.angular.z, 0.0);
  if (obstacle_case == 4) {
    unsigned int mx, my;
    ASSERT_TRUE(costmap_ros->getCostmap()->worldToMap(0.30, 0.0, mx, my));
    costmap_ros->getCostmap()->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
    expect_presteer_veto();
    controller.deactivate();
    controller.cleanup();
    return;
  }

  // Update steering feedback to match target angle
  feedback->header.stamp = node->now();
  feedback->position = {controller.getPresteerTargetAngle()};
  controller.receiveSteeringFeedback(feedback);

  // On next cycle with steering aligned: pre-steer finishes and normal MPPI takes over
  EXPECT_NO_THROW(controller.computeVelocityCommands(robot_pose, velocity, nullptr));
  EXPECT_FALSE(controller.isPresteering());

  controller.deactivate();
  controller.cleanup();
}

TEST(ControllerStateTransitionTest, StandstillPresteerOnHandoff)
{
  exerciseStandstillPresteer(0);
}

TEST(ControllerStateTransitionTest, PresteerRejectsOccupiedCurrentFootprint)
{
  exerciseStandstillPresteer(1);
}

TEST(ControllerStateTransitionTest, PresteerRejectsOutOfMapFootprint)
{
  exerciseStandstillPresteer(2);
}

TEST(ControllerStateTransitionTest, PresteerRejectsCollisionInHeldCommandForecast)
{
  exerciseStandstillPresteer(3);
}

TEST(ControllerStateTransitionTest, PresteerRevalidatesEachReturnAgainstCurrentMap)
{
  exerciseStandstillPresteer(4);
}
