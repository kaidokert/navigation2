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
#include "nav2_mppi_controller/critic_manager.hpp"
#include "nav2_mppi_controller/critics/path_align_critic.hpp"
#include "nav2_mppi_controller/critics/path_follow_critic.hpp"
#include "nav2_mppi_controller/critics/terminal_approach_critic.hpp"

// Tests critic manager

class RosLockGuard
{
public:
  RosLockGuard() {rclcpp::init(0, nullptr);}
  ~RosLockGuard() {rclcpp::shutdown();}
};
RosLockGuard g_rclcpp;

using namespace mppi;  // NOLINT
using namespace mppi::critics;  // NOLINT

class DummyCritic : public CriticFunction
{
public:
  virtual void initialize() {initialized_ = true;}
  virtual void score(CriticData & /*data*/) {scored_ = true;}
  bool initialized_{false}, scored_{false};
};

class CriticManagerWrapper : public CriticManager
{
public:
  CriticManagerWrapper()
  : CriticManager() {}

  virtual ~CriticManagerWrapper() = default;

  virtual void loadCritics()
  {
    critics_.clear();
    auto instance = std::unique_ptr<critics::CriticFunction>(new DummyCritic);
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + "DummyCritic", costmap_ros_,
      parameters_handler_);
  }

  std::string getFullNameWrapper(const std::string & name)
  {
    return getFullName(name);
  }

  bool getDummyCriticInitialized()
  {
    const auto critic = critics_[0].get();
    if (critic == nullptr) {
      return false;
    }
    const auto dummy_critic = dynamic_cast<DummyCritic *>(critic);
    return dummy_critic == nullptr ? false : dummy_critic->initialized_;
  }

  bool getDummyCriticScored()
  {
    const auto critic = critics_[0].get();
    if (critic == nullptr) {
      return false;
    }
    const auto dummy_critic = dynamic_cast<DummyCritic *>(critic);
    return dummy_critic == nullptr ? false : dummy_critic->scored_;
  }
};

class CriticManagerWrapperEnum : public CriticManager
{
public:
  CriticManagerWrapperEnum()
  : CriticManager() {}

  virtual ~CriticManagerWrapperEnum() = default;

  unsigned int getCriticNum()
  {
    return critics_.size();
  }

  const std::vector<std::string> & getCriticNames() const
  {
    return critic_names_;
  }

  critics::CriticFunction * getCritic(size_t idx) const
  {
    return idx < critics_.size() ? critics_[idx].get() : nullptr;
  }
};

TEST(CriticManagerTests, BasicCriticOperations)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State lstate;
  costmap_ros->on_configure(lstate);

  // Configuration should get parameters and initialize critic functions
  CriticManagerWrapper critic_manager;
  critic_manager.on_configure(node, "critic_manager", costmap_ros, &param_handler);
  EXPECT_TRUE(critic_manager.getDummyCriticInitialized());

  // Evaluation of critics should score them, but only if failure flag is not set
  models::State state;
  models::ControlSequence control_sequence;
  models::Trajectories generated_trajectories;
  models::Path path;
  geometry_msgs::msg::Pose goal;
  xt::xtensor<float, 1> costs;
  float model_dt = 0.1;
  CriticData data =
  {state, generated_trajectories, path, goal, costs, model_dt, false, nullptr, nullptr,
    std::nullopt, std::nullopt};

  data.fail_flag = true;
  EXPECT_FALSE(critic_manager.getDummyCriticScored());
  data.fail_flag = false;
  critic_manager.evalTrajectoriesScores(data);
  EXPECT_TRUE(critic_manager.getDummyCriticScored());

  // This should get the full namespaced name of the critics
  EXPECT_EQ(critic_manager.getFullNameWrapper("name"), std::string("mppi::critics::name"));
}

TEST(CriticManagerTests, CriticLoadingTest)
{
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("my_node");
  node->declare_parameter(
    "critic_manager.critics",
    rclcpp::ParameterValue(
      std::vector<std::string>{
    "PathFollowCritic", "PathAlignCritic", "TerminalApproachCritic"}));
  node->declare_parameter(
    "critic_manager.PathFollowCritic.threshold_to_consider", rclcpp::ParameterValue(0.35));
  node->declare_parameter(
    "critic_manager.PathAlignCritic.threshold_to_consider", rclcpp::ParameterValue(0.35));
  node->declare_parameter(
    "critic_manager.TerminalApproachCritic.enabled", rclcpp::ParameterValue(true));
  node->declare_parameter(
    "critic_manager.TerminalApproachCritic.approach_radius", rclcpp::ParameterValue(0.65));
  node->declare_parameter(
    "critic_manager.TerminalApproachCritic.cutoff_radius", rclcpp::ParameterValue(0.35));
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
    "dummy_costmap", "", "dummy_costmap", true);
  ParametersHandler param_handler(node);
  rclcpp_lifecycle::State state;
  costmap_ros->on_configure(state);
  auto * cm = costmap_ros->getCostmap();
  cm->setDefaultValue(nav2_costmap_2d::FREE_SPACE);
  cm->resetMap(0, 0, cm->getSizeInCellsX(), cm->getSizeInCellsY());

  // This should grab the critics parameter and load the 3 requested real plugins via pluginlib
  CriticManagerWrapperEnum critic_manager;
  critic_manager.on_configure(node, "critic_manager", costmap_ros, &param_handler);
  EXPECT_EQ(critic_manager.getCriticNum(), 3u);
  ASSERT_EQ(critic_manager.getCriticNames().size(), 3u);
  EXPECT_EQ(critic_manager.getCriticNames()[0], "PathFollowCritic");
  EXPECT_EQ(critic_manager.getCriticNames()[1], "PathAlignCritic");
  EXPECT_EQ(critic_manager.getCriticNames()[2], "TerminalApproachCritic");

  auto * pfc = dynamic_cast<critics::PathFollowCritic *>(critic_manager.getCritic(0));
  auto * pac = dynamic_cast<critics::PathAlignCritic *>(critic_manager.getCritic(1));
  auto * tac = dynamic_cast<critics::TerminalApproachCritic *>(critic_manager.getCritic(2));
  ASSERT_NE(pfc, nullptr);
  ASSERT_NE(pac, nullptr);
  ASSERT_NE(tac, nullptr);

  // Build real trajectories and path inside the 0.35 m cutoff (d_goal = 0.25 m) within the
  // [0, 5] x [0, 5] costmap and score through evalTrajectoriesScores() on the real loaded critics
  models::State st;
  st.reset(2, 10);
  st.pose.pose.position.x = 2.00;
  st.pose.pose.position.y = 2.00;
  st.pose.pose.orientation.w = 1.0;
  st.speed.linear.x = 0.15;
  models::Trajectories tr;
  tr.reset(2, 10);
  for (size_t k = 0; k < 10; ++k) {
    tr.x(0, k) = 2.00f + 0.02f * static_cast<float>(k);
    tr.y(0, k) = 2.00f;
    tr.yaws(0, k) = 0.0f;
    tr.x(1, k) = 2.00f + 0.01f * static_cast<float>(k);
    tr.y(1, k) = 2.08f;
    tr.yaws(1, k) = 0.25f;
  }
  models::Path mp;
  mp.reset(6);
  for (size_t i = 0; i < 6; ++i) {
    mp.x(i) = 2.00f + 0.05f * static_cast<float>(i);
    mp.y(i) = 2.00f;
    mp.yaws(i) = 0.0f;
  }
  geometry_msgs::msg::Pose goal;
  goal.position.x = 2.25;
  goal.position.y = 2.00;
  goal.orientation.w = 1.0;
  xt::xtensor<float, 1> costs = xt::zeros<float>({2});
  float model_dt = 0.05f;
  CriticData data = {
    st, tr, mp, goal, costs, model_dt, false, nullptr, nullptr, std::nullopt, std::nullopt};

  critic_manager.evalTrajectoriesScores(data);
  EXPECT_TRUE(tac->isLatched());
  EXPECT_EQ(
    tac->guidanceOwnership(),
    critics::GuidanceOwnership::TERMINAL_EXCLUSIVE);
  EXPECT_GT(costs(0) + costs(1), 0.0f);
  EXPECT_LT(costs(0), costs(1));
}
