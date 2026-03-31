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

#include "nav2_mppi_controller/critic_manager.hpp"

namespace mppi
{

void CriticManager::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, ParametersHandler * param_handler)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  name_ = name;
  auto node = parent_.lock();
  logger_ = node->get_logger();
  parameters_handler_ = param_handler;

  getParams();
  loadCritics();
}

void CriticManager::getParams()
{
  auto node = parent_.lock();
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(critic_names_, "critics", std::vector<std::string>{}, ParameterType::Static);
  getParam(publish_critics_stats_, "publish_critics_stats", false, ParameterType::Static);
}

void CriticManager::loadCritics()
{
  if (!loader_) {
    loader_ = std::make_unique<pluginlib::ClassLoader<critics::CriticFunction>>(
      "nav2_mppi_controller", "mppi::critics::CriticFunction");
  }

  critics_.clear();
  for (auto name : critic_names_) {
    std::string fullname = getFullName(name);
    auto instance = std::unique_ptr<critics::CriticFunction>(
      loader_->createUnmanagedInstance(fullname));
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + name, costmap_ros_,
      parameters_handler_);
    RCLCPP_INFO(logger_, "Critic loaded : %s", fullname.c_str());
  }

  if (publish_critics_stats_) {
    auto node = parent_.lock();
    if (node) {
      critics_effect_pub_ = node->create_publisher<nav2_critics_msgs::msg::CriticsStats>(
        "~/critics_stats", rclcpp::QoS(10));
      RCLCPP_INFO(logger_, "Publishing per-critic cost stats to ~/critics_stats");
    }
  }
}

void CriticManager::on_activate()
{
  if (critics_effect_pub_) {
    critics_effect_pub_->on_activate();
  }
}

void CriticManager::on_deactivate()
{
  if (critics_effect_pub_) {
    critics_effect_pub_->on_deactivate();
  }
}

void CriticManager::on_cleanup()
{
  critics_effect_pub_.reset();
  critics_.clear();
}

std::string CriticManager::getFullName(const std::string & name)
{
  return "mppi::critics::" + name;
}

void CriticManager::evalTrajectoriesScores(
  CriticData & data) const
{
  std::unique_ptr<nav2_critics_msgs::msg::CriticsStats> stats_msg;
  // Per-critic cost deltas for each trajectory (to extract best trajectory costs)
  std::vector<xt::xtensor<float, 1>> per_critic_deltas;
  if (publish_critics_stats_) {
    stats_msg = std::make_unique<nav2_critics_msgs::msg::CriticsStats>();
    stats_msg->critics.reserve(critic_names_.size());
    stats_msg->changed.reserve(critic_names_.size());
    stats_msg->costs_sum.reserve(critic_names_.size());
    stats_msg->costs_best.reserve(critic_names_.size());
    per_critic_deltas.reserve(critic_names_.size());
  }

  for (size_t i = 0; i < critics_.size(); ++i) {
    if (data.fail_flag) {
      break;
    }

    xt::xtensor<float, 1> costs_before;
    if (publish_critics_stats_) {
      costs_before = data.costs;
    }

    critics_[i]->score(data);

    if (publish_critics_stats_) {
      auto delta = data.costs - costs_before;
      stats_msg->critics.push_back(critic_names_[i]);
      float costs_sum = static_cast<float>(xt::sum(delta)());
      stats_msg->costs_sum.push_back(costs_sum);
      stats_msg->changed.push_back(costs_sum != 0.0f);
      per_critic_deltas.push_back(std::move(delta));
    }
  }

  if (publish_critics_stats_ && critics_effect_pub_ && stats_msg) {
    // Find the best (lowest total cost) trajectory
    if (data.costs.size() > 0) {
      size_t best_idx = 0;
      float best_cost = data.costs(0);
      for (size_t k = 1; k < data.costs.size(); ++k) {
        if (data.costs(k) < best_cost) {
          best_cost = data.costs(k);
          best_idx = k;
        }
      }
      for (size_t i = 0; i < per_critic_deltas.size(); ++i) {
        stats_msg->costs_best.push_back(per_critic_deltas[i](best_idx));
      }
    }
    auto node = parent_.lock();
    if (node) {
      stats_msg->stamp = node->get_clock()->now();
      critics_effect_pub_->publish(std::move(stats_msg));
    }
  }
}

}  // namespace mppi
