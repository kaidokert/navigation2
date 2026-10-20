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

// Test fixture: wait for native executor startup and perform lifecycle cleanup.
#ifndef SETTLED_COSTMAP_HPP_
#define SETTLED_COSTMAP_HPP_
#include <chrono>
#include <stdexcept>
#include <thread>
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

class SettledTestCostmap : public nav2_costmap_2d::Costmap2DROS
{
public:
  using nav2_costmap_2d::Costmap2DROS::Costmap2DROS;
  nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override
  {
    const auto result = Costmap2DROS::on_configure(state);
    configured_ = result == nav2_util::CallbackReturn::SUCCESS;
    if (configured_) {
      // NodeThread cancels then joins. Destroying it before spin() starts can
      // lose cancellation when spin() subsequently sets its spinning flag.
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!executor_->is_spinning()) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("Native costmap executor did not start in two seconds");
        }
        std::this_thread::yield();
      }
    }
    return result;
  }
  nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override
  {
    configured_ = false;
    return Costmap2DROS::on_cleanup(state);
  }
  ~SettledTestCostmap() override
  {
    if (configured_) {on_cleanup(rclcpp_lifecycle::State{});}
  }

private:
  bool configured_{false};
};
#endif  // SETTLED_COSTMAP_HPP_
