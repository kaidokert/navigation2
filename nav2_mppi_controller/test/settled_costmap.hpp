// Test fixture: wait for native executor startup and perform lifecycle cleanup.
#ifndef OFFICE_TEST_SETTLED_COSTMAP_HPP_
#define OFFICE_TEST_SETTLED_COSTMAP_HPP_
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
#endif
