// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_CONTROLLER__CONTROL_LOOP_CLOCK_HPP_
#define NAV2_CONTROLLER__CONTROL_LOOP_CLOCK_HPP_

#include <chrono>
#include <cstdint>
#include <thread>

namespace nav2_controller
{

// Wait in simulated time, but poll interruption in wall time. Clock::sleep_until
// alone can strand action cancellation while Gazebo's clock is paused.
// A backward jump releases the wait so the next cycle establishes a new epoch.
template<typename Now, typename Interrupted>
void waitForSimControlPeriod(
  int64_t cycle_start_ns, int64_t period_ns, Now now, Interrupted interrupted)
{
  int64_t previous_ns = cycle_start_ns;
  while (!interrupted()) {
    const int64_t current_ns = now();
    if (current_ns < previous_ns || current_ns - cycle_start_ns >= period_ns) {
      return;
    }
    previous_ns = current_ns;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace nav2_controller

#endif  // NAV2_CONTROLLER__CONTROL_LOOP_CLOCK_HPP_
