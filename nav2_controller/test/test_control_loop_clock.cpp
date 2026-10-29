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

#include <atomic>
#include <chrono>
#include <future>

#include "gtest/gtest.h"
#include "nav2_controller/control_loop_clock.hpp"

using namespace std::chrono_literals;

TEST(ControlLoopClock, SlowSimulationMustAdvanceFullModelPeriod)
{
  std::atomic<int64_t> clock_ns{0};
  std::atomic<bool> interrupted{false};
  auto waiter = std::async(std::launch::async, [&]() {
        nav2_controller::waitForSimControlPeriod(
        0, 50000000, [&]() {return clock_ns.load();}, [&]() {return interrupted.load();});
    });
  // Wall time alone cannot consume a prediction step.
  EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);
  clock_ns = 16000000;
  EXPECT_EQ(waiter.wait_for(10ms), std::future_status::timeout);
  clock_ns = 48000000;
  EXPECT_EQ(waiter.wait_for(10ms), std::future_status::timeout);
  clock_ns = 50000000;
  EXPECT_EQ(waiter.wait_for(200ms), std::future_status::ready);
  interrupted = true;  // Ensure test cleanup cannot hang on a regression.
  waiter.get();
}

TEST(ControlLoopClock, CancellationOrDeactivationInterruptsPausedClock)
{
  std::atomic<bool> interrupted{false};
  auto waiter = std::async(std::launch::async, [&]() {
        nav2_controller::waitForSimControlPeriod(
        100000000, 50000000, []() {return 100000000;}, [&]() {return interrupted.load();});
    });
  EXPECT_EQ(waiter.wait_for(10ms), std::future_status::timeout);
  interrupted = true;
  EXPECT_EQ(waiter.wait_for(200ms), std::future_status::ready);
  waiter.get();
}

TEST(ControlLoopClock, BackwardJumpStartsNewEpochWithoutWaitingForOldDeadline)
{
  nav2_controller::waitForSimControlPeriod(
    1000000000, 50000000, []() {return 100000000;}, []() {return false;});
}

TEST(ControlLoopClock, OverrunDoesNotAddAnotherPeriod)
{
  nav2_controller::waitForSimControlPeriod(
    100000000, 50000000, []() {return 160000000;}, []() {return false;});
}

TEST(ControlLoopClock, BackwardJumpWithinCycleAlsoReleasesWait)
{
  const int64_t readings[] = {100000000, 140000000, 120000000, 160000000};
  size_t next = 0;
  nav2_controller::waitForSimControlPeriod(
    100000000, 50000000, [&]() {return readings[next++];}, [&]() {return next == 4;});
  EXPECT_EQ(next, 3u);
}
