// Copyright 2026 Jetacker contributors
// SPDX-License-Identifier: Apache-2.0
#include "nav2_util/control_timing.hpp"
namespace nav2_util
{
ControlTiming::Cycle * & ControlTiming::currentCycle()
{
  static thread_local Cycle * cycle = nullptr;
  return cycle;
}
void ControlTiming::stageCurrent(Stage index, Clock::time_point start)
{
  if (currentCycle()) {currentCycle()->stage(index, start);}
}
}  // namespace nav2_util
