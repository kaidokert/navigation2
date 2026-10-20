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

#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
namespace mppi
{
// Immutable source plan plus absolute real-point indices. Ghost extensions never
// receive IDs. The transform is the single snapshot used for this local path.
struct PathBinding
{
  std::shared_ptr<const nav_msgs::msg::Path> canonical_plan;
  uint64_t generation{0}, segment{0};
  size_t segment_begin{0}, segment_end{0};
  std::vector<size_t> real_ids;
  geometry_msgs::msg::TransformStamped canonical_to_local;
};
}  // namespace mppi
