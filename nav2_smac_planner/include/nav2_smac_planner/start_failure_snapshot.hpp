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

// Diagnostic only: serialize a failed search's locked grid.
#ifndef NAV2_SMAC_PLANNER__START_FAILURE_SNAPSHOT_HPP_
#define NAV2_SMAC_PLANNER__START_FAILURE_SNAPSHOT_HPP_
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_smac_planner/types.hpp"
namespace nav2_smac_planner
{
inline bool writeStartFailureSnapshot(
  const char * path, const nav2_costmap_2d::Costmap2D & grid,
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  float mx, float my, unsigned int start_bin, unsigned int bins,
  bool start_collision, bool allow_unknown,
  const std::vector<geometry_msgs::msg::Point> & footprint, const SearchInfo & search,
  int iterations = 1, int64_t capture_stamp_ns = 0)
{
  if (!path || !*path) {return false;}
  std::ofstream out(path);
  if (!out) {return false;}
  out << std::setprecision(17)
      << "{\"scope\":\"Exact locked grid after failed search; no control or search change\","
      << "\"start_native_collision\":" << (start_collision ? "true" : "false")
      << ",\"search_iterations\":" << iterations
      << ",\"capture_stamp_ns\":" << capture_stamp_ns
      << ",\"allow_unknown\":" << (allow_unknown ? "true" : "false")
      << ",\"start_grid\":[" << mx << ',' << my << ',' << start_bin << ']'
      << ",\"angle_bins\":" << bins
      << ",\"start_header_stamp\":[" << start.header.stamp.sec << ','
      << start.header.stamp.nanosec << ']';
  const auto pose = [&out](const char * name, const auto & p) {
      out << ",\"" << name << "\":[" << p.pose.position.x << ',' << p.pose.position.y
          << ',' << p.pose.orientation.x << ',' << p.pose.orientation.y << ','
          << p.pose.orientation.z << ',' << p.pose.orientation.w << ']';
    };
  pose("start_world_xy_quaternion", start); pose("goal_world_xy_quaternion", goal);
  out << ",\"search\":{\"minimum_turning_radius_cells\":" << search.minimum_turning_radius
      << ",\"reverse_penalty\":" << search.reverse_penalty
      << ",\"change_penalty\":" << search.change_penalty
      << ",\"non_straight_penalty\":" << search.non_straight_penalty
      << ",\"cost_penalty\":" << search.cost_penalty << '}';
  out << ",\"footprint\":[";
  for (size_t i = 0; i < footprint.size(); ++i) {
    if (i) {out << ',';}
    out << '[' << footprint[i].x << ',' << footprint[i].y << ']';
  }
  out << "],\"grid\":{\"width\":" << grid.getSizeInCellsX()
      << ",\"height\":" << grid.getSizeInCellsY()
      << ",\"resolution\":" << grid.getResolution()
      << ",\"origin\":[" << grid.getOriginX() << ',' << grid.getOriginY() << "],\"data\":[";
  for (unsigned int y = 0; y < grid.getSizeInCellsY(); ++y) {
    for (unsigned int x = 0; x < grid.getSizeInCellsX(); ++x) {
      if (x || y) {out << ',';}
      out << static_cast<unsigned int>(grid.getCost(x, y));
    }
  }
  out << "]}}\n";
  out.close();
  return !out.fail();
}
}  // namespace nav2_smac_planner
#endif  // NAV2_SMAC_PLANNER__START_FAILURE_SNAPSHOT_HPP_
