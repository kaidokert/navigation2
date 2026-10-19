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

#ifndef NAV2_SMAC_PLANNER__SUCCESSFUL_PLAN_SNAPSHOT_HPP_
#define NAV2_SMAC_PLANNER__SUCCESSFUL_PLAN_SNAPSHOT_HPP_
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav_msgs/msg/path.hpp"

namespace nav2_smac_planner
{
// Caller holds the same costmap lock as the search. Optional, bounded evidence;
// never used to accept/reject a plan or alter collision checking.
inline bool writeSuccessfulPlanSnapshot(
  const char * prefix, size_t sequence, int64_t stamp_ns,
  const nav2_costmap_2d::Costmap2D & grid, const nav_msgs::msg::Path & plan,
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  const std::vector<geometry_msgs::msg::Point> & footprint,
  unsigned int bins, bool use_radius, bool allow_unknown)
{
  if (!prefix || !*prefix) {return false;}
  const std::string target = std::string(prefix) + ".plan_" +
    std::to_string(sequence % 64) + ".json";
  const std::string temporary = target + ".tmp";
  std::ofstream out(temporary);
  if (!out) {return false;}
  out << std::setprecision(17)
      << "{\"scope\":\"Successful returned plan and exact locked search grid; diagnostic only\","
      << "\"sequence\":" << sequence << ",\"stamp_ns\":" << stamp_ns
      << ",\"frame\":" << std::quoted(plan.header.frame_id)
      << ",\"plan_header_stamp\":[" << plan.header.stamp.sec << ','
      << plan.header.stamp.nanosec << "],\"angle_bins\":" << bins
      << ",\"use_radius\":" << (use_radius ? "true" : "false")
      << ",\"allow_unknown\":" << (allow_unknown ? "true" : "false");
  const auto pose = [&out](const auto & p) {
      out << '[' << p.position.x << ',' << p.position.y << ','
          << p.orientation.x << ',' << p.orientation.y << ','
          << p.orientation.z << ',' << p.orientation.w << ']';
    };
  out << ",\"start_world_xy_quaternion\":"; pose(start.pose);
  out << ",\"goal_world_xy_quaternion\":"; pose(goal.pose);
  out << ",\"plan_world_xy_quaternion\":[";
  for (size_t i = 0; i < plan.poses.size(); ++i) {
    if (i) {out << ',';}
    pose(plan.poses[i].pose);
  }
  out << "],\"footprint\":[";
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
  if (out.fail()) {std::remove(temporary.c_str()); return false;}
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    std::remove(temporary.c_str()); return false;
  }
  return true;
}
}  // namespace nav2_smac_planner
#endif  // NAV2_SMAC_PLANNER__SUCCESSFUL_PLAN_SNAPSHOT_HPP_
