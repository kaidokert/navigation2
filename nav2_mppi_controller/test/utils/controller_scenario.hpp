// Copyright (c) 2026 kaidokert
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"
#include "nav_msgs/msg/path.hpp"

namespace controller_test
{
using Pose = std::array<double, 3>;
using Box = std::array<double, 5>;  // xmin, ymin, xmax, ymax, height

// These tests also build with production -ffast-math; do not let the compiler
// optimize away rejection of non-finite fixture input.
inline double number(const YAML::Node & value)
{
  const double result = value.as<double>();
  static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559);
  uint64_t bits;
  std::memcpy(&bits, &result, sizeof(bits));
  if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000)) {
    throw std::invalid_argument("Non-finite scenario coordinate");
  }
  return result;
}

template<size_t N>
std::array<double, N> numbers(const YAML::Node & value)
{
  if (!value.IsSequence() || value.size() != N) {
    throw std::invalid_argument("Invalid scenario coordinate dimensions");
  }
  std::array<double, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = number(value[i]);
  }
  return result;
}

inline double wrap(double yaw) {return std::atan2(std::sin(yaw), std::cos(yaw));}

struct Scenario
{
  std::string id;
  std::vector<Pose> path;
  std::vector<int> gears;  // One sign for each edge; a shared pose at each cusp.
  std::map<std::string, Pose> starts;
  std::vector<Box> obstacles;
  bool diagnostic_only;

  Scenario variant(bool mirror, bool reverse) const
  {
    Scenario result = *this;
    auto transform_pose = [mirror, reverse](Pose & p) {
        if (mirror) {
          p[1] = -p[1];
          p[2] = -p[2];
        }
        p[2] = wrap(p[2] + (reverse ? M_PI : 0.0));
      };
    for (auto & p : result.path) {transform_pose(p);}
    for (auto & entry : result.starts) {transform_pose(entry.second);}
    if (reverse) {
      for (auto & gear : result.gears) {gear = -gear;}
    }
    if (mirror) {
      for (auto & b : result.obstacles) {
        const double ymin = b[1];
        b[1] = -b[3];
        b[3] = -ymin;
      }
    }
    return result;
  }

  nav_msgs::msg::Path message() const
  {
    nav_msgs::msg::Path result;
    result.header.frame_id = "map";
    for (const auto & p : path) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = result.header;
      pose.pose.position.x = p[0];
      pose.pose.position.y = p[1];
      pose.pose.orientation.z = std::sin(p[2] / 2.0);
      pose.pose.orientation.w = std::cos(p[2] / 2.0);
      result.poses.push_back(pose);
    }
    return result;
  }
};

struct Atlas
{
  YAML::Node document;
  std::vector<Scenario> cases;

  explicit Atlas(const std::string & filename)
  : document(YAML::LoadFile(filename))
  {
    if (document["schema"].as<std::string>() != "nav2_controller_scenarios/v1" ||
      document["frame"].as<std::string>() != "map" ||
      document["reference_point"].as<std::string>() != "rear_axle")
    {
      throw std::invalid_argument("Unsupported controller scenario schema/frame");
    }
    std::set<std::string> ids;
    for (const auto & value : document["cases"]) {
      Scenario scenario;
      scenario.id = value["id"].as<std::string>();
      if (!ids.insert(scenario.id).second) {throw std::invalid_argument("Duplicate scenario");}
      for (const auto & p : value["path"]) {scenario.path.push_back(numbers<3>(p));}
      scenario.gears = value["gears"].as<std::vector<int>>();
      if (scenario.path.size() < 2 || scenario.gears.size() + 1 != scenario.path.size()) {
        throw std::invalid_argument("A scenario needs one gear per path edge");
      }
      for (size_t i = 0; i < scenario.gears.size(); ++i) {
        const auto & a = scenario.path[i];
        const auto & b = scenario.path[i + 1];
        const double dx = b[0] - a[0], dy = b[1] - a[1];
        const int gear = scenario.gears[i];
        if ((gear != 1 && gear != -1) || std::hypot(dx, dy) < 1e-8 ||
          gear * (dx * std::cos(a[2]) + dy * std::sin(a[2])) <= 0.0)
        {
          throw std::invalid_argument("Duplicate pose or gear/body-yaw disagreement");
        }
      }
      for (const auto & entry : value["starts"]) {
        scenario.starts.emplace(entry.first.as<std::string>(), numbers<3>(entry.second));
      }
      if (scenario.starts.empty()) {throw std::invalid_argument("Missing scenario starts");}
      for (const auto & b : value["obstacles"]) {
        const auto box = numbers<5>(b);
        if (box[0] >= box[2] || box[1] >= box[3] || box[4] <= 0) {
          throw std::invalid_argument("Invalid obstacle box");
        }
        scenario.obstacles.push_back(box);
      }
      scenario.diagnostic_only = value["diagnostic_only"].as<bool>();
      cases.push_back(scenario);
    }
    if (cases.empty()) {throw std::invalid_argument("Empty scenario atlas");}
  }

  const Scenario & at(const std::string & id) const
  {
    for (const auto & scenario : cases) {
      if (scenario.id == id) {return scenario;}
    }
    throw std::out_of_range("Unknown controller scenario: " + id);
  }
};
}  // namespace controller_test
