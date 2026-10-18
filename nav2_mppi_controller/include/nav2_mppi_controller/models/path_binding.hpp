#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
namespace mppi {
// Immutable source plan plus absolute real-point indices. Ghost extensions never
// receive IDs. The transform is the single snapshot used for this local path.
struct PathBinding {
  std::shared_ptr<const nav_msgs::msg::Path> canonical_plan;
  uint64_t generation{0}, segment{0};
  size_t segment_begin{0}, segment_end{0};
  std::vector<size_t> real_ids;
  geometry_msgs::msg::TransformStamped canonical_to_local;
};
}
