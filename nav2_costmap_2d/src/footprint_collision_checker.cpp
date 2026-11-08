// Copyright (c) 2019 Intel Corporation
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
//
// Modified by: Shivang Patel (shivaang14@gmail.com)

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "nav2_costmap_2d/footprint_collision_checker.hpp"

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/exceptions.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_util/line_iterator.hpp"

using namespace std::chrono_literals;

namespace nav2_costmap_2d
{

template<typename CostmapT>
FootprintCollisionChecker<CostmapT>::FootprintCollisionChecker()
: costmap_(nullptr)
{
}

template<typename CostmapT>
FootprintCollisionChecker<CostmapT>::FootprintCollisionChecker(
  CostmapT costmap)
: costmap_(costmap)
{
}

/**
 * @brief Rasterize a polygon footprint into the costmap and return the maximum edge cost
 * @param footprint Oriented polygon vertices in world coordinates
 * @return Maximum cost along the footprint perimeter, or LETHAL_OBSTACLE if out of bounds
 */
template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::footprintCost(const Footprint & footprint)
{
  if (footprint.empty()) {
    return 0.0;
  }

  // now we really have to lay down the footprint in the costmap_ grid
  unsigned int x0, x1, y0, y1;
  double footprint_cost = 0.0;

  // get the cell coord of the first point
  if (!worldToMap(footprint[0].x, footprint[0].y, x0, y0)) {
    return static_cast<double>(LETHAL_OBSTACLE);
  }

  if (footprint.size() == 1u) {
    return pointCost(x0, y0);
  }

  // cache the start to eliminate a worldToMap call
  unsigned int xstart = x0;
  unsigned int ystart = y0;

  // we need to rasterize each line in the footprint
  for (unsigned int i = 0; i < footprint.size() - 1; ++i) {
    // get the cell coord of the second point
    if (!worldToMap(footprint[i + 1].x, footprint[i + 1].y, x1, y1)) {
      return static_cast<double>(LETHAL_OBSTACLE);
    }

    footprint_cost = std::max(lineCost(x0, x1, y0, y1), footprint_cost);

    // the second point is next iteration's first point
    x0 = x1;
    y0 = y1;

    // if in collision, no need to continue
    if (footprint_cost == static_cast<double>(LETHAL_OBSTACLE)) {
      return footprint_cost;
    }
  }

  // we also need to connect the first point in the footprint to the last point
  // the last iteration's x1, y1 are the last footprint point's coordinates
  return std::max(lineCost(xstart, x1, ystart, y1), footprint_cost);
}

/**
 * @brief Compute the maximum cost along a rasterized grid line segment
 * @param x0 Starting cell X index
 * @param x1 Ending cell X index
 * @param y0 Starting cell Y index
 * @param y1 Ending cell Y index
 * @return Maximum cell cost along the segment, early-exiting on LETHAL_OBSTACLE
 */
template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::lineCost(int x0, int x1, int y0, int y1) const
{
  const unsigned char * const char_map = costmap_->getCharMap();
  const unsigned int size_x = costmap_->getSizeInCellsX();
  unsigned char line_cost = 0;

  for (nav2_util::LineIterator line(x0, y0, x1, y1); line.isValid(); line.advance()) {
    const unsigned char point_cost =
      char_map[static_cast<size_t>(line.getY()) * size_x +
        static_cast<size_t>(line.getX())];

    // if in collision, no need to continue
    if (point_cost == LETHAL_OBSTACLE) {
      return static_cast<double>(LETHAL_OBSTACLE);
    }

    if (line_cost < point_cost) {
      line_cost = point_cost;
    }
  }

  return static_cast<double>(line_cost);
}

template<typename CostmapT>
bool FootprintCollisionChecker<CostmapT>::worldToMap(
  double wx, double wy, unsigned int & mx, unsigned int & my)
{
  return costmap_->worldToMap(wx, wy, mx, my);
}

template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::pointCost(int x, int y) const
{
  return static_cast<double>(costmap_->getCost(x, y));
}

template<typename CostmapT>
void FootprintCollisionChecker<CostmapT>::setCostmap(CostmapT costmap)
{
  costmap_ = costmap;
}

/**
 * @brief Transform an unoriented footprint to a pose and compute its maximum perimeter cost
 * @param x Robot pose X in world coordinates
 * @param y Robot pose Y in world coordinates
 * @param theta Robot heading in radians
 * @param footprint Unoriented polygon footprint relative to the robot base
 * @return Maximum cost along the transformed footprint perimeter
 */
template<typename CostmapT>
double FootprintCollisionChecker<CostmapT>::footprintCostAtPose(
  double x, double y, double theta, const Footprint & footprint)
{
  if (footprint.empty()) {
    return 0.0;
  }

  const double cos_th = cos(theta);
  const double sin_th = sin(theta);

  unsigned int x0, x1, y0, y1;
  double footprint_cost = 0.0;

  const double wx0 = x + (footprint[0].x * cos_th - footprint[0].y * sin_th);
  const double wy0 = y + (footprint[0].x * sin_th + footprint[0].y * cos_th);
  if (!worldToMap(wx0, wy0, x0, y0)) {
    return static_cast<double>(LETHAL_OBSTACLE);
  }

  if (footprint.size() == 1u) {
    return pointCost(x0, y0);
  }

  const unsigned int xstart = x0;
  const unsigned int ystart = y0;

  for (unsigned int i = 0; i < footprint.size() - 1; ++i) {
    const double wx1 = x + (footprint[i + 1].x * cos_th - footprint[i + 1].y * sin_th);
    const double wy1 = y + (footprint[i + 1].x * sin_th + footprint[i + 1].y * cos_th);
    if (!worldToMap(wx1, wy1, x1, y1)) {
      return static_cast<double>(LETHAL_OBSTACLE);
    }

    footprint_cost = std::max(lineCost(x0, x1, y0, y1), footprint_cost);
    x0 = x1;
    y0 = y1;

    if (footprint_cost == static_cast<double>(LETHAL_OBSTACLE)) {
      return footprint_cost;
    }
  }

  return std::max(lineCost(xstart, x1, ystart, y1), footprint_cost);
}

// declare our valid template parameters
template class FootprintCollisionChecker<std::shared_ptr<nav2_costmap_2d::Costmap2D>>;
template class FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>;

}  // namespace nav2_costmap_2d
