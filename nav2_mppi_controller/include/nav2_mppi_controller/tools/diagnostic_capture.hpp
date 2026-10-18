// Copyright 2026 Kaido Kert
// Licensed under the Apache License, Version 2.0.
#ifndef NAV2_MPPI_CONTROLLER__TOOLS__DIAGNOSTIC_CAPTURE_HPP_
#define NAV2_MPPI_CONTROLLER__TOOLS__DIAGNOSTIC_CAPTURE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace mppi::diagnostic
{
inline bool isFiniteCaptureValue(double value)
{
  std::uint64_t bits;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

inline bool successfulBatchCaptureEligible(
  double stamp, size_t real_points, int direction, std::optional<double> minimum_stamp)
{
  return minimum_stamp && isFiniteCaptureValue(stamp) && stamp >= *minimum_stamp &&
         real_points >= 20 && direction == 1;
}
// Missing option preserves the original first-failure capture. Malformed options
// disable optional capture rather than throwing through the driving controller.
inline std::optional<double> minimumCaptureStamp(const char * text)
{
  if (!text) {return -std::numeric_limits<double>::infinity();}
  char * end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !isFiniteCaptureValue(value) || value < 0.0) {
    return std::nullopt;
  }
  return value;
}

inline bool failedBatchCaptureEligible(
  double stamp, double linear, double angular, bool stationary_only,
  std::optional<double> minimum_stamp)
{
  return minimum_stamp && isFiniteCaptureValue(stamp) && stamp >= *minimum_stamp &&
         (!stationary_only || (std::abs(linear) < 0.005 && std::abs(angular) < 0.05));
}
// Fixed disk bound: eight recent failed batches, at most one per simulated
// second. Disabled by default. A clock reset starts a new cadence window.
class FailedBatchRing
{
public:
  bool eligible(bool enabled, double stamp) const
  {
    // This project uses fast-math; std::isfinite can be optimized away.
    return enabled && isFiniteCaptureValue(stamp) &&
           (!last_stamp_ || stamp < *last_stamp_ || stamp - *last_stamp_ >= 1.0);
  }

  unsigned int take(double stamp)
  {
    last_stamp_ = stamp;
    const auto slot = next_slot_;
    next_slot_ = (next_slot_ + 1) % 8;
    return slot;
  }

private:
  std::optional<double> last_stamp_;
  unsigned int next_slot_{0};
};
}  // namespace mppi::diagnostic
#endif  // NAV2_MPPI_CONTROLLER__TOOLS__DIAGNOSTIC_CAPTURE_HPP_
