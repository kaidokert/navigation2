# Standalone robot collision regression

This is the collision-prevention subset of kaidokert/navigation2#12, based on the
robot's deployed commit `7112c45db7b7531793e6064c8ef73e3764a7ea68`. It does not
include #11's planner policy changes, path-critic changes, or #14's final-output
validator.

## Robot configuration after installing the patched library

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      ObstaclesCritic:
        consider_footprint: true
        collision_check_footprint_only: true
```

The second setting enables unconditional footprint collision checks even where
the reference-point inflation cost is below the circumscribed-radius threshold.
It defaults to false, so existing configurations must explicitly enable it.
The setting is read-only after critic configuration: live activation and
deactivation are rejected. Change YAML/CLI and restart the controller to change modes. Startup overrides and predeclared values are preserved.
With `consider_footprint: false`, hard footprint rejection can be enabled while
retaining point-based soft clearance scoring. A polygon footprint is required.

All forecast poses are checked for collision. The former
`goal_truncation_distance` parameter is accepted with a warning but does not skip
collision checks. Entering the goal radius does not prove a predicted tail will
stop before a wall. A forecast that actually remains stopped at a clear goal is
accepted; one continuing into the wall is rejected. This intentionally replaces
the old `ObstaclesCriticGoalTruncation` expectation and resolves the contradictory
terminal-tail expectations in #12.

## Regression scope

Tests compile the actual production `obstacles_critic.cpp` and parameter handler,
not a reimplementation. One frozen geometry reduction comes from physical robot
bag `run_20261018_195510`, output at Unix timestamp `1792353513.3058`:

- map: 60 x 60, 0.05000000074505806 m/cell;
- origin: (-1.1000000003725297, -1.4500000003725289);
- forecast pose: (0.5028052926, 0.4304492474, -2.0596189499);
- lethal cell: (30, 42);
- padded footprint: front 0.182 m, rear 0.168 m, half width 0.163 m.

The centre cost is zero, but Nav2's rasterized footprint touches the lethal cell.
The regression explicitly reproduces the miss with the new hard check disabled,
and requires detection with it enabled in either soft-scoring mode. Other cases
cover near-goal entry, a post-goal lethal tail, reverse forecasts, all-colliding
fallback, a clear stopping forecast, point-soft-cost preservation, radius-only
misconfiguration, and a footprint extending outside the map.

Run in a ROS 2 Jazzy environment with Nav2 costmap/util/core, xtensor, xsimd, and
ament_cmake_gtest installed:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S nav2_mppi_controller/test/collision_regression -B /tmp/collision-build
cmake --build /tmp/collision-build --parallel 2
ctest --test-dir /tmp/collision-build --output-on-failure
```

The same test source is registered in the normal MPPI package's CMake tests.
The standalone target avoids requiring this fork's external
`nav2_critics_msgs` telemetry package just to compile and run collision tests.

This is an incremental critic fix. It does not validate the final averaged and
filtered MPPI command sequence, fill the polygon interior, certify continuous
clearance between trajectory samples, or implement an actuator watchdog.
Runtime CPU cost and full-robot return behavior still require robot validation.
