# Shared controller scenarios

`approach_atlas.json` is the canonical geometry input for native tests and the
robot repository's existing `drive_nav2.py` FollowPath runner. It is installed to
`share/nav2_mppi_controller/controller_test_fixtures/`. Consumers must record the
file's SHA-256 and the native revision; do not maintain another hand-copied atlas.

The geometry is a reduction of the [approach robustness atlas](http://prometheus-box.tailfa2b6e.ts.net:5244/artifacts/robots/sim/2026-11-09/approach-robustness/design_01/).
It is not a replay of the office bags and does not establish a controller fix.

## Contract

- All positions are rear-axle positions, in metres in a fixed `map` frame; yaw is
  **body** heading in radians, including reverse segments.
- `path` is the immutable ordered reference. `starts` contains independent robot
  poses. Selecting an offset start never translates or replans the reference.
- `gears` has one `+1`/`-1` entry per path edge. A cusp is one shared pose, not a
  duplicated zero-length edge. `m8_full` is one 32-pose F–R–F path (cusps at
  indices 8 and 25), supplied to one FollowPath action.
- Mirror transforms `(x,y,yaw)` to `(x,-y,-yaw)` for reference, robot and boxes.
  Reverse adds pi to body yaw and negates gear, retaining spatial order. The
  asymmetric body footprint rotates: clearance must be checked again.
- Boxes are `[xmin,ymin,xmax,ymax,height]`. The native grid marks every cell whose
  interior intersects a box. Native tests use lethal cells without inflation;
  this is not a substitute for live costmap/sensor testing.
- The footprint is already padded. `limits` describes the design geometry and
  eventual acceptance limits; it must not silently override robot settings.
- References retain endpoints and curvature transitions at approximately 5 cm
  spacing. Pruning probes explicitly vary spacing and phase. Dense plotting
  polylines must never replace the reference supplied to the critics.

## Cases

| ID | Purpose |
| --- | --- |
| `curved_capture` | Nominal, inside and outside starts on a terminal right arc |
| `short_cusp_exit` | M8 final forward arc with nominal and heading-error starts |
| `early_lateral_capture` | Lateral error with enough longitudinal setup distance |
| `late_lateral_capture` | Diagnostic short-runway stress, without a feasibility claim |
| `wall_capture` | Capture beside a box, with clear/colliding footprint controls |
| `m8_full` | Full forward/reverse/forward reference, retaining state at cusps |
| `pruning_<spacing>_<phase>` | Nine representations of a curved terminal reference |

## Native tests

Run the existing package tests:

```sh
colcon build --packages-select nav2_mppi_controller --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select nav2_mppi_controller --ctest-args -R controller_scenario_tests
```

The existing `test/collision_regression` CMake build and Collision regression CI
also run this target. They compile the real critics, optimizer and PathHandler.
The standalone build needs an installed MPPI package's pluginlib metadata, but
binds the critics compiled from the source checkout explicitly; it does not score
through the installed critic library.

Tests cover geometry/sign contracts, actual PathAlign position/heading response,
hard obstacle rejection, sequential optimizer memory and production M8 path
handoffs. The optimizer uses a fixed balanced 64-row noise bank, three iterations,
41 x 50 ms horizon, and R_min=0.558 m. These are a small deterministic test profile,
not a claim to reproduce the robot's complete production configuration.

The memory test compares carried state with deliberately erased tape/history.
Prescribed poses keep this a decision test: it does not predict a physical endpoint.
Characterization exercises can remain green while tracking remains poor. Closed-loop
qualification and any controller-default change require separate evidence.
