# Final MPPI output collision guard

Extracted from kaidokert/navigation2 PR #14 at
`c4aa45e0c17f974a1dbc3e4ca7c313b6851ce6ef`, onto `jetacker` after PR #15.
This is a selective backport, not a cherry-pick of the entire PR.

MPPI scores sampled rollouts before computing their weighted mean and applying
Savitzky–Golay smoothing. Neither operation preserves collision feasibility:
two clear curves can average into a straight path through an obstacle. The
optimizer now reapplies its final constraints and checks the actual emitted
sequence after smoothing, immediately before converting it to a command.

The mandatory check uses the measured pose and velocity, the real motion model,
and the local costmap. It checks the current footprint and the entire final
forecast. Sequence shifting starts the forecast with index 1 when index 1 is the
emitted command. An extra state column includes the last command, which the
model's initial measured-velocity column otherwise displaces. Rectangular
footprints use the native footprint collision checker at every predicted pose;
radius mode uses the inflated centre cost. Unknown/out-of-map poses and invalid
numeric inputs are rejected. IEEE exponent checks remain effective under the
package's production `-ffast-math` compiler flags.

A failed check clears the carried control sequence and smoothing history, then
throws `nav2_core::NoValidControl`. ControllerServer handles this through its
existing zero-command and action failure/retry path. An accepted command is the
last command stored in smoothing history. No velocity adjustment follows the
check. There is no new parameter or opt-out.

This does not import PR #14's cusp binding, motor deadband floor, ranked-sample
fallback, or other maneuver changes. It does not replace hardware command
timeouts, stall detection, costmap freshness checks, or the critic fix in #15.
The native checker tests polygon perimeter cells and the reference centre at
discrete forecast poses; it is not continuous swept-volume or filled-polygon
validation. An already occupied footprint is rejected and requires a separate
bounded recovery. Physical robot timings and driving remain to be qualified.

## Native regression tests

`test/final_output_collision_tests.cpp` exercises the production optimizer:
clear forward/reverse and holonomic motion, colliding weighted means and
smoothing, shifted emission, the final forecast slot, measured-state braking,
free-centre corner collisions (including the recorded table-leg fixture),
lethal centres, unknown/map boundaries, invalid numbers under fast-math,
malformed arrays, Ackermann constraints, and history reset/update behavior.

The standalone CI target compiles the production optimizer, critic manager,
noise generator, parameter handler and obstacle critic. `nav2_critics_msgs` is
the existing robot telemetry contract, copied unchanged from the robot stack;
it allows this native build without substituting a mock optimizer or message.
The normal MPPI test target also registers the new tests.

Reproduce from the repository root inside ROS Jazzy with the workflow's native
dependencies installed:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base /tmp/msg-log build --packages-select nav2_critics_msgs \
  --build-base /tmp/msg-build --install-base /tmp/msg-install
source /tmp/msg-install/setup.bash
cmake -S nav2_mppi_controller/test/collision_regression -B /tmp/collision-build
cmake --build /tmp/collision-build --parallel 2
ctest --test-dir /tmp/collision-build --output-on-failure
```
