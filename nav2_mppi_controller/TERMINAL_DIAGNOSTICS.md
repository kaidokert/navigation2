# Terminal approach diagnostics

`TerminalApproachCritic` publishes
`nav2_critics_msgs/msg/TerminalApproachDiagnostics` on the controller node's private
`~/terminal_approach_diagnostics` topic (normally
`/controller_server/terminal_approach_diagnostics`). This shared implementation is
used on the robot and in simulation. It does not change scoring, tolerances,
collision checks, latch budgets, or the default-off setting.

## Recording contract

The publisher is lifecycle-managed, reliable, volatile, with depth 100. Connect a
recorder **before** starting the attempt. Record the topic with the action result,
run ID, source/build revisions and live parameters. A topic's existence alone is
not evidence that it was recorded. This PR supplies the publisher, not a campaign
exporter or Optuna integration.

An enabled critic publishes one `EVALUATED` message per `score()` invocation, or
one `ABORT` message when that invocation throws a standard exception. The latter
is published before rethrowing, even when normal `CriticsStats` never publishes.
Multiple optimizer iterations can produce multiple evaluations per control tick;
`feasibility_evaluated` and `reuse_obstacles_cache` distinguish fresh decisions
from cached scoring. A dormant enabled critic still emits evaluations. A disabled
critic emits no evaluations; there is no claim to observe failures elsewhere in
the planner, controller or external guards.

`instance_id` changes on configure. `sequence` orders publications within that
instance. `evaluation_id` counts enabled score calls, `task_id` advances at reset,
`plan_id` at every incoming plan and `goal_id` at a changed global goal.
`latch_id` advances whenever a curve is latched or redrawn. These are local IDs,
not the NavigateToPose action UUID. Join them to the campaign's attempt interval
and record the topic namespace when running parallel workers.

Require a stream over the attempt and an `ABORT` for any critic exception in the
controller result. Check sequence continuity. Missing events, a disconnected
recorder, truncation, or a process crash make diagnostic coverage incomplete;
never score missing diagnostics as “no failure.” Reliable DDS is not durable
storage and cannot guarantee delivery after a process crash. Publication errors
are logged without replacing the original controller outcome.

## Reading a decision

`reason` is the terminal decision (for example `CURVE_UNJOINABLE`,
`DRIFT_EXCEEDED`, `ENTRY_INFEASIBLE`, or `STOCK_THRESHOLD_MISMATCH`);
`exception` preserves the original exception text. `checks` contains observed
predicates and components from decision stages reached in that evaluation, with
their booleans and numerical operands. Components of a composite condition can
be reported even when that condition short-circuits; this is not an instruction
trace. Names ending
in `le`, `lt`, `ge`, or `gt` give the comparison direction. Composite predicates
use their `passed` field; NaN operands mean no scalar comparison applies.
Candidate `capture/<family>/<reason>` entries describe the solver's existing
geometric-family trace; `accepted` there means geometric admission, not final
selection. Subsequent collision/endpoint checks may still reject that candidate.
The observer does not invoke a second solve.

For a finite-curve join refusal, `join/runway_le` gives required versus available
distance. `goal_region_result` then distinguishes an unexecuted witness from an
interior arrival, invalid inputs, angular limit, invalid map resolution, invalid
sample step, exhausted sample budget, blocked footprint, lateral-envelope
violation, or an arc that never reaches the interior target. The last tested
witness sample and its errors are included; use the validity flags. Live goal
tolerances and interior ratios accompany them. A failed sufficient witness does
not prove that every possible physical finish is impossible.

`before_clear` marks the snapshot taken immediately before latch invalidation.
Its latch/ownership/error/limit/budget fields describe that instant, not the
cleared state after the exception. Curve geometry is included on latch, redraw,
frame adjustment, abort, clear and reset. Unchanged successful evaluations omit
it. At most 512 points and 96 checks are sent; truncation is explicit. A
`CURVE_CLEARED` or `RESET` outside scoring has
`evaluation_context_valid=false`: pose/velocity/check fields must not be treated
as a fresh observation. Link its curve and IDs to the preceding evaluation.

`header.stamp` is publication time on the node's ROS clock; `pose_stamp` is the
input pose stamp. `header.frame_id` identifies the score input frame;
`curve_frame` and `collision_frame` independently identify those geometries.
All lengths are metres, angles radians, velocities SI. The last rejected
footprint and costmap geometry are included when available. That footprint can
belong to a rejected candidate, so it is not automatically the abort's cause.

This stream explains the executed critic decision. Exact offline replay and
physical attribution additionally require the plan, transforms, raw costmap plus
updates, footprint/configuration, actuator state and odometry/ground truth at the
same times. It contains neither full costmaps nor all MPPI rollouts.
