# Controller timing instrumentation

Status: stationary ARM validation passed; review/CI and sim vetting remain.
No robot deployment or driving qualification.

Enable `controller_server.ros__parameters.timing_enabled: true` before configure.
Default is false. Changing the parameter requires cleanup/configure; it is not a
runtime tuning control. The instrumentation changes no frequency, planner/controller
parameters, affinity, priority, trajectory or guard decision.

## Contract

`/controller_server/control_timing` is a `std_msgs/msg/String` containing JSON schema
1. A normal ROS wall timer on the existing executor emits one message per second,
including zero-sample idle windows. Executor scheduling can delay publication: use
`window_seconds`, not an assumed exact second. `monotonic_seconds` identifies the
end of the aggregation window on this boot. Bag receipt timestamps provide ROS
correlation; steady-clock values are not ROS timestamps.

Each action-loop iteration contributes one row, including iterations that cancel
or throw. `completed`, `cancelled`, `inactive`, `exceptions` separate those outcomes; no action
means zero samples. Nav2 behavior-server recoveries have no controller samples;
FollowPath retries are active controller work. This topic alone does not identify
behavior-tree recovery states: correlate the action and campaign trace.

Configured-but-deactivated nodes also publish empty idle windows; cleanup stops
the publisher and timer.

The row's elapsed and thread-CPU clocks stop before rate sleep. `deadline_misses`
counts elapsed computation/wait time exceeding the configured control period;
`deadline_miss_fraction` divides by all sampled iterations. `scheduler_misses`
separately counts failed WallRate sleeps, which can include scheduling lateness.
The budget is refreshed at each action under the controller parameter lock.
Each sample uses its own captured budget. A window spanning different budgets
reports `deadline_budget_seconds: null` plus min/max budgets; miss counts remain
computed against each sample's actual period.
The terminal successful cycle has no rate sleep. Exceptions/cancellation also
have no scheduler-miss check, but retain elapsed measurements.

All metric values are seconds. Each metric carries its own valid sample count.
p50/p95/p99 are nearest-rank quantiles, max is the observed maximum. Empty metrics
have no fabricated quantiles. Fields:

- elapsed: whole iteration before scheduled sleep, including costmap readiness,
  command calculation/publishing, goal checking, and intervening overhead.
- thread_cpu: CLOCK_THREAD_CPUTIME_ID for that same interval; excludes descheduled
  time and CPU consumed by other threads.
- publication_interval: interval between completed compute/publish calls within
  one action execution; reset between actions and on preemption. This is not DDS receipt latency.
- costmap_ready_wait: waiting for the local costmap to become current.
- path_tf: MPPI goal/path transformation, including its path-handler processing.
- global_path_update: controller-side updateGlobalPath, separate from MPPI path/TF.
- costmap_lock_wait: MPPI acquisition of the local costmap mutex.
- parameter_lock_wait: MPPI acquisition of its parameter mutex.
- optimizer: evalControl, including scoring and final output collision validation.
- visualization: optional trajectory visualization work.
- compute_publish: controller computeAndPublishVelocity, including pose lookup,
  progress checks, MPPI stages and command publishing.

Stages are nested, not additive: compute_publish contains MPPI stages. Exceptions
retain only stages that completed. Stage hooks use thread-local state defined in
nav2_util_core so the plugin contributes to the enclosing controller iteration;
standalone plugin calls without a controller timing scope publish nothing.

Storage is bounded to 1,024 rows/window. Overflow keeps counts and reports
`dropped_samples`, while quantiles cover only retained rows. No worker pool or
per-cycle instrumentation logging is added. Two preallocated row buffers swap pointers under the collection lock; only
counters are reset between windows, with no row copying or clearing under that lock.
Serialization and sorting happen on the existing timer executor outside the lock.

## Thread identity

Linux thread names are bounded to 15 characters. The controller worker is
`nav-control`, the costmap update worker
`nav-map-update`, its callback executor `nav-map-exec`, and the action callback
executor `nav-action-exec`. NodeThread names node executors from the node name; generic
executor threads are `nav-executor`. Each naming change logs a single
`[nav-thread] name=... tid=...` mapping. DDS retains its middleware names.
The main thread/process keeps its original name, preserving pgrep/pkill behavior.
Worker naming is always on (including other servers using NodeThread); control
worker TID logging is enabled only with timing. Other startup naming logs are once
per created worker.
Timing records carry the last contributing control worker's Linux TID. IDs are
namespace-relative: correlate inside the same container/PID namespace.

## Comparison procedure

1. Keep the production frequency/MPPI parameters unchanged. Record exact source,
   configuration, hardware, thermal state and run conditions.
2. Compare instrumentation disabled/enabled on identical isolated inputs, with
   warmup excluded and alternating order. Report CPU and elapsed distributions;
   measurement noise must not be presented as zero overhead.
3. Record a 20 Hz baseline with timing enabled. Compare active-cycle statistics,
   command cadence and outcomes, excluding idle time from the denominator.
4. Only subsequently change affinity, locally on the robot, retaining instrumentation
   and identical controller configuration. Distinguish whole-container affinity
   from reserving a core for the control worker.
5. A 10 Hz trial is a separate configuration: model_dt must be compatible (0.10 s);
   about 21 steps preserves the current ~2.05 s horizon. It requires independent
   driving/geometry qualification and is not enabled by this PR.

Dry runs on robot hardware must be in a network-isolated container with no hardware
or host IPC mounts and an isolated ROS domain. Never launch test action clients
into the production DDS domain. Synthetic state cannot establish driving quality.

## ARM stationary validation (2026-10-27 UTC)

The robot build of nav2_util, nav2_costmap_2d, nav2_controller and MPPI passed.
Collector tests passed on ARM. An isolated controller-server run with the robot
parameter overlay and synthetic stationary TF/odom/free map completed 57 control
cycles, then cancelled cleanly, emitted zero, and produced an empty idle window.
All named workers appeared in the thread mapping and native MPPI stages appeared
in the controller timing topic. The container exited cleanly; no hardware devices,
host network or host IPC were exposed.

Three ABBA overhead repetitions used 1,000 candidates × 41 steps, Ackermann,
PathFollow/PathAlign/Obstacles critics and rectangular hard-footprint checks.
Each block ran 110 calls paced at 50 ms, discarding 10 warmup samples. Every enabled
block received at least three aggregate messages, so aggregation overhead is
included. Corresponding velocity commands matched the disabled reference exactly.

| Repetition | Disabled median | Enabled median | Difference | p95 difference |
|---|---:|---:|---:|---:|
| 1 | 15.533 ms | 15.592 ms | +0.059 ms (+0.38%) | +0.534 ms |
| 2 | 15.476 ms | 15.624 ms | +0.149 ms (+0.96%) | +0.181 ms |
| 3 | 15.585 ms | 15.671 ms | +0.086 ms (+0.55%) | +0.129 ms |

These are observed elapsed differences under shared-host scheduling, not precise
attribution of every microsecond to instrumentation. The overhead is below 1% at
the median in this fixture; this does not establish driving stability or overnight
behavior. JSON receipts include p99/max and the exact fixture limits. The first
short benchmark did not exercise the timer and is not used as overhead evidence;
a subsequent fixture setup failure (missing configured polygon) was corrected
before these runs. The first six-second stationary test hit the existing progress
watchdog; the final three-second test explicitly verifies cancellation instead.

Do not average per-second quantiles to claim an exact whole-run percentile.
Compare window distributions, or calculate command-cadence percentiles from the
recorded commands when a whole-run distribution is required.

Post-review validation also checks mixed-budget windows, inactive/cancellation/exception
classification, exact window-level counts, finite budgets and publication intervals.
The MCAP smoke decoded five timing messages, including the cancelled cycle and idle
window. A networked container was rejected before ROS initialization.
