# Smac analytic-expansion trace

The Hybrid Smac planner publishes one JSON array per completed planning request on the lifecycle topic `/<namespace>/analytic_expansion_trace`; at the root namespace this is `/analytic_expansion_trace`.

Enable it with the planner plugin parameter:

```yaml
planner_server:
  ros__parameters:
    GridBased:
      publish_analytic_expansion_trace: true
```

The default is `false`. This is a configure-time parameter: attempts to change it while the planner is active are rejected, so a successful parameter readback cannot falsely imply that tracing was enabled. Record the topic in the same MCAP as `/plan`, planner action/status, and the costmap/map inputs.

The array starts and ends with `plan_boundary` records carrying a request ID. The ending record gives the result (`succeeded`, `failed`, or `exception`), the number of visited-node opportunities to attempt analytic expansion, the number skipped by the configured cadence, and the number of candidate attempts. Candidate and scoring records carry that request ID and a monotonically increasing attempt ID. Buffering keeps DDS publication to one message per request and also covers same-cell success and exceptions before search begins. Trace serialization still adds CPU work; do not compare traced and untraced planning latency as equivalent.

A `candidate` record includes the start and goal in search-grid coordinates, state-space distance, turning radius in grid cells, OMPL segment types and normalized signed lengths (for Reeds–Shepp) or lengths (Dubins), interpolation sample count, valid sample count, and outcome. `valid_sample_count` is the number of samples that passed node validity. Outcomes distinguish the distance gate, node lookup failure, invalid sampled node, max-cost rejection, a normal accepted candidate, and acceptance by the max-cost override. When a sample causes rejection, the record includes its index, grid pose, and measured cost; cost is `null` when no current sample cost was available. A `score` record gives the traversal score for an evaluated candidate. The final `selection` record identifies the attempt whose path was retained.

## Interpretation limits

OMPL's Reeds–Shepp API returns one shortest path word for the supplied endpoint states. This trace records that returned word and whether Smac accepted and selected it. It does **not** enumerate all Reeds–Shepp words that OMPL could generate, nor report words that OMPL internally considered and discarded. To answer that question, OMPL itself would need instrumentation. The plan-boundary counters distinguish “no analytic attempt was scheduled” from “an attempt ran and failed its gates.”

A `valid` candidate means the interpolated samples passed the current node-validity and analytic max-cost checks. It does not mean the full continuous swept footprint was collision-free between samples. The existing sampling resolution and collision-check semantics still apply.
