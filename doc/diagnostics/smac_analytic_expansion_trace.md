# Smac analytic-expansion trace

The Hybrid Smac planner can publish JSON records on the lifecycle topic `/<planner_server_namespace>/analytic_expansion_trace` (for the usual root namespace, `/planner_server/analytic_expansion_trace`).

Enable it with the planner plugin parameter:

```yaml
planner_server:
  ros__parameters:
    GridBased:
      publish_analytic_expansion_trace: true
```

The default is `false`; when disabled, the planner does not publish trace records. Record the topic in the same MCAP as `/plan`, planner action/status, and the costmap/map inputs.

Each `plan_boundary` record has a request ID, a result (`started`, `succeeded`, `failed`, or `exception`), the number of visited-node opportunities to attempt analytic expansion, the number skipped by the configured analytic-expansion cadence, and the number of candidate attempts. Candidate and scoring records carry that request ID and a monotonically increasing attempt ID.

A `candidate` record includes the start and goal in search-grid coordinates, state-space distance, turning radius in grid cells, OMPL segment types and normalized signed lengths (for Reeds–Shepp) or lengths (Dubins), interpolation sample count, valid sample count, and outcome. Outcomes distinguish the max-length/distance gate, node lookup failure, invalid sampled node, max-cost rejection, and a candidate that passed validation. When a sample causes rejection, the record includes that sample's index, grid pose, and cost. A `score` record gives the traversal score for an evaluated candidate. The final `selection` record identifies the attempt whose path was retained.

## Interpretation limits

OMPL's Reeds–Shepp API returns one shortest path word for the supplied endpoint states. This trace records that returned word and whether Smac accepted and selected it. It does **not** enumerate all Reeds–Shepp words that OMPL could generate, nor report words that OMPL internally considered and discarded. To answer that question, OMPL itself would need instrumentation. The plan-boundary counters distinguish “no analytic attempt was scheduled” from “an attempt ran and failed its gates.”

A `valid` candidate means the interpolated samples passed the current node-validity and analytic max-cost checks. It does not mean the full continuous swept footprint was collision-free between samples. The existing sampling resolution and collision-check semantics still apply.
