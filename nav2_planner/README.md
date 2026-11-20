# Nav2 Planner

The Nav2 planner is a Task Server in Nav2 that implements the `nav2_behavior_tree::ComputePathToPose` interface.

A planning module implementing the `nav2_behavior_tree::ComputePathToPose` interface is responsible for generating a feasible path given start and end robot poses. It loads a map of potential planner plugins to do the path generation in different user-defined situations.

See the [Navigation Plugin list](https://docs.nav2.org/plugins/index.html) for a list of the currently known and available planner plugins. 

See its [Configuration Guide Page](https://docs.nav2.org/configuration/packages/configuring-planner-server.html) for additional parameter descriptions and a [tutorial about writing planner plugins](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2planner_plugin.html).

## Fork parameters (jetacker)

| Parameter | Default | Scope | Meaning |
|---|---|---|---|
| `through_poses_start_at_requested_waypoint` | `false` | GENERIC | In `ComputePathThroughPoses`, start each leg after the first at the requested waypoint pose instead of the end of the path planned so far. Stock behaviour (false) tolerates planners that stop within goal tolerance, but the next leg inherits that end's floating-point pose; e.g. a heading on a Smac heading-bin edge then rounds to either bin. With true, the join is the requested pose; the path's last pose before the join is kept, so a planner that stopped short leaves that gap in the concatenated path. Switchable at runtime. |
