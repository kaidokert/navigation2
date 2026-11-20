# Nav2 Planner

The Nav2 planner is a Task Server in Nav2 that implements the `nav2_behavior_tree::ComputePathToPose` interface.

A planning module implementing the `nav2_behavior_tree::ComputePathToPose` interface is responsible for generating a feasible path given start and end robot poses. It loads a map of potential planner plugins to do the path generation in different user-defined situations.

See the [Navigation Plugin list](https://docs.nav2.org/plugins/index.html) for a list of the currently known and available planner plugins. 

See its [Configuration Guide Page](https://docs.nav2.org/configuration/packages/configuring-planner-server.html) for additional parameter descriptions and a [tutorial about writing planner plugins](https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2planner_plugin.html).

## Fork parameters (jetacker)

| Parameter | Default | Scope | Meaning |
|---|---|---|---|
| `through_poses_start_at_requested_waypoint` | `false` | GENERIC | In `ComputePathThroughPoses`, start each leg after the first at the previous leg's goal (in the global frame) instead of the end of the path planned so far, when that path end lies within the two tolerances below. Stock behaviour (false, or a path that stopped further away) starts at the path end, which carries the previous leg's floating-point end pose; e.g. a heading on a Smac heading-bin edge then rounds to either bin. The concatenation still drops each later leg's first pose, so the output path runs from the previous leg's end to the next leg's second pose. Switchable at runtime. |
| `through_poses_waypoint_start_max_dist` | `0.01` | GENERIC | Largest distance (m) between the path end and the previous leg's goal for the waypoint start. |
| `through_poses_waypoint_start_max_yaw` | `0.01` | GENERIC | Largest heading difference (rad) between the path end and the previous leg's goal for the waypoint start. |
