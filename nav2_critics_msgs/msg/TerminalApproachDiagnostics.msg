# Versioned observation of the production critic, never an input to control.
uint16 schema_version 1
std_msgs/Header header
string critic
# Instance is unique across configure cycles; counters are monotonic within it.
string instance_id
uint64 sequence
uint64 evaluation_id
uint64 task_id
uint64 plan_id
uint64 goal_id
uint64 latch_id
# EVALUATED, ABORT, CURVE_CLEARED, RESET. ABORT includes the original exception.
string event
string reason
string exception
# False on lifecycle/plan/parameter events: robot_pose/goal/velocity are unavailable.
bool evaluation_context_valid
bool feasibility_evaluated
bool reuse_obstacles_cache
geometry_msgs/Pose robot_pose
geometry_msgs/Pose goal
geometry_msgs/Twist velocity
builtin_interfaces/Time pose_stamp
bool goal_tolerance_valid
float64 goal_xy_tolerance
float64 goal_yaw_tolerance
float64 min_turning_radius
float64 goal_region_xy_ratio
float64 goal_region_yaw_ratio
float64 goal_distance
float64 goal_yaw_error
string policy
uint8 ownership
bool latched
bool latch_consumed
bool reference_path
bool cusp_braking
bool reversal_established
bool replan_pending
bool replan_allowance_consumed
int32 redraw_count
int32 max_redraws
int32 replan_count
float64 gear
float64 lateral_error
float64 lateral_limit
float64 rollback
float64 cutoff
# Snapshot immediately before clearing if an evaluation invalidated its latch.
bool before_clear
# Full curve at latch/change/abort/clear. Empty on unchanged evaluations.
uint32 curve_size
bool curve_truncated
string curve_frame
geometry_msgs/Pose2D[<=512] curve
# Observed predicates/components in the decision stages reached on this evaluation.
bool checks_truncated
TerminalApproachCheck[<=96] checks
# NOT_EVALUATED, REACHED_INTERIOR, INVALID_INPUT, CAPTURE_ANGLE, RESOLUTION,
# SAMPLE_STEP, SAMPLE_BUDGET, FOOTPRINT_BLOCKED, LATERAL_ENVELOPE, MISSED_INTERIOR.
string goal_region_result
bool witness_sample_valid
bool witness_lateral_evaluated
uint32 witness_sample_index
uint32 witness_sample_count
geometry_msgs/Pose2D witness_pose
float64 witness_xy_error
float64 witness_yaw_error
float64 witness_lateral
float64 witness_lateral_limit
# Last rejected footprint in this evaluation (may belong to a rejected candidate).
bool collision_check_failed
string collision_reason
string collision_frame
geometry_msgs/Pose2D collision_pose
float64 collision_cost
float64 costmap_resolution
float64 costmap_origin_x
float64 costmap_origin_y
uint32 costmap_size_x
uint32 costmap_size_y
