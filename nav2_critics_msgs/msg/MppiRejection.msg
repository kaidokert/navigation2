# Full state of a rejected MPPI cycle, published when FollowPath.trace_rejections is true (default false).
builtin_interfaces/Time stamp
uint64 cycle                      # matches MppiCycle.cycle
# The controller's costmap at the check (copied under the costmap lock)
uint32 size_x
uint32 size_y
float64 resolution
float64 origin_x
float64 origin_y
uint8[] costs                     # row-major, size_x * size_y
uint64 costmap_hash
float64[] footprint_x             # padded footprint used by the check, robot frame
float64[] footprint_y
# Inputs
float64 pose_x
float64 pose_y
float64 pose_yaw
float64 speed_vx
float64 speed_wz
float64[] plan_x                  # transformed plan given to the optimizer
float64[] plan_y
float64[] plan_yaw
float64 goal_x
float64 goal_y
float64 goal_yaw
# The final check's forecast (index 0 = first predicted pose; the current pose is checked separately)
float64[] forecast_x
float64[] forecast_y
float64[] forecast_yaw
int32 first_collision_index
# Last optimisation iteration: every rollout
uint32 batch_size
uint32 time_steps
float32[] rollout_vx              # batch_size * time_steps, row-major, the sampled (constrained) controls
float32[] rollout_wz
string[] critic_names
float32[] critic_costs            # len(critic_names) * batch_size, row-major: each critic's cost per rollout
float32[] total_costs             # batch_size: critic + control costs used by the softmax
float32[] softmax_weights         # batch_size
int32[] rollout_final_check_first_collision   # batch_size: the production check on each rollout's controls (MppiCycle index convention: -2 passes, -4 not evaluable)
