# One MPPI control cycle, published by the optimizer when FollowPath.trace_cycles is true (default false).
# Diagnostic only: it reads the cycle's state, it never changes it.
# Lateral (vy) sequences and speed are not recorded: complete for non-holonomic models only.
builtin_interfaces/Time stamp     # node clock at the end of the cycle
uint64 cycle                      # optimizer cycle counter (matches MppiRejection.cycle)
# Inputs actually used by this cycle
float64 pose_x
float64 pose_y
float64 pose_yaw
float64 speed_vx
float64 speed_wz
builtin_interfaces/Time plan_stamp
uint32 plan_size
uint64 costmap_hash               # FNV-1a 64 over size, resolution, origin and every cell of the controller's costmap
float64 costmap_origin_x
float64 costmap_origin_y
# Control sequences (time_steps entries each)
float32[] warm_vx                 # sequence entering the cycle (warm start)
float32[] warm_wz
float32[] opt_vx                  # after optimisation (softmax mean of the last iteration, then constraints)
float32[] opt_wz
float32[] smooth_vx               # after the Savitzky-Golay filter
float32[] smooth_wz
float32[] final_vx                # after the final constraint pass: the sequence the final check tests
float32[] final_wz
# Outcome
uint8 RESULT_ACCEPTED = 0
uint8 RESULT_REJECTED_COLLISION = 1   # final check failed ("Final MPPI control sequence is not collision-free")
uint8 RESULT_REJECTED_NONFINITE = 2
uint8 RESULT_ALL_ROLLOUTS_COLLIDE = 3 # fallback exhausted ("Optimizer fail to compute path")
uint8 RESULT_ACCEPTED_FALLBACK = 4    # final check rejected the mean; a sampled rollout passed the same
                                      # check and was emitted (final_check_fallback_rollouts > 0).
                                      # final_* and the collision fields describe the rejected mean.
uint8 result
float64 cmd_vx                    # command returned (0 when rejected; the emitted rollout's for RESULT_ACCEPTED_FALLBACK)
float64 cmd_wz
# Final check detail (the production check: shifted when shift_control_sequence, measured speed first)
int32 first_collision_index       # -2 passes, -1 current pose, 0..time_steps forecast index,
                                  # -4 rejected before any pose was tested (non-finite input,
                                  # degenerate pose, footprint/radius unusable)
float64 collision_cell_x          # world centre of the first lethal / unknown / off-map cell, NaN if none
float64 collision_cell_y
uint8 collision_cell_cost
# The same check applied to earlier stages, computed every cycle (side-effect free).
# Same index convention; -3 = stage not reached (e.g. all rollouts collided).
int32 opt_unshifted_first_collision     # optimised mean, no shift: the rollout convention MPPI scored
int32 opt_shifted_first_collision       # optimised mean, production shift
int32 smooth_unshifted_first_collision  # smoothed mean, no shift
int32 smooth_shifted_first_collision    # smoothed mean, production shift, before the final constraint pass
# Batch summary of the last iteration
uint32 rollouts
# The four fields below need trace_rejections (rollout costs); else -1 / -3.
int32 rollouts_clear_unshifted          # dumped cycles only (rejected, fallback or collapse), else -1: rollouts passing the check without the shift
int32 rollouts_pass_final_check         # dumped cycles only (rejected, fallback or collapse), else -1: rollouts passing the production (shifted) check
int32 best_passing_rollout_rank         # dumped cycles only (rejected, fallback or collapse), else -1: rank by total cost of the cheapest passing rollout
int32 cheapest_rollout_final_collision  # dumped cycles only (rejected, fallback or collapse), else -3: production-check first collision of the cheapest rollout
# Final-check fallback (final_check_fallback_rollouts > 0)
int32 fallback_rollout_rank             # RESULT_ACCEPTED_FALLBACK: cost rank of the emitted rollout (0 = cheapest), else -1
