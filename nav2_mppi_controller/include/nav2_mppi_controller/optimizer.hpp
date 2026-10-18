// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_MPPI_CONTROLLER__OPTIMIZER_HPP_
#define NAV2_MPPI_CONTROLLER__OPTIMIZER_HPP_

#include <string>
#include <memory>
#include <optional>
#include <utility>
#include <cmath>

// xtensor creates warnings that needs to be ignored as we are building with -Werror
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#pragma GCC diagnostic pop

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_core/controller_exceptions.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include "nav2_mppi_controller/models/optimizer_settings.hpp"
#include "nav2_mppi_controller/models/path_binding.hpp"
#include "nav2_mppi_controller/motion_models.hpp"
#include "nav2_mppi_controller/critic_manager.hpp"
#include "nav2_mppi_controller/models/state.hpp"
#include "nav2_mppi_controller/models/trajectories.hpp"
#include "nav2_mppi_controller/models/path.hpp"
#include "nav2_mppi_controller/tools/noise_generator.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

namespace mppi
{

/**
 * @class mppi::Optimizer
 * @brief Main algorithm optimizer of the MPPI Controller
 */
class Optimizer
{
public:
  /**
    * @brief Constructor for mppi::Optimizer
    */
  Optimizer() = default;

  /** Bounded diagnostic departure forecast; never changes optimizer memory or emits controls. */
  bool longCuspDepartureAvailable(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
    const nav_msgs::msg::Path & outgoing, double xy_gate, double yaw_gate,
    bool terminal_endpoint = false) const;

  /** Check the actual final command during a retained reverse approach.
   * Caller holds the costmap lock. Forecast only; never modifies controls.
   */
  bool retainedDepartureCommandSafe(
    const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
    const geometry_msgs::msg::Twist & command,
    const geometry_msgs::msg::PoseStamped & cusp, double stop_margin) const;

  /**
   * @brief Destructor for mppi::Optimizer
   */
  ~Optimizer() {shutdown();}


  /**
   * @brief Initializes optimizer on startup
   * @param parent WeakPtr to node
   * @param name Name of plugin
   * @param costmap_ros Costmap2DROS object of environment
   * @param dynamic_parameter_handler Parameter handler object
   */
  void initialize(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    ParametersHandler * dynamic_parameters_handler);

  /**
   * @brief Shutdown for optimizer at process end
   */
  void shutdown();

  /**
   * @brief Activate optimizer (activate critic manager publishers)
   */
  void on_activate();

  /**
   * @brief Deactivate optimizer (deactivate critic manager publishers)
   */
  void on_deactivate();

  /** Real path prefix, excluding virtual preview points, for position scoring. */
  void setPathBinding(const std::optional<PathBinding> & binding)
  {path_binding_required_=true;path_binding_=binding;}
  void setRealPathPointCount(size_t count) {critics_data_.real_path_point_count = count;}
  /** Effective internal gates supplied by the path handler, not final-goal tolerances. */
  void setCuspTolerances(std::optional<std::pair<double, double>> gates)
  {
    if (gates && (!(gates->first > 0.0) || !(gates->second > 0.0) ||
      !std::isfinite(gates->first) || !std::isfinite(gates->second)))
    {
      throw nav2_core::InvalidPath("Invalid effective internal cusp tolerances");
    }
    effective_cusp_tolerances_ = gates;
  }
  /** Request one optional mean initialization after an actual path handoff. */
  void requestShortCuspSeed() {
    clearCuspCompletion();
    short_cusp_seed_requested_ = true;
    long_entry_seed_requested_ = true;
  }
  void requestPlanEntrySeed() {
    clearCuspCompletion();
    if (initialize_long_cusp_entry_) {short_cusp_seed_requested_ = true;}
  }
  /** Unpruned active cusp segment gear; zero for an unconstrained final goal. */
  /** Keep final segment gear separate from active-cusp critic metadata. */
  void setPathDirection(int direction) {path_direction_ = direction;}

  void setActiveCuspDirection(int direction)
  {
    if (direction != active_cusp_direction_ || direction == 0) {clearCuspCompletion();}
    active_cusp_direction_ = direction;
    critics_data_.active_cusp_direction = direction;
    if (direction == 0) {effective_cusp_tolerances_.reset();}
  }

  bool requiresSteeringFeedback() const
  {
    auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    return model && model->predictsSteeringConditioning();
  }

  void setInitialSteering(const std::optional<float> & angle)
  {
    auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    // Controller refresh provisionally clears then reinstalls both measurements.
    // Validate the completed refresh in prepare/final emission, not midway through it.
    if (model) {model->setInitialSteering(angle);}
  }

  std::optional<float> getInitialSteering() const
  {
    auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    return model ? model->initialSteering() : std::nullopt;
  }

  bool requiresConditionedSteering() const
  {
    auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    return model && model->requiresConditionedSteering();
  }

  void setInitialConditionedSteering(const std::optional<float> & angle)
  {
    auto model = std::dynamic_pointer_cast<AckermannMotionModel>(motion_model_);
    if (model) {model->setInitialConditionedSteering(angle);}
  }


  /**
   * @brief Compute control using MPPI algorithm
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   * @return TwistStamped of the MPPI control
   */
  geometry_msgs::msg::TwistStamped evalControl(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed, const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::Pose & goal, nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Get the trajectories generated in a cycle for visualization
   * @return Set of trajectories evaluated in cycle
   */
  models::Trajectories & getGeneratedTrajectories();

  /**
   * @brief Get the optimal trajectory for a cycle for visualization
   * @return Optimal trajectory
   */
  xt::xtensor<float, 2> getOptimizedTrajectory();

  /**
   * @brief Set the maximum speed based on the speed limits callback
   * @param speed_limit Limit of the speed for use
   * @param percentage Whether the speed limit is absolute or relative
   */
  void setSpeedLimit(double speed_limit, bool percentage);

  /**
   * @brief Reset the optimization problem to initial conditions
   * @param Whether to reset the constraints to its base values
   */
  void reset(bool reset_dynamic_speed_limits = true);

protected:
  std::optional<std::pair<double, double>> effective_cusp_tolerances_;
  std::pair<double, double> cuspTolerances() const;
  std::string serializedCuspTolerances() const;
  /**
   * @brief Main function to generate, score, and return trajectories
   */
  void optimize(bool diag = false);

  /**
   * @brief Prepare state information on new request for trajectory rollouts
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   */
  void prepare(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    const nav_msgs::msg::Path & plan,
    const geometry_msgs::msg::Pose & goal, nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Obtain the main controller's parameters
   */
  void getParams();

  /**
   * @brief Set the motion model of the vehicle platform
   * @param model Model string to use
   */
  void setMotionModel(const std::string & model);

  /**
   * @brief Shift the optimal control sequence after processing for
   * next iterations initial conditions after execution
   */
  void shiftControlSequence();

  /**
   * @brief updates generated trajectories with noised trajectories
   * from the last cycle's optimal control
   */
  void generateNoisedTrajectories();

  // Keep bounded short-cusp sampling and mean controls in the same gear domain.
  int shortCuspControlDirection() const;
  void constrainShortCuspSampleDirections();

  /**
   * @brief Apply hard vehicle constraints on control sequence
   */
  void applyControlSequenceConstraints();

  /**
   * @brief  Update velocities in state
   * @param state fill state with velocities on each step
   */
  void updateStateVelocities(models::State & state) const;

  /**
   * @brief  Update initial velocity in state
   * @param state fill state
   */
  void updateInitialStateVelocities(models::State & state) const;

  /**
   * @brief predict velocities in state using model
   * for time horizon equal to timesteps
   * @param state fill state
   */
  void propagateStateVelocitiesFromInitials(models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    models::Trajectories & trajectories,
    const models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    xt::xtensor<float, 2> & trajectories,
    const xt::xtensor<float, 2> & state) const;

  /**
   * @brief Update control sequence with state controls weighted by costs
   * using softmax function
   */
  void updateControlSequence(bool diag = false);

  /// Experimental conditioning on the gear of real path geometry; default off.
  bool conditionControlExponents(xt::xtensor<float, 1> & exponents);

  /**
   * @brief Convert control sequence to a twist commant
   * @param stamp Timestamp to use
   * @return TwistStamped of command to send to robot base
   */
  geometry_msgs::msg::TwistStamped
  getControlFromSequenceAsTwist(const builtin_interfaces::msg::Time & stamp);
    geometry_msgs::msg::TwistStamped getControlFromSequenceAsTwist(
      const builtin_interfaces::msg::Time & stamp, const models::ControlSequence * warm_start);
  geometry_msgs::msg::TwistStamped getControlFromSequenceAsTwist(
    const builtin_interfaces::msg::Time & stamp, const models::ControlSequence * warm_start,
    const models::ControlSequence * before_filter);

  /**
   * @brief Check the actual emitted control sequence using the measured state.
   */
  bool controlSequenceCollisionFree() const;

  /**
   * @brief Apply final constraints and veto colliding output before conversion.
   */
  geometry_msgs::msg::TwistStamped
  getValidatedControlFromSequenceAsTwist(
    const builtin_interfaces::msg::Time & stamp,
    const models::ControlSequence * warm_start = nullptr,
    const models::ControlSequence * before_filter = nullptr);

  bool controlSequenceFinite() const;

  /**
   * @brief Whether the motion model is holonomic
   * @return Bool if holonomic to populate `y` axis of state
   */
  bool isHolonomic() const;

  /**
   * @brief Using control frequence and time step size, determine if trajectory
   * offset should be used to populate initial state of the next cycle
   */
  void setOffset(double controller_frequency);

  /**
   * @brief Perform fallback behavior to try to recover from a set of trajectories in collision
   * @param fail Whether the system failed to recover from
   */
  bool fallback(bool fail);
  bool fallback(bool fail, const models::ControlSequence * warm_start);

  /// Optional diagnostic isolation of the existing control-sequence filter.
  void smoothControlSequence();

  /// Screen finite arc means with the same native predictor, critics and grid.
  bool initializeShortCuspMean();
  bool initializeLongCuspPrefixMean();

  /// Opt-in check of the actual constrained mean, after optimization/filtering.
  void enforceShortCuspMeanFeasibility();
    void enforceShortCuspMeanFeasibility(const models::ControlSequence * warm_start);
  bool shortCuspSequenceFeasible() const;
  bool cuspSequenceFeasible(const models::ControlSequence & sequence) const;
  bool actualCuspStreamFeasible(const models::ControlSequence & sequence) const;
  void enforceGentleCuspEmission();
  bool gentle_short_cusp_required_{false};
  void clearCuspCompletion() {
    gentle_short_cusp_required_ = false;
    clearLongEntryCertificate();
    cusp_completion_.reset();
    cusp_completion_attempted_ = false;
    cusp_completion_shifts_ = 0;
  }
  // Separate local-entry proof: never uses the distant path endpoint.
  struct LongEntryCertificate {
    models::ControlSequence controls;
    geometry_msgs::msg::Pose target;
    double anchor_x{0}, anchor_y{0}, edge{0};
    std::pair<double,double> gates;
    std::string frame;
    int64_t stamp_ns{0}, last_stamp_ns{0};
    uint64_t epoch{0};
    size_t shifts{0};
    std::shared_ptr<const nav_msgs::msg::Path> canonical_plan;
    uint64_t plan_generation{0}, segment_generation{0};
    size_t source_id{0};
    geometry_msgs::msg::Pose canonical_anchor;

  };
  void clearLongEntryCertificate() {
    long_entry_.reset();long_entry_required_=false;long_entry_seed_requested_=false;
    ++long_entry_epoch_;
  }
  bool initializeLongEntryCertificate();
  bool validateLongEntrySequence(models::ControlSequence & sequence,
    size_t & first_hit, bool select_target = false);
  void enforceLongEntryCertificate(bool allow_optimized);
  bool longEntryStateValid() const;
  bool refreshLongEntryBinding();
  bool longEntryIdentityValid() const;
  bool pathBindingValid() const;
  bool longEntryMeasuredAligned() const;
  std::optional<PathBinding> path_binding_;
  bool path_binding_required_{false};
  std::optional<LongEntryCertificate> long_entry_;
  bool long_entry_required_{false};
  bool long_entry_seed_requested_{false};
  uint64_t long_entry_epoch_{0};
  bool tryExtendedShortCuspContinuation(bool initialize);
  bool extend_short_cusp_completion_{false};
  bool cusp_completion_attempted_{false};
  std::optional<models::ControlSequence> cusp_completion_;
  size_t cusp_completion_shifts_{0};
  void preserveFilteredCollisionFeasibility(const models::ControlSequence * before_filter);

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  std::string name_;

  std::shared_ptr<MotionModel> motion_model_;

  ParametersHandler * parameters_handler_;
  CriticManager critic_manager_;
  NoiseGenerator noise_generator_;
  int diag_tick_{0};

  models::OptimizerSettings settings_;

  models::State state_;
  models::ControlSequence control_sequence_;
  std::array<mppi::models::Control, 4> control_history_;
  models::Trajectories generated_trajectories_;
  models::Path path_;
  geometry_msgs::msg::Pose goal_;
  xt::xtensor<float, 1> costs_;

  CriticData critics_data_ = {
    state_, generated_trajectories_, path_, goal_,
    costs_, settings_.model_dt, false, nullptr, nullptr,
    std::nullopt, std::nullopt};  /// Caution, keep references

  rclcpp::Logger logger_{rclcpp::get_logger("MPPIController")};
  bool smooth_control_sequence_{true};
  bool condition_control_on_path_direction_{false};
  bool allow_path_direction_braking_{false};
  int active_cusp_direction_{0};
  int path_direction_{0};
  bool condition_terminal_path_direction_{false};
  bool seed_short_cusp_on_handoff_{false};
  bool short_cusp_seed_requested_{false};
  bool preserve_short_cusp_feasibility_{false};
  bool initialize_long_cusp_entry_{false};
  bool repair_long_cusp_warm_start_{false};
  bool preserve_filtered_collision_feasibility_{false};
  bool retain_collision_free_long_path_sample_{false};
  float motor_deadband_floor_{0.0f};
  float motor_deadband_approach_radius_{0.15f};
  float motor_deadband_terminal_radius_{0.04f};
  float motor_deadband_max_heading_error_{0.5236f};
};

}  // namespace mppi

#endif  // NAV2_MPPI_CONTROLLER__OPTIMIZER_HPP_
