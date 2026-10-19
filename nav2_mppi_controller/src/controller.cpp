#include "nav2_mppi_controller/tools/decision_recorder.hpp"
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

#include <stdint.h>
#include <chrono>
#include "nav2_mppi_controller/controller.hpp"
#include "nav2_mppi_controller/tools/utils.hpp"

// #define BENCHMARK_TESTING

namespace nav2_mppi_controller
{

void MPPIController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  tf_buffer_ = tf;
  name_ = name;
  parameters_handler_ = std::make_unique<ParametersHandler>(parent);

  auto node = parent_.lock();
  // Get high-level controller parameters
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(visualize_, "visualize", false);
  getParam(reset_on_path_handoff_, "reset_on_path_handoff", false);
  getParam(enable_presteer_on_handoff_, "enable_presteer_on_handoff", false);
  getParam(presteer_creep_speed_, "presteer_creep_speed", 0.038);
  getParam(presteer_timeout_, "presteer_timeout", 1.0);
  getParam(presteer_angle_tolerance_, "presteer_angle_tolerance", 0.05);
  getParam(presteer_wheelbase_, "AckermannConstraints.steering_wheelbase", 0.213);
  getParam(presteer_max_angle_, "AckermannConstraints.steering_max_angle", 0.576);
  if (presteer_wheelbase_ <= 0.0) {presteer_wheelbase_ = 0.213;}
  if (presteer_max_angle_ <= 0.0) {presteer_max_angle_ = 0.576;}

  // Configure composed objects
  optimizer_.initialize(parent_, name_, costmap_ros_, parameters_handler_.get());
  if (optimizer_.requiresSteeringFeedback() || enable_presteer_on_handoff_) {
    std::string topic;
    getParam(topic, "AckermannConstraints.steering_feedback_topic",
      std::string("/robot_joint_states"), ParameterType::Static);
    getParam(steering_feedback_timeout_, "AckermannConstraints.steering_feedback_timeout",
      0.3, ParameterType::Static);
    uint64_t timeout_bits;
    std::memcpy(&timeout_bits, &steering_feedback_timeout_, sizeof(timeout_bits));
    if ((timeout_bits & 0x7ff0000000000000ull) == 0x7ff0000000000000ull ||
      steering_feedback_timeout_ <= 0.0 || steering_feedback_timeout_ > 1.0 || topic.empty())
    {
      throw std::invalid_argument("Invalid steering feedback topic/timeout");
    }
    steering_subscription_ = node->create_subscription<sensor_msgs::msg::JointState>(
      topic, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {
        receiveSteeringFeedback(message);
      });
  }
  if (optimizer_.requiresConditionedSteering()) {
    std::string topic;
    getParam(topic, "AckermannConstraints.conditioned_steering_topic",
      std::string(""), ParameterType::Static);
    if (topic.empty()) {
      throw std::invalid_argument("Actuator prediction needs a conditioned steering topic");
    }
    conditioned_steering_subscription_ = node->create_subscription<std_msgs::msg::Float64>(
      topic, rclcpp::SensorDataQoS(),
      [this](std_msgs::msg::Float64::ConstSharedPtr message) {
        receiveConditionedSteering(message);
      });
  }
  path_handler_.initialize(parent_, name_, costmap_ros_, tf_buffer_, parameters_handler_.get());
  trajectory_visualizer_.on_configure(
    parent_, name_,
    costmap_ros_->getGlobalFrameID(), parameters_handler_.get());

  RCLCPP_INFO(logger_, "Configured MPPI Controller: %s", name_.c_str());
}

void MPPIController::cleanup()
{
  presteering_ = false;
  steering_subscription_.reset();
  conditioned_steering_subscription_.reset();
  {
    std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
    steering_feedback_.reset();
    conditioned_steering_.reset();
    conditioned_steering_receipt_.reset();
  }
  optimizer_.shutdown();
  trajectory_visualizer_.on_cleanup();
  parameters_handler_.reset();
  RCLCPP_INFO(logger_, "Cleaned up MPPI Controller: %s", name_.c_str());
}

void MPPIController::activate()
{
  trajectory_visualizer_.on_activate();
  optimizer_.on_activate();
  parameters_handler_->start();
  RCLCPP_INFO(logger_, "Activated MPPI Controller: %s", name_.c_str());
}

void MPPIController::deactivate()
{
  presteering_ = false;
  {
    std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
    steering_feedback_.reset();
    conditioned_steering_.reset();
    conditioned_steering_receipt_.reset();
  }
  optimizer_.setInitialSteering(std::nullopt);
  optimizer_.setInitialConditionedSteering(std::nullopt);
  trajectory_visualizer_.on_deactivate();
  optimizer_.on_deactivate();
  RCLCPP_INFO(logger_, "Deactivated MPPI Controller: %s", name_.c_str());
}

void MPPIController::reset()
{
  presteering_ = false;
  optimizer_.setInitialSteering(std::nullopt);
  optimizer_.setInitialConditionedSteering(std::nullopt);
  optimizer_.reset(false /*Don't reset zone-based speed limits between requests*/);
}

void MPPIController::receiveSteeringFeedback(
  sensor_msgs::msg::JointState::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
  steering_feedback_ = std::move(message);
}

void MPPIController::updateSteeringFeedback()
{
  if (!optimizer_.requiresSteeringFeedback() && !enable_presteer_on_handoff_) {return;}
  optimizer_.setInitialSteering(std::nullopt);
  optimizer_.setInitialConditionedSteering(std::nullopt);
  sensor_msgs::msg::JointState::ConstSharedPtr message;
  std::optional<float> conditioned;
  std::optional<rclcpp::Time> conditioned_receipt;
  {
    std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
    message = steering_feedback_;
    conditioned = conditioned_steering_;
    conditioned_receipt = conditioned_steering_receipt_;
  }
  if (!message) {throw nav2_core::NoValidControl("Missing measured steering feedback");}
  if (message->header.stamp.sec < 0 || message->header.stamp.nanosec >= 1000000000u) {
    throw nav2_core::NoValidControl("Malformed measured steering timestamp");
  }
  auto node = parent_.lock();
  const rclcpp::Time stamp(message->header.stamp, node->get_clock()->get_clock_type());
  const double age = (node->now() - stamp).seconds();
  if (stamp.nanoseconds() <= 0 || age < 0.0 || age > steering_feedback_timeout_) {
    throw nav2_core::NoValidControl("Stale or future measured steering feedback");
  }
  const auto & names = message->name;
  const auto found = std::find(names.begin(), names.end(), "front_steering_joint");
  if (found == names.end() || std::count(names.begin(), names.end(), "front_steering_joint") != 1 ||
    static_cast<size_t>(found - names.begin()) >= message->position.size())
  {
    throw nav2_core::NoValidControl("Missing or ambiguous front_steering_joint position");
  }
  optimizer_.setInitialSteering(static_cast<float>(message->position[found - names.begin()]));
  if (optimizer_.requiresConditionedSteering()) {
    if (!conditioned || !conditioned_receipt) {
      throw nav2_core::NoValidControl("Missing conditioned steering feedback");
    }
    const double target_age = (node->now() - *conditioned_receipt).seconds();
    if (target_age < 0 || target_age > steering_feedback_timeout_) {
      throw nav2_core::NoValidControl("Stale or future conditioned steering receipt");
    }
    optimizer_.setInitialConditionedSteering(conditioned);
  }
}

void MPPIController::receiveConditionedSteering(std_msgs::msg::Float64::ConstSharedPtr message)
{
  auto node = parent_.lock();
  std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
  conditioned_steering_ = static_cast<float>(message->data);
  conditioned_steering_receipt_ = node->now();
}

geometry_msgs::msg::TwistStamped MPPIController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  nav2_core::GoalChecker * goal_checker)
{
#ifdef BENCHMARK_TESTING
  auto start = std::chrono::system_clock::now();
#endif

  mppi::recorder::Invocation recording(rclcpp::Time(robot_pose.header.stamp).nanoseconds());
  std::lock_guard<std::mutex> param_lock(*parameters_handler_->getLock());
  // Validate before path pruning can consume a cusp or advance segment state.
  updateSteeringFeedback();
  if (goal_checker) {
    geometry_msgs::msg::Pose pose_tol;
    geometry_msgs::msg::Twist vel_tol;
    if (goal_checker->getTolerances(pose_tol, vel_tol)) {
      path_handler_.setGoalTolerance(pose_tol.position.x);
    }
  }
  geometry_msgs::msg::Pose goal = path_handler_.getTransformedGoal(robot_pose.header.stamp).pose;

  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(*(costmap->getMutex()));
  nav_msgs::msg::Path transformed_plan = path_handler_.transformPath(robot_pose,
    [&](const auto & pose, const auto & outgoing, double xy, double yaw, bool terminal_endpoint) {
      return optimizer_.longCuspDepartureAvailable(
        pose, robot_speed, outgoing, xy, yaw, terminal_endpoint);
    });

  if (reset_on_path_handoff_ && path_handler_.didHandoff()) {
    // A warm start and smoothing history from the old gear bias sampling
    // against the new segment. Clear only optimizer memory; evalControl still
    // starts its rollout from the measured physical velocity below. Retain
    // zone speed limits, odometry and the hardware steering filter.
    reset();
    updateSteeringFeedback();
    RCLCPP_INFO(logger_, "[MPPI] Handoff cleared previous gear control history");
  }

  auto node = parent_.lock();
  if (enable_presteer_on_handoff_ && node) {
    if (path_handler_.didHandoff()) {
      presteer_gear_ = path_handler_.getPathDirection();
      if (presteer_gear_ != 0 && transformed_plan.poses.size() >= 2) {
        const double rx = robot_pose.pose.position.x;
        const double ry = robot_pose.pose.position.y;
        const double ryaw = tf2::getYaw(robot_pose.pose.orientation);
        const double cos_yaw = std::cos(ryaw);
        const double sin_yaw = std::sin(ryaw);

        size_t lookahead_idx = 1;
        for (size_t i = 1; i < transformed_plan.poses.size(); ++i) {
          const double dx = transformed_plan.poses[i].pose.position.x - rx;
          const double dy = transformed_plan.poses[i].pose.position.y - ry;
          if (std::hypot(dx, dy) >= 0.20 || i == transformed_plan.poses.size() - 1) {
            lookahead_idx = i;
            break;
          }
        }

        const double dx = transformed_plan.poses[lookahead_idx].pose.position.x - rx;
        const double dy = transformed_plan.poses[lookahead_idx].pose.position.y - ry;
        const double local_x = dx * cos_yaw + dy * sin_yaw;
        const double local_y = -dx * sin_yaw + dy * cos_yaw;
        const double dist_sq = local_x * local_x + local_y * local_y;

        if (dist_sq > 1e-4) {
          const double kappa = 2.0 * local_y / dist_sq;
          const double target_steer = std::clamp(
            std::atan(presteer_wheelbase_ * kappa),
            -presteer_max_angle_, presteer_max_angle_);

          float current_steer = 0.0f;
          {
            std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
            if (steering_feedback_) {
              const auto & names = steering_feedback_->name;
              const auto found = std::find(names.begin(), names.end(), "front_steering_joint");
              if (found != names.end() && static_cast<size_t>(found - names.begin()) < steering_feedback_->position.size()) {
                current_steer = static_cast<float>(steering_feedback_->position[found - names.begin()]);
              }
            }
          }
          if (std::abs(current_steer - target_steer) > presteer_angle_tolerance_) {
            presteering_ = true;
            presteer_start_time_ = node->now();
            presteer_target_angle_ = static_cast<float>(target_steer);
            RCLCPP_INFO(logger_,
              "[PRESTEER] Initiated: current=%.2f deg, target=%.2f deg, gear=%d, lookahead=%.3fm",
              current_steer * 180.0f / M_PI, target_steer * 180.0f / M_PI, presteer_gear_,
              std::sqrt(dist_sq));
          }
        }
      }
    }

    if (presteering_) {
      float current_steer = 0.0f;
      {
        std::lock_guard<std::mutex> lock(steering_feedback_mutex_);
        if (steering_feedback_) {
          const auto & names = steering_feedback_->name;
          const auto found = std::find(names.begin(), names.end(), "front_steering_joint");
          if (found != names.end() && static_cast<size_t>(found - names.begin()) < steering_feedback_->position.size()) {
            current_steer = static_cast<float>(steering_feedback_->position[found - names.begin()]);
          }
        }
      }
      double elapsed = (node->now() - presteer_start_time_).seconds();
      if (elapsed < 0.0 ||
          std::abs(current_steer - presteer_target_angle_) <= presteer_angle_tolerance_ ||
          elapsed >= presteer_timeout_)
      {
        RCLCPP_INFO(logger_,
          "[PRESTEER] Finished in %.3fs: current=%.2f deg, target=%.2f deg (timeout=%.1fs)",
          elapsed, current_steer * 180.0f / M_PI, presteer_target_angle_ * 180.0f / M_PI,
          presteer_timeout_);
        presteering_ = false;
      } else {
        const double vx = presteer_gear_ * presteer_creep_speed_;
        const double wz = vx * std::tan(presteer_target_angle_) / presteer_wheelbase_;
        geometry_msgs::msg::TwistStamped cmd = utils::toTwistStamped(
          static_cast<float>(vx), static_cast<float>(wz),
          robot_pose.header.stamp, costmap_ros_->getBaseFrameID());
        // Custom handoff creep bypasses evalControl, but must obey the same
        // current-body and full-horizon collision contract as optimized output.
        if (!optimizer_.isCommandCollisionFree(robot_pose, robot_speed, cmd.twist)) {
          throw nav2_core::NoValidControl("Presteer command forecast is not collision-free");
        }
        if (visualize_) {
          visualize(std::move(transformed_plan), cmd.header.stamp);
        }
        mppi::recorder::observe("controller_return", [&] {return mppi::recorder::Json{
          {"mode","presteer"},{"vx",cmd.twist.linear.x},{"vy",cmd.twist.linear.y},{"wz",cmd.twist.angular.z}};});
        return cmd;
      }
    }
  }

  optimizer_.setPathBinding(path_handler_.getPathBinding());
  optimizer_.setRealPathPointCount(path_handler_.getRealPathPointCount());
  optimizer_.setActiveCuspDirection(path_handler_.getActiveCuspDirection());
  optimizer_.setCuspTolerances(path_handler_.getCuspTolerances());
  optimizer_.setPathDirection(path_handler_.getPathDirection());
  if (path_handler_.didHandoff()) {optimizer_.requestShortCuspSeed();}
  geometry_msgs::msg::TwistStamped cmd =
    optimizer_.evalControl(robot_pose, robot_speed, transformed_plan, goal, goal_checker);
  if (const auto approach = path_handler_.getRetainedDepartureApproach()) {
    // The departure decision retained the incoming gear. Check the actual
    // post-filter command, not merely a carried mean that MPPI can replace.
    if (!optimizer_.retainedDepartureCommandSafe(
        robot_pose, robot_speed, cmd.twist, approach->cusp, approach->stop_margin))
    {
      throw nav2_core::FailedToMakeProgress(
              "Rejected departure approach command exceeds bounded incoming envelope");
    }
  }

#ifdef BENCHMARK_TESTING
  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  RCLCPP_INFO(logger_, "Control loop execution time: %ld [ms]", duration);
#endif

  if (visualize_) {
    visualize(std::move(transformed_plan), cmd.header.stamp);
  }

  mppi::recorder::observe("controller_return", [&] {return mppi::recorder::Json{
    {"mode","optimizer"},{"vx",cmd.twist.linear.x},{"vy",cmd.twist.linear.y},{"wz",cmd.twist.angular.z}};});
  return cmd;
}

void MPPIController::visualize(
  nav_msgs::msg::Path transformed_plan,
  const builtin_interfaces::msg::Time & cmd_stamp)
{
  trajectory_visualizer_.add(optimizer_.getGeneratedTrajectories(), "Candidate Trajectories");
  trajectory_visualizer_.add(optimizer_.getOptimizedTrajectory(), "Optimal Trajectory", cmd_stamp);
  trajectory_visualizer_.visualize(std::move(transformed_plan));
}

void MPPIController::setPlan(const nav_msgs::msg::Path & path)
{
  presteering_ = false;
  path_handler_.setPath(path);
  optimizer_.requestPlanEntrySeed();
}

void MPPIController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  optimizer_.setSpeedLimit(speed_limit, percentage);
}

}  // namespace nav2_mppi_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_mppi_controller::MPPIController, nav2_core::Controller)
