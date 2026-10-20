// Copyright (c) 2026 Kaido Kert
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

// Regression reductions for the small-gym dynamic-tolerance claim.
#include <memory>
#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_controller/plugins/stopped_goal_checker.hpp"
#include "nav2_util/geometry_utils.hpp"

class DynamicStoppedGoalChecker : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("stopped_checker_reduction");
    checker.initialize(node, "checker", nullptr);
  }
  void set(const std::vector<rclcpp::Parameter> & parameters)
  {
    ASSERT_TRUE(node->set_parameters_atomically(parameters).successful);
  }
  bool reached(double x, double yaw = 0.0, double vx = 0.0, double wz = 0.0)
  {
    geometry_msgs::msg::Pose robot, goal;
    robot.position.x = x;
    robot.orientation = nav2_util::geometry_utils::orientationAroundZAxis(yaw);
    goal.orientation.w = 1.0;
    geometry_msgs::msg::Twist velocity;
    velocity.linear.x = vx; velocity.angular.z = wz;
    return checker.isGoalReached(robot, goal, velocity);
  }
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  nav2_controller::StoppedGoalChecker checker;
};

TEST_F(DynamicStoppedGoalChecker, TightenAndWidenXYUpdatesDecisionAndReportedTolerance)
{
  EXPECT_TRUE(reached(0.20));
  set({rclcpp::Parameter("checker.xy_goal_tolerance", 0.08),
      rclcpp::Parameter("checker.stateful", false)});
  checker.reset();
  EXPECT_FALSE(reached(0.20));
  EXPECT_TRUE(reached(0.06));
  geometry_msgs::msg::Pose pose; geometry_msgs::msg::Twist velocity;
  ASSERT_TRUE(checker.getTolerances(pose, velocity));
  EXPECT_DOUBLE_EQ(pose.position.x, 0.08);
  set({rclcpp::Parameter("checker.xy_goal_tolerance", 0.30)});
  EXPECT_TRUE(reached(0.20));
  EXPECT_FALSE(reached(0.31));
}

TEST_F(DynamicStoppedGoalChecker, TightenYawUpdatesDecision)
{
  EXPECT_TRUE(reached(0.0, 0.20));
  set({rclcpp::Parameter("checker.yaw_goal_tolerance", 0.10)});
  checker.reset();
  EXPECT_FALSE(reached(0.0, 0.20));
  EXPECT_TRUE(reached(0.0, 0.05));
}

TEST_F(DynamicStoppedGoalChecker, StatefulToggleAndResetAreEffective)
{
  EXPECT_FALSE(reached(0.05, 0.50));  // Position latches, heading has not arrived.
  EXPECT_TRUE(reached(0.30));
  checker.reset();
  EXPECT_FALSE(reached(0.30));
  set({rclcpp::Parameter("checker.stateful", false)});
  EXPECT_FALSE(reached(0.05, 0.50));
  EXPECT_FALSE(reached(0.30));
  EXPECT_TRUE(reached(0.05));
}

TEST_F(DynamicStoppedGoalChecker, CombinedPoseAndStoppedVelocityUpdatePreservesArrest)
{
  set({rclcpp::Parameter("checker.xy_goal_tolerance", 0.08),
      rclcpp::Parameter("checker.yaw_goal_tolerance", 0.10),
      rclcpp::Parameter("checker.stateful", false),
      rclcpp::Parameter("checker.trans_stopped_velocity", 0.02),
      rclcpp::Parameter("checker.rot_stopped_velocity", 0.02)});
  EXPECT_TRUE(reached(0.06, 0.05, 0.01, 0.01));
  EXPECT_FALSE(reached(0.06, 0.05, 0.03, 0.01));
  EXPECT_FALSE(reached(0.06, 0.05, 0.01, 0.03));
  EXPECT_FALSE(reached(0.09, 0.05));
  EXPECT_FALSE(reached(0.06, 0.11));
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
