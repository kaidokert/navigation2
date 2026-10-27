#!/usr/bin/env python3
"""Synthetic stationary FollowPath smoke test. Run ONLY in the isolated test container.

Container must have --network none, no devices/host IPC and ROS_DOMAIN_ID=176.
The robot's real stack is neither addressed nor modified. Not a driving benchmark.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from nav2_msgs.action import FollowPath
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


def main():
    if os.environ.get('ROS_DOMAIN_ID') != '176' or not Path('/.dockerenv').exists():
        raise SystemExit('Requires isolated Docker test container and ROS_DOMAIN_ID=176')
    # The launch wrapper supplies ROS parameters and launches controller_server.
    rclpy.init()
    node = rclpy.create_node('timing_stationary_fixture')
    tf = TransformBroadcaster(node)
    odom_pub = node.create_publisher(Odometry, '/odom', 10)
    scan_pub = node.create_publisher(LaserScan, '/scan', 10)
    map_pub = node.create_publisher(OccupancyGrid, '/map', QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    timing, commands = [], []
    node.create_subscription(String, '/controller_server/control_timing',
                             lambda msg: timing.append(json.loads(msg.data)), 10)
    node.create_subscription(TwistStamped, '/dry_run/cmd_vel',
                             lambda msg: commands.append([msg.twist.linear.x, msg.twist.angular.z]), 10)

    def inputs():
        stamp = node.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.rotation.w = 1.0
        tf.sendTransform(transform)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.orientation.w = 1.0
        odom_pub.publish(odom)
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = 'base_link'
        scan.angle_min = -3.14159
        scan.angle_max = 3.14159
        scan.angle_increment = 6.28318 / 359
        scan.range_min = .05
        scan.range_max = 8.0
        scan.ranges = [8.0] * 360
        scan_pub.publish(scan)

    node.create_timer(.02, inputs)
    grid = OccupancyGrid()
    grid.header.frame_id = 'map'
    grid.info.resolution = .05
    grid.info.width = grid.info.height = 200
    grid.info.origin.position.x = grid.info.origin.position.y = -5.0
    grid.info.origin.orientation.w = 1.0
    grid.data = [0] * 40000
    map_pub.publish(grid)

    def wait(future, timeout=15):
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.02)
        if not future.done():
            raise RuntimeError('ROS operation timed out')
        return future.result()

    lifecycle = node.create_client(ChangeState, '/controller_server/change_state')
    if not lifecycle.wait_for_service(timeout_sec=20):
        raise RuntimeError('Controller lifecycle service missing')
    for transition in (Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE):
        request = ChangeState.Request()
        request.transition.id = transition
        if not wait(lifecycle.call_async(request)).success:
            raise RuntimeError(f'Lifecycle transition failed: {transition}')

    client = ActionClient(node, FollowPath, '/follow_path')
    if not client.wait_for_server(timeout_sec=10):
        raise RuntimeError('FollowPath missing')
    goal = FollowPath.Goal()
    goal.controller_id = 'FollowPath'
    goal.goal_checker_id = 'general_goal_checker'
    goal.path = NavPath()
    goal.path.header.frame_id = 'map'
    goal.path.header.stamp = node.get_clock().now().to_msg()
    for i in range(21):
        pose = PoseStamped()
        pose.header = goal.path.header
        pose.pose.position.x = i * .05
        pose.pose.orientation.w = 1.0
        goal.path.poses.append(pose)
    handle = wait(client.send_goal_async(goal))
    if not handle.accepted:
        raise RuntimeError('Synthetic goal rejected')
    end = time.monotonic() + 6
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=.02)
    wait(handle.cancel_goal_async())
    result = wait(handle.get_result_async())
    end = time.monotonic() + 2
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=.02)
    output = {'fixture': 'synthetic stationary, no hardware', 'action_status': result.status,
              'command_count': len(commands), 'windows': timing}
    Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + '\n')
    assert len(commands) > 20, 'No sustained controller output'
    assert any(w['completed'] > 5 for w in timing), 'No completed cycle telemetry'
    for field in ('optimizer', 'path_tf', 'costmap_lock_wait', 'thread_cpu'):
        assert any(w['metrics_seconds'][field]['samples'] > 0 for w in timing), field
    assert any(w['samples'] == 0 for w in timing), 'Idle window missing'
    request = ChangeState.Request()
    request.transition.id = Transition.TRANSITION_DEACTIVATE
    wait(lifecycle.call_async(request))
    print(json.dumps({'result': 'PASS', 'windows': len(timing), 'commands': len(commands)}))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
