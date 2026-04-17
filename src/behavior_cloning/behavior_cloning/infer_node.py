#!/usr/bin/env python3

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient

import torch
import torch.nn as nn
import numpy as np

from behavior_cloning.action import MoveToJoint, ExecutePlan

JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]


class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.head = nn.Linear(256, 6)

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


class BCInferNode(Node):
    def __init__(self):
        super().__init__('bc_infer_node')

        self.get_logger().info("Initializing BCInferNode...")

        # Load model
        self.model = BCModel()
        model_path = "/home/gmr/Downloads/ur_ws/src/behavior_cloning/bc_ur5_policy.pth"
        self.get_logger().info(f"Loading model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.get_logger().info("Model loaded successfully.")

        # Joint state subscriber
        self.sub = self.create_subscription(JointState, "/joint_states", self.cb, 10)

        # Action clients
        self.plan_client    = ActionClient(self, MoveToJoint, 'plan_to_joint')
        self.execute_client = ActionClient(self, ExecutePlan,  'execute_plan')

        self.goal_in_progress = False
        self.last_time = self.get_clock().now()

        self.get_logger().info("Waiting for action servers...")
        self.plan_client.wait_for_server()
        self.execute_client.wait_for_server()
        self.get_logger().info("Action servers connected.")

    # ── Joint state callback ──────────────────────────────────────────────

    def cb(self, msg):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        if dt < 0.5:
            return
        self.last_time = now

        if self.goal_in_progress:
            return

        joint_map = dict(zip(msg.name, msg.position))
        try:
            state = np.array([joint_map[j] for j in JOINT_ORDER], dtype=np.float32)
        except KeyError as e:
            self.get_logger().warn(f"Joint mapping failed: {e}")
            return

        self.get_logger().info(f"Joint order: {JOINT_ORDER}")
        self.get_logger().info(f"Current state: {[f'{v:.4f}' for v in state]}")

        with torch.no_grad():
            action = self.model(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy()

        self.get_logger().info(f"Model action: {[f'{v:.4f}' for v in action]}")

        next_joints = np.clip(state + action, -6.28, 6.28).tolist()

        self.get_logger().info(f"Next joints (clipped): {[f'{v:.4f}' for v in next_joints]}")
        self.goal_in_progress = True
        self._send_plan(next_joints)

    # ── Step 1: plan ──────────────────────────────────────────────────────

    def _send_plan(self, joints):
        goal = MoveToJoint.Goal()
        goal.joint_positions = joints
        self.get_logger().info("Sending plan_to_joint goal...")
        future = self.plan_client.send_goal_async(goal)
        future.add_done_callback(self._plan_goal_response)

    def _plan_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("plan_to_joint: goal rejected")
            self.goal_in_progress = False
            return
        self.get_logger().info("plan_to_joint: goal accepted, waiting for result...")
        goal_handle.get_result_async().add_done_callback(self._plan_result)

    def _plan_result(self, future):
        result = future.result().result
        if not result.success:
            self.get_logger().warn("plan_to_joint: planning failed")
            self.goal_in_progress = False
            return

        self.get_logger().info("plan_to_joint: planning succeeded.")
        # Prompt user in a separate thread so we don't block the spin
        threading.Thread(target=self._prompt_and_execute, daemon=True).start()

    # ── Step 2: ask user, then execute ────────────────────────────────────

    def _prompt_and_execute(self):
        try:
            answer = input("\n[infer_node] Execute plan? [y/N]: ").strip().lower()
        except EOFError:
            answer = ''

        if answer == 'y':
            self.get_logger().info("User confirmed. Sending execute_plan goal...")
            future = self.execute_client.send_goal_async(ExecutePlan.Goal())
            future.add_done_callback(self._exec_goal_response)
        else:
            self.get_logger().info("User skipped execution.")
            self.goal_in_progress = False

    def _exec_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("execute_plan: goal rejected (no plan stored?)")
            self.goal_in_progress = False
            return
        goal_handle.get_result_async().add_done_callback(self._exec_result)

    def _exec_result(self, future):
        result = future.result().result
        self.get_logger().info(f"execute_plan: {'succeeded' if result.success else 'failed'}")
        self.goal_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = BCInferNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
