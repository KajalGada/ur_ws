import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import os
from datetime import datetime
import math

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

THRESHOLD_RAD = 2 * math.pi / 180.0  # 2 degrees in radians


class FreeDriveLogger(Node):
    def __init__(self):
        super().__init__('freedrive_logger')

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.initial_state = None
        self.recording = False
        self.states = []
        self._logged_first_msg = False

        self.get_logger().info(
            "Freedrive logger started. Waiting for 2-degree movement to begin recording..."
        )

    def joint_callback(self, msg):
        if not self._logged_first_msg:
            self.get_logger().info(f"Received joint_states msg, joints: {list(msg.name)}")
            self._logged_first_msg = True
        name_to_pos = dict(zip(msg.name, msg.position))

        # Only proceed when all 6 joints are present
        try:
            state = np.array([name_to_pos[j] for j in JOINT_NAMES])
        except KeyError as e:
            self.get_logger().warn(f"Missing joint: {e}")
            return

        if not self.recording:
            # Capture initial position on first valid message
            if self.initial_state is None:
                self.initial_state = state.copy()
                self.get_logger().info("Initial joint state captured. Move the arm to start recording.")
                return

            # Wait until any joint moves more than 2 degrees from the initial position
            diff = np.abs(state - self.initial_state)
            if np.any(diff >= THRESHOLD_RAD):
                self.recording = True
                self.get_logger().info("Motion detected! Recording started.")

        if self.recording:
            self.states.append(state.copy())

    def save(self, base_dir="dataset"):
        if not self.states:
            self.get_logger().warn("No data recorded. Nothing to save.")
            return

        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(base_dir, f"demo_{timestamp}.npz")

        states = np.array(self.states)
        actions = np.diff(states, axis=0)   # actions[i] = states[i+1] - states[i]
        states = states[:-1]                # align: drop last state (no action for it)

        np.savez(
            filename,
            joint_names=np.array(JOINT_NAMES),
            states=states,
            actions=actions,
        )

        self.get_logger().info(f"Saved {len(states)} (state, action) pairs to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = FreeDriveLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
