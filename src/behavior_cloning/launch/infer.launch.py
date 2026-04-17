from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='behavior_cloning',
            executable='moveit_action_server',
            name='moveit_action_server',
            output='screen',
        ),
        Node(
            package='behavior_cloning',
            executable='infer_node.py',
            name='bc_infer_node',
            output='screen',
        ),
    ])
