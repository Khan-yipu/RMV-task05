from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = "hik_camera"
    package_dir = get_package_share_directory(package_name)

    # 装甲板识别参数配置
    detector_params = {
        "debug_mode": True,
        "process_interval": 0.0,  # 0表示处理每一帧
        "image_topic": "/image_raw",
        "result_topic": "/armor_detection_result",
        "enable_statistics": True,
        "statistics_interval": 30,
    }

    return LaunchDescription(
        [
            Node(
                package=package_name,
                executable="armor_detector_node",
                name="armor_detector",
                output="screen",
                parameters=[detector_params],
            )
        ]
    )
