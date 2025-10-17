#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = "hik_camera"
    config_dir = get_package_share_directory(package_name)

    default_config_path = Path(config_dir) / "config" / "hik_camera_params.yaml"

    # 使用预配置的RViz文件
    default_rviz_config = Path(config_dir) / "config" / "hik_camera.rviz"

    # 相机参数声明
    exposure_arg = DeclareLaunchArgument(
        name="exposure",
        default_value="4000.0",
        description="Exposure time in microseconds",
    )

    gain_arg = DeclareLaunchArgument(
        name="image_gain", default_value="16.9807", description="Image gain value"
    )

    fps_arg = DeclareLaunchArgument(
        name="fps", default_value="165.0", description="Frame rate setting"
    )

    trigger_arg = DeclareLaunchArgument(
        name="use_trigger", default_value="false", description="Enable trigger mode"
    )

    rviz_flag = DeclareLaunchArgument(
        name="enable_rviz",
        default_value="true",
        description="Enable RViz visualization",
    )

    rviz_config_argument = DeclareLaunchArgument(
        name="rviz_config_file",
        default_value="",
        description="Custom RViz configuration file",
    )

    camera_driver = Node(
        package=package_name,
        executable="hik_camera_node",
        name="camera_driver",
        output="screen",
        parameters=[
            {
                "exposure": LaunchConfiguration("exposure"),
                "image_gain": LaunchConfiguration("image_gain"),
                "fps": LaunchConfiguration("fps"),
                "use_trigger": LaunchConfiguration("use_trigger"),
                "pixel_format_code": 17301513,
                "camera_frame": "camera_optical_frame",
                "serial_number": "",
            }
        ],
    )

    rviz_visualizer = Node(
        package="rviz2",
        executable="rviz2",
        name="visualization_tool",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config_file")],
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
    )

    return LaunchDescription(
        [
            exposure_arg,
            gain_arg,
            fps_arg,
            trigger_arg,
            rviz_flag,
            rviz_config_argument,
            camera_driver,
            rviz_visualizer,
        ]
    )
