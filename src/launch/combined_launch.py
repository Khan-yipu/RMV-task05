#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from pathlib import Path
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    package_name = "hik_camera"
    package_dir = get_package_share_directory(package_name)

    # 使用预配置的RViz文件
    default_rviz_config = Path(package_dir) / "config" / "armor_detect.rviz"

    # 相机参数配置
    camera_params = {
        "exposure": 4000.0,
        "image_gain": 16.98,
        "use_trigger": False,
        "fps": 165.0,
        "pixel_format_code": 17301513,  # BayerRG8
        "camera_frame": "camera_optical_frame",
        "serial_number": "",  # 留空使用第一个可用相机
    }

    # 装甲板识别参数配置
    detector_params = {
        "debug_mode": True,  # 开启debug模式，输出PNP坐标信息
        "process_interval": 0.0,
        "image_topic": "/image_raw",
        "result_topic": "/armor_detection_result",
        "enable_statistics": True,
        "statistics_interval": 30,
    }

    # RViz启动参数
    rviz_flag = DeclareLaunchArgument(
        name="enable_rviz",
        default_value="true",
        description="Enable RViz visualization",
    )

    rviz_config_argument = DeclareLaunchArgument(
        name="rviz_config_file",
        default_value=str(default_rviz_config),
        description="Custom RViz configuration file",
    )

    # 海康相机节点
    camera_node = Node(
        package=package_name,
        executable="hik_camera_node",
        name="hik_camera",
        output="screen",
        parameters=[camera_params],
        remappings=[
            ("/image_raw", "/camera/image_raw")  # 可选：重映射话题
        ],
    )

    # 装甲板识别节点
    detector_node = Node(
        package=package_name,
        executable="armor_detector_node",
        name="armor_detector",
        output="screen",
        parameters=[detector_params],
        remappings=[
            ("/image_raw", "/camera/image_raw"),  # 订阅相机图像
            ("/armor_detection_result", "/detection/result"),  # 发布检测结果
        ],
    )

    # RViz可视化节点
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config_file")],
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
    )

    return LaunchDescription(
        [
            rviz_flag,
            rviz_config_argument,
            camera_node,
            detector_node,
            rviz_node,
        ]
    )
