from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # 声明启动参数
    video_file_arg = DeclareLaunchArgument(
        'video_file',
        default_value='/home/khanif/yipu/robomaster/wuhu5/armor_detect/blue.mp4',
        description='Path to the video file for armor detection'
    )

    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug mode'
    )

    result_topic_arg = DeclareLaunchArgument(
        'result_topic',
        default_value='/video_armor_detection_result',
        description='Topic to publish detection results'
    )

    enable_statistics_arg = DeclareLaunchArgument(
        'enable_statistics',
        default_value='true',
        description='Enable processing statistics'
    )

    statistics_interval_arg = DeclareLaunchArgument(
        'statistics_interval',
        default_value='30',
        description='Interval for printing statistics (in frames)'
    )

    loop_delay_arg = DeclareLaunchArgument(
        'loop_delay',
        default_value='30.0',
        description='Delay between frame processing in milliseconds'
    )

    # 获取参数值
    video_file = LaunchConfiguration('video_file')
    debug_mode = LaunchConfiguration('debug_mode')
    result_topic = LaunchConfiguration('result_topic')
    enable_statistics = LaunchConfiguration('enable_statistics')
    statistics_interval = LaunchConfiguration('statistics_interval')
    loop_delay = LaunchConfiguration('loop_delay')

    # 视频装甲板识别参数配置
    detector_params = {
        "debug_mode": debug_mode,
        "video_file": video_file,
        "result_topic": result_topic,
        "enable_statistics": enable_statistics,
        "statistics_interval": statistics_interval,
        "loop_delay": loop_delay,
    }

    return LaunchDescription(
        [
            video_file_arg,
            debug_mode_arg,
            result_topic_arg,
            enable_statistics_arg,
            statistics_interval_arg,
            loop_delay_arg,
            Node(
                package="hik_camera",
                executable="video_armor_detect",
                name="video_armor_detector",
                output="screen",
                parameters=[detector_params],
            )
        ]
    )