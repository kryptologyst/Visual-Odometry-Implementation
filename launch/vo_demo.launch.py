#!/usr/bin/env python3
"""ROS 2 launch file for Visual Odometry demo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import os


def generate_launch_description():
    """Generate launch description."""
    
    # Declare launch arguments
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='ORB',
        description='Visual Odometry algorithm (ORB, SIFT, OpticalFlow)'
    )
    
    trajectory_arg = DeclareLaunchArgument(
        'trajectory',
        default_value='circular',
        description='Trajectory type (circular, straight, figure_eight)'
    )
    
    num_frames_arg = DeclareLaunchArgument(
        'num_frames',
        default_value='100',
        description='Number of frames to generate'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='',
        description='Path to configuration file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Visual Odometry node
    vo_node = Node(
        package='visual_odometry',
        executable='visual_odometry_node',
        name='visual_odometry',
        parameters=[{
            'algorithm': LaunchConfiguration('algorithm'),
            'trajectory': LaunchConfiguration('trajectory'),
            'num_frames': LaunchConfiguration('num_frames'),
            'config_file': LaunchConfiguration('config_file'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/visual_odometry/pose', '/visual_odometry/pose'),
            ('/visual_odometry/trajectory', '/visual_odometry/trajectory'),
        ]
    )
    
    # Visualization node
    viz_node = Node(
        package='visual_odometry',
        executable='visualization_node',
        name='visualization',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
        condition=IfCondition('true')  # Always launch visualization
    )
    
    # TF2 static transform publisher for camera
    tf_camera_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_link'],
        output='screen'
    )
    
    # Log info
    log_info = LogInfo(
        msg=['Launching Visual Odometry with algorithm: ', LaunchConfiguration('algorithm')]
    )
    
    return LaunchDescription([
        algorithm_arg,
        trajectory_arg,
        num_frames_arg,
        config_file_arg,
        use_sim_time_arg,
        log_info,
        vo_node,
        viz_node,
        tf_camera_node,
    ])
