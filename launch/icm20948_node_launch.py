from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription


def generate_launch_description():

    pub_rate = LaunchConfiguration('pub_rate', default='200')
    
    return LaunchDescription(
        [
            DeclareLaunchArgument('pub_rate', default_value='200', description='Publishing rate in Hz'),

            Node(
                package="ros2_icm20948",
                executable="icm20948_node",
                name="icm20948_node",
                parameters=[
                    {"i2c_address": 0x68},
                    {"frame_id": "imu_icm20948"},
                    {"pub_rate": pub_rate},
                ],
            )
        ]
    )
