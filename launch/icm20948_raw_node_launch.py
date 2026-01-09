from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription

#
# See https://github.com/slgrobotics/robots_bringup/blob/main/Docs/Sensors/ICM20948%20IMU.md
#
# Testing: ros2 launch ros2_icm20948 icm20948_raw_node_launch.py
#

def generate_launch_description():

    pub_rate_hz = LaunchConfiguration('pub_rate_hz', default='200')
    
    return LaunchDescription(
        [
            DeclareLaunchArgument('pub_rate_hz', default_value='200', description='Publishing rate in Hz'),

            Node(
                package="ros2_icm20948",
                executable="icm20948_raw_node",
                name="icm20948_raw_node",
                parameters=[{
                    # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py
                    # SparkFun address is likely 0x69, generic GY-ICM20948 - 0x68
                    # Use "i2cdetect -y 1"
                    "i2c_address": 0x68,
                    "frame_id": "imu_link",
                    "pub_rate_hz": pub_rate_hz,  # integer, default 50 in code, 200 here
                    "temp_pub_rate_hz": 1.0,     # float, default 1.0
                    "startup_calib_seconds": 3.0,
                    "gyro_calib_max_std_dps": 2.0,    # warning threshold - if std dev is too high during calibration; default 1.0
                    "accel_calib_max_std_mps2": 0.35  # same for accel; default 0.35
                }]
            ),

            # https://github.com/CCNYRoboticsLab/imu_tools/tree/jazzy
            # sudo apt install ros-${ROS_DISTRO}-imu-tools
            Node(
                package='imu_filter_madgwick',
                executable='imu_filter_madgwick_node',
                name='imu_filter',
                output='screen',
                parameters=[{
                    "stateless": False,
                    "use_mag": True,
                    "publish_tf": True,
                    "reverse_tf": False,
                    "fixed_frame": "odom",
                    "constant_dt": 0.0,
                    "publish_debug_topics": False,
                    "world_frame": "enu",
                    "gain": 0.1,
                    "zeta": 0.0,
                    "mag_bias_x": 0.0,
                    "mag_bias_y": 0.0,
                    "mag_bias_z": 0.0,
                    "orientation_stddev": 0.0
                }],
                remappings=[("imu/mag", "imu/mag_raw")]
            ),

            # for experiments: RViz starts with "map" as Global Fixed Frame, provide a TF to see axes etc.
            Node(
                package = "tf2_ros", 
                executable = "static_transform_publisher",
                arguments=[
                    '--x', '0.0',     # X translation in meters
                    '--y', '0.0',     # Y translation in meters
                    '--z', '0.1',     # Z translation in meters
                    '--roll', '0.0',  # Roll in radians
                    '--pitch', '0.0', # Pitch in radians
                    '--yaw', '0.0',   # Yaw in radians (e.g., 90 degrees)
                    '--frame-id', 'map', # Parent frame ID
                    '--child-frame-id', 'odom' # Child frame ID
                ]
            )
        ]
    )
