from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription

#
# See https://github.com/slgrobotics/robots_bringup/blob/main/Docs/Sensors/ICM20948%20IMU.md
#

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
                    # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py
                    # SparkFun address is likely 0x69, generic GY-ICM20948 - 0x68
                    # Use "i2cdetect -y 1"
                    {"i2c_address": 0x68},
                    {"frame_id": "imu_link"},
                    {"pub_rate": pub_rate},
                    {"madgwick_beta": 0.08},
                    {"madgwick_use_mag": True},
                    {"gyro_calib_seconds": 3.0},
                    {"gyro_calib_max_std_dps": 1.0} # warning threshold - if std dev is high during calibration
                ],
            )
        ]
    )
