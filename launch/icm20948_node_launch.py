from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription

#
# See https://github.com/slgrobotics/robots_bringup/blob/main/Docs/Sensors/ICM20948%20IMU.md
#
# Testing: ros2 launch ros2_icm20948 icm20948_node_launch.py
#

def generate_launch_description():

    pub_rate_hz = LaunchConfiguration('pub_rate_hz', default='200')
    
    return LaunchDescription([

        DeclareLaunchArgument('pub_rate_hz', default_value='200', description='Publishing rate in Hz'),

        Node(
            package="ros2_icm20948",
            executable="icm20948_node",
            name="icm20948_node",
            parameters=[{
                "print": True,
                # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py
                # SparkFun address is likely 0x69, generic GY-ICM20948 - 0x68
                # Use "i2cdetect -y 1"
                "i2c_address": [0x68, 0x69],  # try both common addresses
                "frame_id": "imu_link",
                "raw_only": False,    # default False ("fusing" mode). When True - only publish raw IMU data - /imu/data_raw and /imu/mag
                "pub_rate_hz": pub_rate_hz,  # integer, default 50 in code, 200 here
                "temp_pub_rate_hz": 1.0,     # float, default 1.0
                "madgwick_beta": 0.1,        # 0.01–0.2 ballpark, weight of correction from accelerometer/magnetometer vs gyroscope
                "madgwick_use_mag": True,
                "startup_calib_seconds": 3.0,
                "gyro_calib_max_std_dps": 2.0,    # warning threshold - if std dev is too high during calibration; default 1.0
                "accel_calib_max_std_mps2": 0.35, # same for accel; default 0.35
                "magnetometer_bias": [-3.28, -25.93, 21.88]  # use ../tests/icm_calibrate_mag.py to find these values
            }]
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
                '--child-frame-id', 'imu_link' # Child frame ID
            ]
        )
    ])
