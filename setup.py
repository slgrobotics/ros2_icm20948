import os
from glob import glob
from setuptools import setup

"""
ROS2 setup file for the ros2_icm20948 package.

Note: code from the following SparkFun packages is directly included in this package:

- https://pypi.org/project/sparkfun-qwiic-icm20948/#files
- https://pypi.org/project/sparkfun-qwiic-i2c/#files      (https://qwiic-i2c-py.readthedocs.io/en/latest/)

All credits and copyrights belong to SparkFun Electronics for these packages.

"""

package_name = "ros2_icm20948"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include all launch files.
        (os.path.join("share", package_name), glob("launch/*launch.[pxy][yma]*")),
    ],
    install_requires=[
        "setuptools"
    ],
    zip_safe=True,
    maintainer="Simon-Pierre Deschênes",
    maintainer_email="simon-pierre.deschenes.1@ulaval.ca",
    description="Driver for the ICM-20948 IMU",
    license="BSD-2.0",
    entry_points={
        "console_scripts": [
            "icm20948_node = ros2_icm20948.icm20948_node:main",
        ],
    },
)
