# ros2_icm20948
Driver for the ICM-20948 IMU

**Note:** code from the following SparkFun packages is directly included in this package:

- https://pypi.org/project/sparkfun-qwiic-icm20948/#files
- https://pypi.org/project/sparkfun-qwiic-i2c/#files      (https://qwiic-i2c-py.readthedocs.io/en/latest/)

All credits and copyrights belong to SparkFun Electronics for these packages.

All credits and copyrights for the ros2_icm20948 driver belong to:
 - https://github.com/norlab-ulaval
 - https://github.com/JadenG13
 - https://github.com/GAVLab

## Dependencies
```bash
sudo apt install python3-smbus2
```

## Permissions
In order to run this node, i2c access permissions must be granted to the user that runs it. To do so run the following command: 
```bash
sudo adduser <your_user> i2c
```