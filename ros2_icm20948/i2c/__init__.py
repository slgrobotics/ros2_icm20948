"""I2C driver package for ICM20948 IMU."""

from .qwiic_i2c import getI2CDriver, get_i2c_driver, isDeviceConnected, ping
from .qwiic_icm20948 import QwiicIcm20948

__all__ = [
    "getI2CDriver",
    "get_i2c_driver",
    "isDeviceConnected",
    "ping",
    "QwiicIcm20948",
]
