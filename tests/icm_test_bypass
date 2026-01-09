from smbus2 import SMBus
import time

# Try 0x69 if 0x68 fails
ADDR = 0x68

"""
@file icm_test_bypass.py
@brief A Test for AK09916 Magnetometer via ICM-20948 I2C Master interface.

This script configures the ICM-20948's internal I2C Master to communicate 
with the integrated AK09916 magnetometer.

After running this test, you should be able to see the magnetometer on the I2C bus
at address 0x0C (AK09916) if the bypass was successful.
    i2cdetect -y 1

@author [Sergei Grichine]
@date 2026-01-09
"""

with SMBus(1) as bus:
    # 1. Wake up ICM
    bus.write_byte_data(ADDR, 0x7F, 0x00) # Bank 0
    bus.write_byte_data(ADDR, 0x06, 0x01) # PWR_MGMT_1: Wake
    time.sleep(0.1)

    # 2. Enable Bypass
    bus.write_byte_data(ADDR, 0x03, 0x00) # USER_CTRL: Disable Master
    bus.write_byte_data(ADDR, 0x0F, 0x02) # INT_PIN_CFG: Enable Bypass
    time.sleep(0.1)

    # 3. Check if Magnetometer (0x0C) is now visible
    try:
        # Read WHO_AM_I from AK09916 (Register 0x01)
        wia2 = bus.read_byte_data(0x0C, 0x01)
        print(f"Bypass Success! Mag ID: 0x{wia2:02X}")
    except:
        print("Bypass Failed: Magnetometer 0x0C not found on bus.")

