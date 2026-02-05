from smbus2 import SMBus
import time
import signal
import sys
import icm_mag_lib

import numpy as np

from pathlib import Path

# Add workspace src/ to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ros2_icm20948.i2c import qwiic_icm20948
from ros2_icm20948.helpers import G0, std_dev_from_sums, accel_raw_to_mps2, gyro_raw_to_rads

i2c_addresses = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

#magnetometer_bias = [-10.777835913962377, -11.856655801720644, 23.791090191349884]  # values from previous calibration run
magnetometer_bias = [0.0, 0.0, 0.0]

def read_magnetometer(bus):
    last = None
    while True:
        m = icm_mag_lib.read_mag(bus)
        if m is not None:
            mx, my, mz = m
            # Apply calibration bias - assuming it was calibrated in REP-103 body frame (x fwd, y left, z up):
            cmx = mx - magnetometer_bias[0]
            cmy = my - magnetometer_bias[1]
            cmz = mz - magnetometer_bias[2]

            # Note: if after applying calibration you rotate the sensor in XY plane
            #       and the values are not centered evenly around zero, apply additional adjustments.

            if last != m:
                print(f"Mag [µT] Raw: X:{mx:8.2f}  Y:{my:8.2f}  Z:{mz:8.2f}   Calibrated: X:{cmx:8.2f}  Y:{cmy:8.2f}  Z:{cmz:8.2f}")
                last = m
        time.sleep(0.2)

"""
Rotate the robot in place.
The published values should roughly conform to the following matrix:

           |    x    |    y    |    z    |
----------------------------------------   When robot rotates in place:
    North  |   +20   |     0   |   -40   |     N -> S  x changes from + to - (y stays the same around 0)
    East   |     0   |   +20   |   -40   |     E -> W  y changes from + to - (x stays the same around 0)
    South  |   -20   |     0   |   -40   |     z axis is Up; Earth field in the US typically has negative z (points down into Earth)
    West   |     0   |   -20   |   -40   |     z shouldn't change much
----------------------------------------
values are in microTesla (µT), Earth's field is about 25 to 65 µT depending on location
See https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml?#igrfwmm - magnetic field by location (microTesla, NED frame)
"""

def main():
    try:
        # Try each address until one connects
        imu = None
        i2c_addr = None
        for addr in i2c_addresses:
            try:
                test_imu = qwiic_icm20948.QwiicIcm20948(address=addr)
                if test_imu.connected:
                    imu = test_imu
                    i2c_addr = addr
                    print(f"   i2c_addr: 0x{i2c_addr:X} ✓ (connected)")
                    break
            except Exception as e:
                print(f"   i2c address 0x{addr:X} failed: {e}")
                continue

        if imu is None or not imu.connected:
            print("Error: ICM20948 not connected. Check wiring / I2C bus / addresses.")
            raise RuntimeError("ICM20948 not connected")

        # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py 

        accel_fsr = qwiic_icm20948.gpm2
        gyro_fsr  = qwiic_icm20948.dps250

        imu.setFullScaleRangeAccel(accel_fsr)
        imu.setFullScaleRangeGyro(gyro_fsr)

        _accel_mul = accel_raw_to_mps2(accel_fsr)
        _gyro_mul  = gyro_raw_to_rads(gyro_fsr)

        print(
            f"   accel_fsr={accel_fsr} mul={_accel_mul:.6g} m/s^2 per LSB, "
            f"gyro_fsr={gyro_fsr} mul={_gyro_mul:.6g} rad/s per LSB"
        )

        imu.begin()

        while True:
            time.sleep(0.2)
            if imu.dataReady():
                try:
                    imu.getAgmt()
                except Exception as e:
                    print(f"Error: ICM20948 getAgmt() failed: {e}")
                    continue

                if imu.mxRaw is None or imu.myRaw is None or imu.mzRaw is None:
                    print("Error: ICM20948 magnetometer data is None")
                    continue

                # --- Convert raw -> SI (already in REP-103 ENU: x fwd, y left, z up) ---
                ax_raw = imu.axRaw * _accel_mul
                ay_raw = imu.ayRaw * _accel_mul
                az_raw = imu.azRaw * _accel_mul

                gx_raw = imu.gxRaw * _gyro_mul
                gy_raw = imu.gyRaw * _gyro_mul
                gz_raw = imu.gzRaw * _gyro_mul

                # The imu.getAgmt() delivers "raw" magnetometer data in microTesla,
                #   in REP-103 body frame (x fwd, y left, z up), not calibrated.
                # Apply user mag offset (calibration parameter "magnetometer_bias", in microtesla):
                mx_uT = imu.mxRaw - float(magnetometer_bias[0])
                my_uT = imu.myRaw - float(magnetometer_bias[1])
                mz_uT = imu.mzRaw - float(magnetometer_bias[2])

                print(
                    f"Accel: [{ax_raw:9.4f}, {ay_raw:9.4f}, {az_raw:9.4f}] m/s^2   "
                    f"Gyro: [{gx_raw:9.4f}, {gy_raw:9.4f}, {gz_raw:9.4f}] rad/s   "
                    f"Mag raw: [{imu.mxRaw:9.4f}, {imu.myRaw:9.4f}, {imu.mzRaw:9.4f}] micro Tesla   "
                    f"Mag cal: [{mx_uT:9.4f}, {my_uT:9.4f}, {mz_uT:9.4f}] micro Tesla"
                )

    except Exception as e:
        print(f"I2C Error: {e}")

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()

