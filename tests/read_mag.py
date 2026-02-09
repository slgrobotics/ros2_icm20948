import time
import signal
import sys

import numpy as np

import icm_lib

#magnetometer_bias = [-12.53908133, -11.08949955, 23.40434220]  # values from previous calibration run
magnetometer_bias = [0.0, 0.0, 0.0]

POLL_DT_S = 0.2  # seconds

def format_sample(s):
    ax, ay, az = s["accel_mps2"]
    gx, gy, gz = s["gyro_rads"]
    mx, my, mz = s["mag_raw_uT"]
    cmx, cmy, cmz = s["mag_cal_uT"]

    return (
        f"Accel: [{ax:9.4f}, {ay:9.4f}, {az:9.4f}] m/s^2   "
        f"Gyro:  [{gx:9.4f}, {gy:9.4f}, {gz:9.4f}] rad/s   "
        f"Mag raw: [{mx:9.4f}, {my:9.4f}, {mz:9.4f}] uT   "
        f"Mag cal: [{cmx:9.4f}, {cmy:9.4f}, {cmz:9.4f}] uT"
    )

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
    imu, addr = icm_lib.find_imu()
    print(f"i2c_addr: 0x{addr:02x} ✓ (connected)")

    accel_mul, gyro_mul = icm_lib.configure_imu(imu)
    print(
        f"accel_fsr={icm_lib.qwiic_icm20948.gpm2} mul={accel_mul:.6g} m/s^2 per LSB, "
        f"gyro_fsr={icm_lib.qwiic_icm20948.dps250} mul={gyro_mul:.6g} rad/s per LSB"
    )

    while True:
        time.sleep(POLL_DT_S)  # we sleep on "continue" too
        try:
            s = icm_lib.read_sample(imu, accel_mul, gyro_mul, np.array(magnetometer_bias, dtype=float))
        except Exception as e:
            print(f"Error: getAgmt/read_sample failed: {e}")
            continue

        if s is None:
            continue

        print(format_sample(s))

def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()

