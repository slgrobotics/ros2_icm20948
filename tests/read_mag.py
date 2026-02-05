import time
import signal
import sys
from pathlib import Path

import numpy as np

import icm_mag_lib

# Add workspace src/ to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ros2_icm20948.i2c import qwiic_icm20948
from ros2_icm20948.helpers import accel_raw_to_mps2, gyro_raw_to_rads

I2C_ADDRESSES = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)
POLL_DT_S = 0.2  # seconds

#magnetometer_bias = [-10.777835913962377, -11.856655801720644, 23.791090191349884]  # values from previous calibration run
magnetometer_bias = [0.0, 0.0, 0.0]

def find_imu(i2c_addresses=I2C_ADDRESSES):
    """
    Try each address, return (imu, addr). Raises RuntimeError if none found.
    """
    last_exc = None
    for addr in i2c_addresses:
        try:
            imu = qwiic_icm20948.QwiicIcm20948(address=addr)
            if imu.connected:
                return imu, addr
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"ICM20948 not connected (tried {i2c_addresses}). Last error: {last_exc}")


def configure_imu(imu, accel_fsr=qwiic_icm20948.gpm2, gyro_fsr=qwiic_icm20948.dps250):
    """
    Initialize IMU and set FSRs. Returns (accel_mul, gyro_mul).
    """
    # Best practice: begin() first (ensures device is configured / mag started)
    if not imu.begin():
        raise RuntimeError("imu.begin() returned False")

    # Override ranges after begin()
    imu.setFullScaleRangeAccel(accel_fsr)
    imu.setFullScaleRangeGyro(gyro_fsr)

    accel_mul = accel_raw_to_mps2(accel_fsr)
    gyro_mul = gyro_raw_to_rads(gyro_fsr)
    return accel_mul, gyro_mul

def read_sample(imu, accel_mul, gyro_mul, mag_bias_uT):
    """
    Read one sample if ready; returns dict or None if not ready / invalid mag.
    """
    if not imu.dataReady():
        print("read_sample(): data not ready")
        return None

    imu.getAgmt()

    # If mag isn't ready yet, SparkFun code may keep old values or None depending on your getAgmt() edits.
    if imu.mxRaw is None or imu.myRaw is None or imu.mzRaw is None:
        print("read_sample(): mag invalid")
        return None

    # Convert accel/gyro
    ax = imu.axRaw * accel_mul
    ay = imu.ayRaw * accel_mul
    az = imu.azRaw * accel_mul

    gx = imu.gxRaw * gyro_mul
    gy = imu.gyRaw * gyro_mul
    gz = imu.gzRaw * gyro_mul

    # Magnetometer in microTesla (already in your REP-103 aligned body frame)
    m_raw = np.array([imu.mxRaw, imu.myRaw, imu.mzRaw], dtype=float)
    m_cal = m_raw - mag_bias_uT

    return {
        "accel_mps2": (ax, ay, az),
        "gyro_rads": (gx, gy, gz),
        "mag_raw_uT": tuple(m_raw),
        "mag_cal_uT": tuple(m_cal),
        "temp_raw": imu.tmpRaw,
    }


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
    imu, addr = find_imu()
    print(f"i2c_addr: 0x{addr:X} ✓ (connected)")

    accel_mul, gyro_mul = configure_imu(imu)
    print(
        f"accel_fsr={qwiic_icm20948.gpm2} mul={accel_mul:.6g} m/s^2 per LSB, "
        f"gyro_fsr={qwiic_icm20948.dps250} mul={gyro_mul:.6g} rad/s per LSB"
    )

    while True:
        time.sleep(POLL_DT_S)
        try:
            s = read_sample(imu, accel_mul, gyro_mul, np.array(magnetometer_bias, dtype=float))
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

