import sys
from pathlib import Path

import numpy as np

# Add workspace src/ to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ros2_icm20948.i2c import qwiic_icm20948
from ros2_icm20948.helpers import accel_raw_to_mps2, gyro_raw_to_rads

I2C_ADDRESSES = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

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

def read_sample(imu, accel_mul, gyro_mul, mag_bias_uT=None, mag_transform=None, mag_scale=None):
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

    # apply calibration only if requested:
    m_cal = m_raw
    if mag_bias_uT is not None:
        m_cal = m_cal - mag_bias_uT
    if mag_transform is not None:
        m_cal = mag_transform @ m_cal
    if mag_scale is not None:
        m_cal = m_cal * mag_scale

    return {
        "accel_mps2": (ax, ay, az),
        "gyro_rads": (gx, gy, gz),
        "mag_raw_uT": tuple(m_raw),
        "mag_cal_uT": tuple(m_cal),  # m_raw if no calibration requested
        "temp_raw": imu.temperatureRaw,
    }

