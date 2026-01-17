
import math
from .i2c import qwiic_icm20948

# Standard gravity (m/s^2), ISO 80000-3
G0 = 9.80665

def std_dev_from_sums(sum_, sumsq_, n):
    """
    Compute standard deviation from sum(x), sum(x^2), and count.

    This is numerically stable enough for IMU calibration windows.

    Args:
        sum_   : sum of samples
        sumsq_ : sum of squares of samples
        n      : number of samples (must be > 0)

    Returns:
        Standard deviation (float)
    """
    if n <= 0:
        return 0.0

    mean = sum_ / n
    var = (sumsq_ / n) - mean * mean

    # Guard against tiny negative due to FP error
    if var < 0.0:
        var = 0.0

    return math.sqrt(var)

# Accel full scale range options, G forces Plus or Minus (aka "gpm")
_ACCEL_LSB_PER_G = {
    qwiic_icm20948.gpm2:  16384.0, # library default, +-2 G, most sensitive
    qwiic_icm20948.gpm4:   8192.0,
    qwiic_icm20948.gpm8:   4096.0,
    qwiic_icm20948.gpm16:  2048.0,
}

# Gyro full scale range options, degrees per second (aka "dps")
_GYRO_LSB_PER_DPS = {
    qwiic_icm20948.dps250:  131.0, # library default, +-250 dps, most sensitive
    qwiic_icm20948.dps500:   65.5,
    qwiic_icm20948.dps1000:  32.8,
    qwiic_icm20948.dps2000:  16.4,
}

def accel_raw_to_mps2(scale_enum: int) -> float:
    if scale_enum not in _ACCEL_LSB_PER_G:
        raise ValueError(f"Unknown accel FSR enum: {scale_enum}")
    return G0 / _ACCEL_LSB_PER_G[scale_enum]

def gyro_raw_to_rads(scale_enum: int) -> float:
    if scale_enum not in _GYRO_LSB_PER_DPS:
        raise ValueError(f"Unknown gyro FSR enum: {scale_enum}")
    return (math.pi / 180.0) / _GYRO_LSB_PER_DPS[scale_enum]

