from smbus2 import SMBus
import time
import signal
import sys
import numpy as np

import icm_mag_lib

ICM_ADDRS = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

# initial values for biases and scales - will be updated in calibrateMagPrecise():
MagBias = np.array([0.0, 0.0, 0.0])   # microTesla, aligned with accel/gyro frame as published by ROS node
MagScale = np.array([1.0, 1.0, 1.0])  # should be 1.0 or omitted if "magnetometer_transform" is present
Magtransform = np.eye(3)  # magnetometer calibration is unknown initially. A 3x3 matrix, calculated in calibrateMagPrecise()

# current readings:
MagVals = np.array([0.0, 0.0, 0.0])     # microTesla, aligned with accel/gyro frame as published by ROS node
AccelVals = np.array([0.0, 0.0, 9.81])  # assume level and stationary at start

# calculated in computeOrientation():
roll = None
pitch = None
yaw = None

eps = 1e-6  # small number to avoid division by zero in calculations
min_field_uT = 1.0  # reject near-zero field magnitudes

def calibrateMagPrecise(bus, numSamples=1000):
    """Calibrate Magnetometer Use this method for more precise calculation
    
    This function uses ellipsoid fitting to get an estimate of the bias and
    transformation matrix required for mag data

    Note: Make sure you rotate the sensor in 8 shape and cover all the 
    pitch and roll angles.
    """

    global MagBias, Magtransform

    # make sure we are not applying calibration in readSensor():
    MagBias[:] = 0.0
    Magtransform = np.eye(3)

    samples = []
    while len(samples) < numSamples:
        m = icm_mag_lib.read_mag(bus)
        if m is None:
            continue
        m = np.asarray(m, dtype=float)
        if not np.all(np.isfinite(m)):
            continue
        if np.linalg.norm(m) < eps:
            continue
        samples.append(m)
        if len(samples) % 10 == 0:
            print(f"Calibration progress: {len(samples)}/{numSamples}", end="\r", flush=True)
        time.sleep(0.02)

    X = np.vstack(samples)
    centre, evecs, radii, v = __ellipsoid_fit(X)

    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    Magtransform = evecs.dot(D).dot(evecs.T)
    MagBias = centre

def __ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                x * x + z * z - 2 * y * y,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    M = D.dot(D.T)
    u = np.linalg.solve(M + 1e-9*np.eye(M.shape[0]), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                [v[3], v[1], v[5], v[7]],
                [v[4], v[5], v[2], v[8]],
                [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = center

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    S = R[:3, :3] / -R[3, 3]
    S = 0.5*(S + S.T)
    evals, evecs = np.linalg.eigh(S)
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))

    return center, evecs, radii, v

def computeOrientation(mag_vals):
    """ Computes roll, pitch and yaw

    The function uses accelerometer and magnetometer values
    to estimate roll, pitch and yaw. These values could be 
    having some noise, hence look at madgwick filter
    in filters folder to get a better estimate.
    
    """

    global AccelVals, roll, pitch, yaw

    roll = pitch = yaw = None

    # sanity check:
    if mag_vals is None:
        return

    if np.linalg.norm(AccelVals) < eps:
        return

    magLength = np.linalg.norm(mag_vals)
    if magLength < min_field_uT:
        return

    # Zeroes as we don't read accel values:
    roll_r  = np.arctan2(AccelVals[1], AccelVals[2])
    pitch_r = np.arctan2(-AccelVals[0], np.sqrt(AccelVals[1]**2 + AccelVals[2]**2))

    mx, my, mz = mag_vals / magLength

    # remove roll/pitch from mag to get "level" mag
    cr, sr = np.cos(roll_r),  np.sin(roll_r)
    cp, sp = np.cos(pitch_r), np.sin(pitch_r)

    mx2 = mx*cp + mz*sp
    my2 = mx*sr*sp + my*cr - mz*sr*cp

    # yaw_r = np.arctan2(my2, mx2)  # ENU convention: yaw=0 East, yaw=+90 North (CCW about +Up)
    yaw_r = np.arctan2(mx2, my2)  # NAV convention yaw: 0=North, +90=East

    roll = np.degrees(roll_r)
    pitch = np.degrees(pitch_r)
    yaw = np.degrees(yaw_r)
    yaw = (yaw + 180.0) % 360.0 - 180.0  # normalize to [-180, +180]

def apply_mag_cal(m):
    m = np.asarray(m, dtype=float).reshape(3,)
    if not np.all(np.isfinite(m)):
        return None  # defensively reject NaNs / bad shapes
    m_corr = m - MagBias
    m_corr = Magtransform @ m_corr
    m_corr = m_corr * MagScale
    return m_corr

def read_orientation(bus, count, message=""):
    """Read and print IMU orientation for specified number of iterations."""

    global MagVals, MagBias, MagScale, Magtransform, AccelVals, roll, pitch, yaw

    if message:
        print(message)
    
    for i in range(count):
        time.sleep(0.2)
        m = icm_mag_lib.read_mag(bus)  # aligned with accel/gyro frame as published by ROS node

        if m is None:
            print("Mag read failed")
            continue

        MagVals[0], MagVals[1], MagVals[2] = m  # microTesla, aligned with accel/gyro frame as published by ROS node

        MagVals_c = apply_mag_cal(MagVals)  # still microTesla, same frame, calibrated
        if MagVals_c is None:
            print("apply_mag_cal() failed")
            continue

        computeOrientation(MagVals_c)  # assume static level position for now, no Accel reading.

        if roll is None:
            print(f"MagVals: x={MagVals_c[0]:8.2f} y={MagVals_c[1]:8.2f} z={MagVals_c[2]:8.2f} µT    Orientation invalid (insufficient accel/mag)")
        else:
            print(f"MagVals: x={MagVals_c[0]:8.2f} y={MagVals_c[1]:8.2f} z={MagVals_c[2]:8.2f} µT    Orientation: roll:{roll:8.2f}   pitch:{pitch:8.2f}   yaw:{yaw:8.2f} degrees (Nav convention: 0=North, +90=East)")

def print_calibration():

    print("--------------------------------------------------------------------------------")
    print()
    print("---- Calibration results: copy this and paste into your ROS2 launch file:")
    print()
    print("#\"magnetometer_scale\": [" + ", ".join(f"{x}" for x in MagScale) + "],  # should be 1.0 or omitted if \"magnetometer_transform\" is present")
    print("\"magnetometer_bias\": [" + ", ".join(f"{x}" for x in MagBias) + "],")
    if Magtransform is not None:
        print("\"magnetometer_transform\": [")
        for i, row in enumerate(Magtransform):
            row_str = ", ".join(f"{x}" for x in row)
            if i < len(Magtransform) - 1:
                print(f"    {row_str},")
            else:
                print(f"    {row_str}],")
    print()
    print("---- Calibration results in Python for direct assignment (e.g. into read_mag.py)")
    print()
    # MagBias
    bias_str = ", ".join(f"{v:.16e}" for v in MagBias)
    print(f"MagBias = np.array([{bias_str}])")

    # Magtransform
    print("Magtransform = np.array([")
    for i, row in enumerate(Magtransform):
        row_str = ", ".join(f"{v:.16e}" for v in row)
        if i < len(Magtransform) - 1:
            print(f"    [{row_str}],")
        else:
            print(f"    [{row_str}]")
    print("])")
    print("--------------------------------------------------------------------------------")

def main():

    global MagVals, MagBias, MagScale, Magtransform, roll, pitch, yaw

    try:
        with SMBus(1) as bus:

            addr = icm_mag_lib.find_icm_address(bus, ICM_ADDRS)
            if addr is None:
                raise RuntimeError("No ICM-20948 found on I2C bus")

            print("OK: IMU initialized")

            # Note: Make sure you rotate the sensor in 8 shape and cover all the pitch and roll angles.

            read_orientation(bus, 5, "IP: Reading initial mag values and orientation")

            print("calibrating - rotate the sensor in 8 shape and cover all the pitch and roll angles")

            # Note: initial values for biases and scales as defined globally:
            #		MagBias = np.array([0.0, 0.0, 0.0])
            #		MagScale = np.array([1.0, 1.0, 1.0])  # should be 1.0 or omitted if "magnetometer_transform" is present
            #       Magtransform = np.eye(3)  # magnetometer calibration is unknown. A 3x3 matrix, calculated in calibrateMagPrecise()

            calibrateMagPrecise(bus)

            # Here we have the calculated calibration values in imu object. Print them for a ROS2 launch file:
            print_calibration()

            input("\nCalibration complete. Press Enter to see calibrated values...")

            """
            Run tests/icm_test_mag.py
            Rotate the robot in place.
            The published values should roughly conform to the following matrix:

              ENU    |    x    |    y    |    z    |
            ----------------------------------------   When robot rotates in place:
              North  |   +20   |     0   |   -40   |     N -> S  x changes from + to - (y stays the same around 0)
              East   |     0   |   +20   |   -40   |     E -> W  y changes from + to - (x stays the same around 0)
              South  |   -20   |     0   |   -40   |     z axis is Up; Earth field in the US typically has negative z (points down into Earth)
              West   |     0   |   -20   |   -40   |     z shouldn't change much
            ----------------------------------------
            values are in microTesla (µT), Earth's field is about 25 to 65 µT depending on location
            See https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml?#igrfwmm - magnetic field by location (microTesla, NED frame)
            """

            # Read and print calibrated values:
            read_orientation(bus, 10000, "IP: Reading mag values and orientation after calibration")

            print("Done")

    except Exception as e:
        print(f"I2C Error: {e}")

def signal_handler(sig, frame):
    print_calibration()
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()
