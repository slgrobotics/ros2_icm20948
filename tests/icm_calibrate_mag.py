from smbus2 import SMBus
import time
import signal
import sys
import numpy as np

import icm_mag_lib

ICM_ADDRS = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

# initial values for biases and scales:
MagBias = np.array([0.0, 0.0, 0.0])
Mags = np.array([1.0, 1.0, 1.0])      # optional magnetometer scale adjustment
Magtransform = None  # magnetometer calibration is unknown. A 3x3 matrix, calculated in calibrateMagPrecise()
MagVals = np.array([0.0, 0.0, 0.0])
AccelVals = np.array([0.0, 0.0, 9.81])  # assume stationary at start
roll = 0.0
pitch = 0.0
yaw = 0.0
eps = 1e-9

def apply_mag_cal(m):
    # axis remap should happen before this if needed
    m = np.asarray(m, dtype=float)
    m_corr = m - MagBias
    if Magtransform is not None:
        m_corr = Magtransform @ m_corr
    m_corr = m_corr * Mags
    return m_corr

def calibrateMagPrecise(bus, numSamples=1000):
    """Calibrate Magnetometer Use this method for more precise calculation
    
    This function uses ellipsoid fitting to get an estimate of the bias and
    transformation matrix required for mag data

    Note: Make sure you rotate the sensor in 8 shape and cover all the 
    pitch and roll angles.
    """

    global MagBias, Magtransform

    samples = []
    while len(samples) < numSamples:
        m = icm_mag_lib.read_mag_enu(bus)
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
    r = (a*b*c) ** (1./3.)
    Dm = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    Magtransform = evecs.dot(Dm).dot(evecs.T)
    MagBias = centre
    #MagBias[2] = -MagBias[2]  # change in z bias


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
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
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
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v

def computeOrientation(mag_vals):
    """ Computes roll, pitch and yaw

    The function uses accelerometer and magnetometer values
    to estimate roll, pitch and yaw. These values could be 
    having some noise, hence look at madgwick filter
    in filters folder to get a better estimate.
    
    """

    global AccelVals, roll, pitch, yaw

    if np.linalg.norm(AccelVals) < eps: return
    if np.linalg.norm(mag_vals) < eps: return

    roll  = np.arctan2(AccelVals[1], AccelVals[2])
    pitch = np.arctan2(-AccelVals[0], np.sqrt(AccelVals[1]**2 + AccelVals[2]**2))
    magLength = np.linalg.norm(mag_vals)
    if magLength < eps: return
    normMagVals = mag_vals / magLength
    yaw = np.arctan2(np.sin(roll)*normMagVals[2] - np.cos(roll)*normMagVals[1],\
                np.cos(pitch)*normMagVals[0] + np.sin(roll)*np.sin(pitch)*normMagVals[1] \
                + np.cos(roll)*np.sin(pitch)*normMagVals[2])

    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)


def read_orientation(bus, count, message=""):
    """Read and print IMU orientation for specified number of iterations."""

    global MagVals, MagBias, Magtransform, Mags, AccelVals, roll, pitch, yaw

    if message:
        print(message)
    
    for i in range(count):
        time.sleep(0.2)
        m = icm_mag_lib.read_mag_enu(bus)

        if m is None:
            print("Mag read failed")
            continue

        MagVals[0], MagVals[1], MagVals[2] = m  # microTesla

        MagVals = MagVals * 1e-6   # Tesla

        MagVals_c = apply_mag_cal(MagVals)

        computeOrientation(MagVals_c)  # assume static level position for now, no Accel reading.

        MagVals_uT = MagVals * 1e6  # convert to microTesla for printing

        print(f"MagVals: x={MagVals_uT[0]:8.2f} y={MagVals_uT[1]:8.2f} z={MagVals_uT[2]:8.2f} µT    roll:{roll:8.2f}     pitch:{pitch:8.2f}     yaw:{yaw:8.2f} degrees")

def print_calibration():

    print()
    print("Calibration results: copy this and paste into your ROS2 launch file:")
    print()
    print("\"magnetometer_scale\": [" + ", ".join(f"{x}" for x in Mags) + "],  # should be around 1.0")
    print("\"magnetometer_bias\": [" + ", ".join(f"{x}" for x in MagBias) + "],")
    if Magtransform is not None:
        print("\"magnetometer_transform\": [")
        for i, row in enumerate(Magtransform):
            row_str = ", ".join(f"{x}" for x in row)
            if i < len(Magtransform) - 1:
                print(f"    {row_str},")
            else:
                print(f"    {row_str}]")
    print()


def main():

    global MagVals, MagBias, Magtransform, Mags, roll, pitch, yaw

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
            #		Mags = np.array([1.0, 1.0, 1.0])      # optional magnetometer scale adjustment
            #       Magtransform = None  # magnetometer calibration is unknown. A 3x3 matrix, calculated in calibrateMagPrecise()

            calibrateMagPrecise(bus)

            # Here we have the calculated calibration values in imu object. Print them for a ROS2 launch file:
            print_calibration()

            input("\nCalibration complete. Press Enter to see calibrated values...")

            """
            Run tests/icm_test_mag.py
            Rotate the robot in place.
            The published values should roughly conform to the following matrix:

              ENU    |    x    |    y    |    z    |
            ----------------------------------------     When robot rotates in place:
              North  |     0   |   +20   |   -40   |     N -> S  y changes from + to - (x stays the same)
              East   |   +20   |     0   |   -40   |     E -> W  x changes from + to - (y stays the same)
              South  |     0   |   -20   |   -40   |     z looks down and doesn't change much
              West   |   -20   |     0   |   -40   |
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
