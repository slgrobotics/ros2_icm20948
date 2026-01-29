from smbus2 import SMBus
import time
import signal
import sys
import icm_mag_lib

ICM_ADDRS = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

#magnetometer_bias = [-12.578835747278259, -10.999672868949329, 23.65817527842366]
magnetometer_bias = [0.0, 0.0, 0.0]

def read_magnetometer(bus):
    last = None
    while True:
        m = icm_mag_lib.read_mag(bus)
        if m is not None:
            mx, my, mz = m
            # Apply calibration bias:
            cmx = mx - magnetometer_bias[0]
            cmy = -(my - magnetometer_bias[1])  # flip Y axis
            cmz = -(mz - magnetometer_bias[2])  # flip Z axis

            # Note: if after applying calibration you rotate the sensor in XY plane
            #       and the values are not centered evenly around zero, apply additional adjustments.

            if last != m:
                print(f"Mag [µT] Raw: X:{mx:8.2f}  Y:{my:8.2f}  Z:{mz:8.2f}   Calibrated: X:{cmx:8.2f}  Y:{cmy:8.2f}  Z:{cmz:8.2f}")
                last = m
        time.sleep(0.2)

"""
Rotate the robot in place.
The published values should roughly conform to the following matrix:

         |   x   |   y   |   z   |
----------------------------------
  North  |  20   |   0   |  -40  |
  East   |   0   |  20   |  -40  |
  South  | -20   |   0   |  -40  |
  West   |   0   |  -20  |  -40  |
----------------------------------
"""

def main():
    try:
        with SMBus(1) as bus:
            addr = icm_mag_lib.find_icm_address(bus, ICM_ADDRS)
            if addr is None:
                raise RuntimeError("No ICM-20948 found on I2C bus")

            read_magnetometer(bus)

    except Exception as e:
        print(f"I2C Error: {e}")

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()

