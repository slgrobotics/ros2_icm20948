from smbus2 import SMBus
import time
import signal
import sys
import icm_mag_lib

ICM_ADDRS = [0x68, 0x69]  # 0x69 (Adafruit) or 0x68 (generic board)

#magnetometer_bias = [-10.777835913962377, -11.856655801720644, 23.791090191349884]  # values from previous calibration run
magnetometer_bias = [0.0, 0.0, 0.0]

def read_magnetometer(bus):
    last = None
    while True:
        m = icm_mag_lib.read_mag_enu(bus)
        if m is not None:
            mx, my, mz = m
            # Apply calibration bias - assuming it was calibrated in ENU frame:
            cmx = mx - magnetometer_bias[0]
            cmy = my - magnetometer_bias[1]
            cmz = mz - magnetometer_bias[2]

            # Note: if after applying calibration you rotate the sensor in XY plane
            #       and the values are not centered evenly around zero, apply additional adjustments.

            if last != m:
                print(f"Mag [µT] Raw: X:{mx:8.2f}  Y:{my:8.2f}  Z:{mz:8.2f}   Calibrated in ENU frame: X:{cmx:8.2f}  Y:{cmy:8.2f}  Z:{cmz:8.2f}")
                last = m
        time.sleep(0.2)

"""
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

