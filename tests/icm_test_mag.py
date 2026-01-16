from smbus2 import SMBus
import time
import signal
import sys
import icm_mag_lib

ICM_ADDR = 0x68  # 0x69 (Adafruit) or 0x68 (generic board)

def read_magnetometer(bus):
    last = None
    while True:
        m = icm_mag_lib.read_mag(bus)
        if m is not None:
            mx, my, mz = m
            # Apply Hard-Iron Calibration Offsets:
            #       Calibration Done! Offsets: X=-9.28, Y=-11.93, Z=21.88
            cmx = mx # - (-9.28) -6.0
            cmy = my # - (-11.93) + 14.0
            cmz = mz # - (21.88)

            # Note: if after applying calibration you rotate the sensor in XY plane
            #       and the values are not centered evenly around zero, apply additional adjustments.

            if last != m:
                print(f"Mag [µT] X:{mx:8.2f}  Y:{my:8.2f}  Z:{mz:8.2f}   Calibrated: X:{cmx:8.2f}  Y:{cmy:8.2f}  Z:{cmz:8.2f}")
                last = m
        time.sleep(0.2)

def main():
    try:
        with SMBus(1) as bus:
            icm_mag_lib.enable_i2c_master(bus, ICM_ADDR) # Enable Master
            icm_mag_lib.mag_init(bus)          # Init Mag

            read_magnetometer(bus)
            #calibrate_magnetometer(bus)

    except Exception as e:
        print(f"I2C Error: {e}")

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()

