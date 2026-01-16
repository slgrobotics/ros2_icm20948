from smbus2 import SMBus
import time
import signal
import sys
import icm_mag_lib

ICM_ADDR = 0x68  # 0x69 (Adafruit) or 0x68 (generic board)

def read_magnetometer(bus, offsets):

    print("Calibration finished. Reading magnetometer with applied offsets:")

    last = None
    while True:
        m = icm_mag_lib.read_mag(bus)
        if m is not None:
            mx, my, mz = m
            # Apply Hard-Iron Calibration Offsets (and additional zero-centering offsets in the XY plane):
            # Calibration Done! Offsets: X=-9.28, Y=-11.93, Z=21.88
            cmx = mx - offsets[0] # - (-9.28) -6.0
            cmy = my - offsets[1] # - (-11.93) + 14.0
            cmz = mz - offsets[2] # - (21.88)

            if last != m:
                print(f"Mag [µT] X:{mx:8.2f}  Y:{my:8.2f}  Z:{mz:8.2f}   Calibrated: X:{cmx:8.2f}  Y:{cmy:8.2f}  Z:{cmz:8.2f}")
                last = m
        time.sleep(1.0)

def calibrate_magnetometer(bus):

    print("Starting Calibration. Rotate sensor in all directions for 30 seconds...")

    mag_min = [9999, 9999, 9999]
    mag_max = [-9999, -9999, -9999]

    start_time = time.time()
    while time.time() - start_time < 30:  # Calibrate for 30 seconds
        m = icm_mag_lib.read_mag(bus)  # Use library read_mag function
        if m is not None:
            for i in range(3):
                if m[i] < mag_min[i]: mag_min[i] = m[i]
                if m[i] > mag_max[i]: mag_max[i] = m[i]
        time.sleep(0.05)

    # Calculate Hard Iron offsets
    offsets = [
        (mag_max[0] + mag_min[0]) / 2,
        (mag_max[1] + mag_min[1]) / 2,
        (mag_max[2] + mag_min[2]) / 2
    ]

    print(f"Calibration Done! Offsets: X={offsets[0]:.2f}, Y={offsets[1]:.2f}, Z={offsets[2]:.2f}")
    return offsets

def main():
    try:
        with SMBus(1) as bus:
            icm_mag_lib.enable_i2c_master(bus, ICM_ADDR) # Enable Master
            icm_mag_lib.mag_init(bus)          # Init Mag

            # Calibrate magnetometer ("hard-iron" offsets):
            offsets = calibrate_magnetometer(bus)

            # print it with calibration applied:
            read_magnetometer(bus, offsets)

    except Exception as e:
        print(f"I2C Error: {e}")

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    main()

