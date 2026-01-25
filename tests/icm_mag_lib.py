from smbus2 import SMBus
import time
import struct

"""
@file icm_mag_lib.py
@brief A simple library for AK09916 Magnetometer via ICM-20948 I2C Master interface.

Configures the ICM-20948's internal I2C Master to communicate 
with the integrated AK09916 magnetometer. It handles register bank switching, 
Slave 4 control transactions for initialization, and Slave 0 automated 
burst-reads for high-frequency magnetic field data acquisition.

@author [Sergei Grichine]
@date 2026-01-09
"""

icm_addr = 0x0   # will be set by caller, typically 0x69 (Adafruit) or 0x68 (generic board)
AK_ADDR  = 0x0C  # I2C address of AK09916 when bypass enabled

# ---- ICM bank select ----
REG_BANK_SEL = 0x7F

def set_bank(bus, bank):
    write_reg(bus, REG_BANK_SEL, (bank & 0x3) << 4)

def write_reg(bus, reg, val):
    bus.write_byte_data(icm_addr, reg, val)

def read_reg(bus, reg):
    return bus.read_byte_data(icm_addr, reg)

def read_block(bus, reg, n):
    return bus.read_i2c_block_data(icm_addr, reg, n)

# ---- BANK 3 regs (I2C master) ----
I2C_MST_CTRL = 0x01
I2C_SLV0_ADDR = 0x03
I2C_SLV0_REG  = 0x04
I2C_SLV0_CTRL = 0x05

I2C_SLV4_ADDR = 0x13
I2C_SLV4_REG  = 0x14
I2C_SLV4_DO   = 0x16
I2C_SLV4_CTRL = 0x15
I2C_SLV4_DI   = 0x17

# ---- AK09916 regs ----
AK_WIA2   = 0x01  # should be 0x09
AK_ST1    = 0x10
AK_HXL    = 0x11
AK_CNTL2  = 0x31

I2C_MST_STATUS = 0x17  # Bank 0
MST_ST_SLV4_DONE = 0x40
MST_ST_SLV4_NACK = 0x10

INT_PIN_CFG = 0x0F     # Bank 0 on ICM-20948
BYPASS_EN = 0x02
USER_CTRL = 0x03       # Bank 0
EXT_SENS_DATA_00 = 0x3B
I2C_MST_EN = 0x20
I2C_MST_RST = 0x02

PWR_MGMT_1 = 0x06          # Bank 0
I2C_MST_ODR_CONFIG = 0x00  # Bank 3

chip_wait_time = 0.01

def enable_i2c_master(bus, addr):

    global icm_addr
    icm_addr = addr

    print(f"Device address: 0x{icm_addr:x}")

    set_bank(bus, 0)
    write_reg(bus, PWR_MGMT_1, 0x01)  # wake, auto clock
    time.sleep(chip_wait_time)

    # Disable bypass
    v = read_reg(bus, INT_PIN_CFG)
    write_reg(bus, INT_PIN_CFG, v & ~BYPASS_EN)
    time.sleep(chip_wait_time)

    # Reset I2C master then enable it
    write_reg(bus, USER_CTRL, I2C_MST_RST)
    time.sleep(chip_wait_time)
    write_reg(bus, USER_CTRL, I2C_MST_EN)
    time.sleep(chip_wait_time)

    set_bank(bus, 3)
    write_reg(bus, I2C_MST_CTRL, 0x0D)        # 400 kHz; 0x07=345kHz may be safer
    write_reg(bus, I2C_MST_ODR_CONFIG, 0x04)  # ensure EXT_SENS updates
    time.sleep(chip_wait_time)

def slv4_txn(bus, dev_addr, reg_addr, data=0x00, read=False):
    # Ensure we are in Bank 3 for setup
    set_bank(bus, 3)

    # R/W bit: 0x80 for read, 0x00 for write
    addr_val = (dev_addr & 0x7F) | (0x80 if read else 0x00)
    write_reg(bus, I2C_SLV4_ADDR, addr_val)
    write_reg(bus, I2C_SLV4_REG, reg_addr)

    if not read:
        write_reg(bus, I2C_SLV4_DO, data)

    # Trigger transaction: 0x80 (Enable)
    write_reg(bus, I2C_SLV4_CTRL, 0x80)

    # CRITICAL: Wait for the Master state machine to latch the command
    # before you switch the Bank back to 0 to poll status.
    time.sleep(chip_wait_time)

    max_retries = 50
    for _ in range(max_retries):
        set_bank(bus, 0) # Poll status in Bank 0
        status = read_reg(bus, I2C_MST_STATUS)

        if status & 0x40: # I2C_SLV4_DONE
            if status & 0x10: # I2C_SLV4_NACK
                return None if read else False

            if read:
                set_bank(bus, 3)
                return read_reg(bus, I2C_SLV4_DI)
            return True
        time.sleep(chip_wait_time * 2)

    return None if read else False

def slv0_setup_read(bus, dev_addr, start_reg, nbytes):
    """Configure continuous read via SLV0 into EXT_SENS_DATA_00."""
    set_bank(bus, 3)
    write_reg(bus, I2C_SLV0_ADDR, (dev_addr & 0x7F) | 0x80)  # read
    write_reg(bus, I2C_SLV0_REG, start_reg)
    write_reg(bus, I2C_SLV0_CTRL, 0x80 | (nbytes & 0x0F))    # enable + length

def mag_init(bus):
    # Verify Mag ID (WIA2 should be 0x09)
    wia2 = slv4_txn(bus, AK_ADDR, 0x01, read=True) # 0x01 is AK_WIA2
    if wia2 is None:
        print("Error: Could not read Mag ID (Transaction failed)")
        return False
    if wia2 != 0x09:
        print(f"Error: Mag ID mismatch. Got 0x{wia2:02X}")
        return False

    print(f"Mag ID: 0x{wia2:02X}")

    # Reset Mag: AK_CNTL3 (0x32) = 0x01
    if not slv4_txn(bus, AK_ADDR, 0x32, 0x01):
        print("Failed to reset magnetometer")
        return False
    time.sleep(0.1)

    # Set Mode: AK_CNTL2 (0x31) = 0x08 (100Hz Continuous)
    slv4_txn(bus, AK_ADDR, 0x31, 0x08)

    # Setup continuous read of 9 bytes into EXT_SENS_DATA_00
    slv0_setup_read(bus, AK_ADDR, AK_ST1, 9)
    return True


def read_mag(bus):
    set_bank(bus, 0)
    # Read 9 bytes if slv0_setup_read was set to 9
    data = read_block(bus, EXT_SENS_DATA_00, 9)

    st1 = data[0]
    if not (st1 & 0x01):
        return None

    # Magnetic data is in bytes 1 to 7
    hx, hy, hz = struct.unpack_from('<hhh', bytes(data[1:7]))

    # Check ST2 (byte 8) for overflow (optional but recommended)
    st2 = data[8]
    if st2 & 0x08:
        return None  # Magnetic sensor overflow

    scale = 0.15 # µT/LSB
    return (hx * scale, hy * scale, hz * scale)  # microTesla

