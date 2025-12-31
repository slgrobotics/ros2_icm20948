import math

import rclpy
import sensor_msgs.msg
from rclpy.node import Node

from . import qwiic_icm20948
from .madgwick import MadgwickAHRS

G0 = 9.80665  # standard gravity

_ACCEL_LSB_PER_G = {
    qwiic_icm20948.gpm2:  16384.0,
    qwiic_icm20948.gpm4:   8192.0,
    qwiic_icm20948.gpm8:   4096.0,
    qwiic_icm20948.gpm16:  2048.0,
}

_GYRO_LSB_PER_DPS = {
    qwiic_icm20948.dps250:  131.0,
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

class ICM20948Node(Node):
    def __init__(self):
        super().__init__("icm20948_node")

        # Logger
        self.logger = self.get_logger()

        self.get_logger().info("IP: ICM20948 IMU Sensor node has been started")

        # Parameters
        self.declare_parameter("i2c_address", 0x68)
        self.i2c_addr = self.get_parameter("i2c_address").get_parameter_value().integer_value
        self.get_logger().info(f"   i2c_addr: 0x{self.i2c_addr:X}")

        # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py 

        self.declare_parameter("frame_id", "imu_icm20948")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.get_logger().info(f"   frame_id: {self.frame_id}")

        self.declare_parameter("pub_rate", 50)
        self.pub_rate = self.get_parameter("pub_rate").get_parameter_value().integer_value
        self.get_logger().info(f"   pub_rate: {self.pub_rate} Hz")

        # Madgwick params
        self.declare_parameter("madgwick_beta", 0.08)   # 0.04-0.2 typical
        self.declare_parameter("use_mag", True)
        self.beta = float(self.get_parameter("madgwick_beta").value)
        self.use_mag = bool(self.get_parameter("use_mag").value)
        self.filter = MadgwickAHRS(beta=self.beta)

        self._last_stamp = None
        self._shutting_down = False

        # choose FSRs, configure the device, and precompute multipliers once:
        self.accel_fsr = qwiic_icm20948.gpm16
        self.gyro_fsr  = qwiic_icm20948.dps2000

        self.imu.setFullScaleRangeGyro(self.gyro_fsr)
        self.imu.setFullScaleRangeAccel(self.accel_fsr)

        self._accel_mul = accel_raw_to_mps2(self.accel_fsr)
        self._gyro_mul  = gyro_raw_to_rads(self.gyro_fsr)

        self.get_logger().info(
            f"   accel_fsr={self.accel_fsr} mul={self._accel_mul:.6g} m/s^2 per LSB, "
            f"gyro_fsr={self.gyro_fsr} mul={self._gyro_mul:.6g} rad/s per LSB"
        )

        # IMU instance
        self.imu = qwiic_icm20948.QwiicIcm20948(address=self.i2c_addr)
        if not self.imu.connected:
            self.logger.error("ICM20948 not connected. Check wiring / I2C bus / address.")
        self.imu.begin()
        # the library’s begin() sets accel+gyro to defaults (gpm2, dps250). We override them here:
        self.imu.setFullScaleRangeAccel(qwiic_icm20948.gpm16)
        self.imu.setFullScaleRangeGyro(qwiic_icm20948.dps2000)

        # Publishers
        self.imu_raw_pub = self.create_publisher(sensor_msgs.msg.Imu, "/imu/data_raw", 10)
        self.imu_pub = self.create_publisher(sensor_msgs.msg.Imu, "/imu/data", 10)
        self.mag_pub = self.create_publisher(sensor_msgs.msg.MagneticField, "/imu/mag_raw", 10)
        self.temp_pub = self.create_publisher(sensor_msgs.msg.Temperature, "/imu/temp", 10)

        self.pub_clk = self.create_timer(1.0 / float(self.pub_rate), self.publish_cback)

        self.get_logger().info("OK: ICM20948 Node: init successful")

    def publish_cback(self):

        """
          standard convention is:
            /imu/data_raw = raw accel+gyro, orientation unknown (cov[0] = -1)
            /imu/data = accel+gyro + orientation estimated
            /imu/mag or /imu/mag_raw = magnetometer

          That keeps downstream packages (robot_localization, Nav2) happier.
        """

        if self._shutting_down:
            return

        try:
            now = self.get_clock().now()

            # Compute dt for filter
            if self._last_stamp is None:
                dt = 1.0 / float(self.pub_rate)
            else:
                dt = (now - self._last_stamp).nanoseconds * 1e-9
                # Clamp dt to sane bounds (prevents huge jumps if system pauses)
                if dt <= 0.0:
                    dt = 1.0 / float(self.pub_rate)
                elif dt > 0.2:
                    dt = 0.2
            self._last_stamp = now

            imu_raw_msg = sensor_msgs.msg.Imu()
            imu_msg = sensor_msgs.msg.Imu()
            mag_msg = sensor_msgs.msg.MagneticField()
            temp_msg = sensor_msgs.msg.Temperature()

            imu_raw_msg.header.stamp = now.to_msg()
            imu_raw_msg.header.frame_id = self.frame_id

            imu_msg.header.stamp = imu_raw_msg.header.stamp
            imu_msg.header.frame_id = self.frame_id

            mag_msg.header.stamp = imu_raw_msg.header.stamp
            mag_msg.header.frame_id = self.frame_id

            temp_msg.header.stamp = imu_raw_msg.header.stamp
            temp_msg.header.frame_id = self.frame_id

            if self.imu.dataReady():
                try:
                    self.imu.getAgmt()
                except Exception as e:
                    self.logger.error(str(e))
                    # Publish empty messages with timestamps anyway
                    self.imu_raw_pub.publish(imu_raw_msg)
                    self.imu_pub.publish(imu_msg)
                    self.mag_pub.publish(mag_msg)
                    self.temp_pub.publish(temp_msg)
                    return

                # ---- Convert raw -> SI units (you already do this) ----
                # Accel (m/s^2) -- your scaling assumes gpm16 => 2048 LSB/g; keep if correct for your config
                ax = self.imu.axRaw * self._accel_mul
                ay = self.imu.ayRaw * self._accel_mul
                az = self.imu.azRaw * self._accel_mul

                # Gyro (rad/s) -- your scaling assumes dps2000 => 16.4 LSB/(deg/s); keep if correct for your config
                gx = self.imu.gxRaw * self._gyro_mul
                gy = self.imu.gyRaw * self._gyro_mul
                gz = self.imu.gzRaw * self._gyro_mul

                # Mag (Tesla) -- your scaling may need calibration; leave as-is for now
                mx = self.imu.mxRaw * 1e-6 / 0.15
                my = self.imu.myRaw * 1e-6 / 0.15
                mz = self.imu.mzRaw * 1e-6 / 0.15

                # Fill raw message (no orientation)
                imu_raw_msg.linear_acceleration.x = ax
                imu_raw_msg.linear_acceleration.y = ay
                imu_raw_msg.linear_acceleration.z = az
                imu_raw_msg.angular_velocity.x = gx
                imu_raw_msg.angular_velocity.y = gy
                imu_raw_msg.angular_velocity.z = gz
                imu_raw_msg.orientation_covariance[0] = -1.0

                # Fill mag message
                mag_msg.magnetic_field.x = mx
                mag_msg.magnetic_field.y = my
                mag_msg.magnetic_field.z = mz

                # ---- Run Madgwick to compute orientation ----
                if self.use_mag:
                    self.filter.update(gx, gy, gz, ax, ay, az, mx, my, mz, dt=dt)
                else:
                    self.filter.update(gx, gy, gz, ax, ay, az, dt=dt)

                qx, qy, qz, qw = self.filter.quaternion_xyzw()

                # Fill fused IMU message: copy accel/gyro + add orientation
                imu_msg.linear_acceleration = imu_raw_msg.linear_acceleration
                imu_msg.angular_velocity = imu_raw_msg.angular_velocity

                imu_msg.orientation.x = qx
                imu_msg.orientation.y = qy
                imu_msg.orientation.z = qz
                imu_msg.orientation.w = qw

                # Provide non-negative covariances (tune later)
                imu_msg.orientation_covariance[0] = 0.05
                imu_msg.orientation_covariance[4] = 0.05
                imu_msg.orientation_covariance[8] = 0.10

                imu_msg.angular_velocity_covariance[0] = 0.02
                imu_msg.angular_velocity_covariance[4] = 0.02
                imu_msg.angular_velocity_covariance[8] = 0.02

                imu_msg.linear_acceleration_covariance[0] = 0.10
                imu_msg.linear_acceleration_covariance[4] = 0.10
                imu_msg.linear_acceleration_covariance[8] = 0.10

                temp_msg.temperature = self.imu.tmpRaw / 100.0
                temp_msg.variance = 0.0 # 0 means unknown

            else:
                # If no new data, keep publishing timestamps but mark orientation unknown
                imu_raw_msg.orientation_covariance[0] = -1.0
                imu_msg.orientation_covariance[0] = -1.0

            self.imu_raw_pub.publish(imu_raw_msg)
            self.imu_pub.publish(imu_msg)
            self.mag_pub.publish(mag_msg)
            self.temp_pub.publish(temp_msg)

        except Exception as e:
            # During shutdown, suppress noise; otherwise log
            if not self._shutting_down:
                self.logger.error(f"publish_cback exception: {e}")

def destroy_node(self):
    self._shutting_down = True
    try:
        if hasattr(self, "pub_clk_") and self.pub_clk_ is not None:
            self.pub_clk_.cancel()
    except Exception:
        pass
    return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ICM20948Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C received, shutting down...")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
