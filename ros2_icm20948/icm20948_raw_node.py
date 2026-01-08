import math

import rclpy
import sensor_msgs.msg
from rclpy.node import Node
import numpy as np

from . import qwiic_icm20948
from .helpers import G0, std_dev_from_sums, accel_raw_to_mps2, gyro_raw_to_rads

class ICM20948RawNode(Node):
    def __init__(self):
        super().__init__("icm20948_raw_node")

        # Logger
        self.logger = self.get_logger()

        self.logger.info("IP: ICM20948 IMU Sensor RAW node has been started")

        # Parameters
        self.declare_parameter("i2c_address", 0x68)
        self.i2c_addr = self.get_parameter("i2c_address").get_parameter_value().integer_value
        self.logger.info(f"   i2c_addr: 0x{self.i2c_addr:X}")

        # Note: for Linux on Raspberry Pi iBus=1 is hardcoded in linux_i2c.py 

        self.declare_parameter("frame_id", "imu_icm20948")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.logger.info(f"   frame_id: {self.frame_id}")

        self.declare_parameter("pub_rate_hz", 50)
        self.pub_rate_hz = self.get_parameter("pub_rate_hz").get_parameter_value().integer_value
        self.logger.info(f"   pub_rate_hz: {self.pub_rate_hz} Hz")
        
        self.declare_parameter("temp_pub_rate_hz", 1.0)
        self.temp_pub_rate_hz = float(self.get_parameter("temp_pub_rate_hz").value)
        self.logger.info(f"   temp_pub_rate_hz: {self.temp_pub_rate_hz} Hz")

        # Temperature averaging (accumulate at IMU rate, publish averaged at ~temp_pub_rate_hz)
        self._temp_sum_c = 0.0
        self._temp_count = 0
        # Divider: publish temperature every N IMU ticks
        self._temp_div = max(1, int(round(self.pub_rate_hz / max(0.1, self.temp_pub_rate_hz))))

        # Gyro and Accel calibration on startup:
        self.declare_parameter("startup_calib_seconds", 3.0)
        self.startup_calib_seconds = float(self.get_parameter("startup_calib_seconds").value)

        self.declare_parameter("gyro_calib_max_std_dps", 2.0)  # warn if more. Usually measures around 1.7 deg/s
        self.gyro_calib_max_std_dps = float(self.get_parameter("gyro_calib_max_std_dps").value)

        self._gyro_bias = [0.0, 0.0, 0.0]      # rad/s
        self._gyro_sum = [0.0, 0.0, 0.0]
        self._gyro_sumsq = [0.0, 0.0, 0.0]

        self.declare_parameter("accel_calib_max_std_mps2", 0.35)  # warn if more, Usually measures around 0.06 m/s^2
        self.accel_calib_max_std_mps2 = float(self.get_parameter("accel_calib_max_std_mps2").value)

        self._accel_bias = [0.0, 0.0, 0.0]     # m/s^2
        self._accel_sum = [0.0, 0.0, 0.0]
        self._accel_sumsq = [0.0, 0.0, 0.0]

        self.logger.info(f"   startup_calib_seconds: {self.startup_calib_seconds}   gyro_calib_max_std_dps: {self.gyro_calib_max_std_dps}  accel_calib_max_std_mps2: {self.accel_calib_max_std_mps2}")

        # Mag calibration accumulators:
        self._mag_avg_calib = [0.0, 0.0, 0.0]  # Tesla
        self._mag_sum = [0.0, 0.0, 0.0]

        self._calib_start_time = self.get_clock().now()
        self._calib_samples = 0
        self._calibration_done = False
        self._shutting_down = False

        # IMU instance
        self.imu = qwiic_icm20948.QwiicIcm20948(address=self.i2c_addr)
        if not self.imu.connected:
            self.logger.error("ICM20948 not connected. Check wiring / I2C bus / address.")
        self.imu.begin()

        # Choose FSRs, configure the device, and precompute multipliers once.
    
        # the library’s begin() sets accel+gyro to defaults (gpm2, dps250). We may override them here.
        # Practical guidance: https://chatgpt.com/s/t_6956bac992908191894dca63ff53b68d
        # Accel: ±2g (gpm2) Gyro: ±250 dps (dps250) for wheeled home robot, most sensitive to movement.
        self.accel_fsr = qwiic_icm20948.gpm2
        self.gyro_fsr  = qwiic_icm20948.dps250

        self.imu.setFullScaleRangeAccel(self.accel_fsr)
        self.imu.setFullScaleRangeGyro(self.gyro_fsr)

        self._accel_mul = accel_raw_to_mps2(self.accel_fsr)
        self._gyro_mul  = gyro_raw_to_rads(self.gyro_fsr)

        self.logger.info(
            f"   accel_fsr={self.accel_fsr} mul={self._accel_mul:.6g} m/s^2 per LSB, "
            f"gyro_fsr={self.gyro_fsr} mul={self._gyro_mul:.6g} rad/s per LSB"
        )

        # Publishers
        self.imu_raw_pub = self.create_publisher(sensor_msgs.msg.Imu, "/imu/data_raw", 10)
        self.mag_pub = self.create_publisher(sensor_msgs.msg.MagneticField, "/imu/mag_raw", 10)
        self.temp_pub = self.create_publisher(sensor_msgs.msg.Temperature, "/imu/temp", 10)

        self.pub_clk = self.create_timer(1.0 / float(self.pub_rate_hz), self.publish_cback)

        self.logger.info("OK: ICM20948 RAW Node: init successful")

    #
    # callback called at pub_rate_hz:
    #
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
            imu_raw_msg = sensor_msgs.msg.Imu()
            mag_msg = sensor_msgs.msg.MagneticField()

            now = self.get_clock().now()

            imu_raw_msg.header.stamp = now.to_msg()
            imu_raw_msg.header.frame_id = self.frame_id

            mag_msg.header.stamp = imu_raw_msg.header.stamp
            mag_msg.header.frame_id = self.frame_id
            # mag covariance unknown for now - uncalibrated mag, no noise model
            mag_msg.magnetic_field_covariance[0] = -1.0

            # If no new data, keep publishing timestamps but mark orientation etc. unknown
            imu_raw_msg.orientation_covariance[0] = -1.0
            imu_raw_msg.linear_acceleration_covariance[0] = -1.0
            imu_raw_msg.angular_velocity_covariance[0] = -1.0

            if self.imu.dataReady():
                try:
                    self.imu.getAgmt()
                except Exception as e:
                    self.logger.error(str(e))
                    # Publish empty messages with timestamps anyway
                    self.imu_raw_pub.publish(imu_raw_msg)
                    self.mag_pub.publish(mag_msg)
                    return

                # ---- Convert raw -> SI units ----

                # keep raw values per ROS convention:
                #     /imu/data_raw: unfused, unfiltered, unmodified accel+gyro (orientation unknown)

                # Accel (m/s^2) -- apply our scaling;
                ax = ax_raw = self.imu.axRaw * self._accel_mul
                ay = ay_raw = self.imu.ayRaw * self._accel_mul
                az = az_raw = self.imu.azRaw * self._accel_mul

                # Gyro (rad/s) -- apply our scaling;
                gx = gx_raw = self.imu.gxRaw * self._gyro_mul
                gy = gy_raw = self.imu.gyRaw * self._gyro_mul
                gz = gz_raw = self.imu.gzRaw * self._gyro_mul

                # Mag (micro Teslas for printing and averaging)
                # The Conversion Formula Multiply the raw 16-bit integer (LSB) by 0.1499 to get the value in microTeslas,
                #  then (at publishing) multiply by 10^-6 to convert to Teslas. 
                mag_mul = 0.1499  # Sensitivity Scale Factor: 0.1499 uT/LSB
                mx = self.imu.mxRaw * mag_mul
                my = self.imu.myRaw * mag_mul
                mz = self.imu.mzRaw * mag_mul

                # ---- Gyro and Accel biases calibration phase ----
                if not self._calibration_done:
                    self._gyro_sum[0] += gx
                    self._gyro_sum[1] += gy
                    self._gyro_sum[2] += gz
                    self._gyro_sumsq[0] += gx*gx
                    self._gyro_sumsq[1] += gy*gy
                    self._gyro_sumsq[2] += gz*gz

                    self._accel_sum[0] += ax
                    self._accel_sum[1] += ay
                    self._accel_sum[2] += az
                    self._accel_sumsq[0] += ax*ax
                    self._accel_sumsq[1] += ay*ay
                    self._accel_sumsq[2] += az*az

                    # to get measurement average during calibration:
                    self._mag_sum[0] += mx
                    self._mag_sum[1] += my
                    self._mag_sum[2] += mz

                    self._calib_samples += 1

                    elapsed = (now - self._calib_start_time).nanoseconds * 1e-9
                    if elapsed >= self.startup_calib_seconds and self._calib_samples > 50:
                        self._calibration_done = True
                        self._madgwick_updates = 0
                        self._orientation_valid = False

                        n = float(self._calib_samples)

                        self.logger.info(f"IMU biases calibrated over {elapsed:.2f}s - ({int(n)} samples): ")

                        # Gyro calibration calculations:
                        bgx = self._gyro_sum[0] / n
                        bgy = self._gyro_sum[1] / n
                        bgz = self._gyro_sum[2] / n
                        self._gyro_bias = [bgx, bgy, bgz]

                        g_sx = std_dev_from_sums(self._gyro_sum[0], self._gyro_sumsq[0], n) * 180.0 / math.pi  # convert to deg/s for readability
                        g_sy = std_dev_from_sums(self._gyro_sum[1], self._gyro_sumsq[1], n) * 180.0 / math.pi
                        g_sz = std_dev_from_sums(self._gyro_sum[2], self._gyro_sumsq[2], n) * 180.0 / math.pi

                        self.logger.info(
                            f"Gyro  bias=[{bgx:.6g}, {bgy:.6g}, {bgz:.6g}] rad/s,  std dev=[{g_sx:.2f}, {g_sy:.2f}, {g_sz:.2f}] deg/s"
                        )

                        if max(g_sx, g_sy, g_sz) > self.gyro_calib_max_std_dps:
                            self.logger.warn(
                                "Gyro calibration std dev is high — robot may have been moving during startup."
                            )

                        # Accel calibration calculations (we assume the robot is level during calibration, Z axis is up)
                        axm = self._accel_sum[0] / n
                        aym = self._accel_sum[1] / n
                        azm = self._accel_sum[2] / n
                        # Note: With ENU and “Z up”, at rest you typically want: az ≈ +9.80665 (not -9.8)

                        # Expect stationary accel: [0, 0, +G0]
                        bax = axm
                        bay = aym
                        baz = azm - G0  # Keep gravity. Set "imu0_remove_gravitational_acceleration:false" in robots/.../config/ekf_odom_params.yaml

                        self._accel_bias = [bax, bay, baz]

                        a_sx = std_dev_from_sums(self._accel_sum[0], self._accel_sumsq[0], n) # m/s^2
                        a_sy = std_dev_from_sums(self._accel_sum[1], self._accel_sumsq[1], n)
                        a_sz = std_dev_from_sums(self._accel_sum[2], self._accel_sumsq[2], n)

                        self.logger.info(
                            f"Accel bias=[{bax:.4f}, {bay:.4f}, {baz:.4f}] m/s^2, std=[{a_sx:.3f}, {a_sy:.3f}, {a_sz:.3f}] m/s^2"
                        )

                        if max(a_sx, a_sy, a_sz) > self.accel_calib_max_std_mps2:
                            self.logger.warn("Accel calibration std dev is high — robot may have been moving during startup.")

                        # Mag measurement average vector - heading at startup:
                        mxm = self._mag_sum[0] / n
                        mym = self._mag_sum[1] / n
                        mzm = self._mag_sum[2] / n

                        self._mag_avg_calib = [mxm, mym, mzm]  # cannot be used as bias

                        self.logger.info(
                            f"Mag    avg=[{mxm:.4f}, {mym:.4f}, {mzm:.4f}] micro Tesla, measurement average during calibration"
                        )

                        # reset calibration accumulators; we don’t re-run calibration currently:
                        self._gyro_sum = [0.0, 0.0, 0.0]
                        self._gyro_sumsq = [0.0, 0.0, 0.0]
                        self._accel_sum = [0.0, 0.0, 0.0]
                        self._accel_sumsq = [0.0, 0.0, 0.0]
                        self._mag_sum = [0.0, 0.0, 0.0]
                        self._calib_samples = 0

                # Always subtract biases once ready
                if self._calibration_done:
                    gx -= self._gyro_bias[0]
                    gy -= self._gyro_bias[1]
                    gz -= self._gyro_bias[2]

                    ax -= self._accel_bias[0]
                    ay -= self._accel_bias[1]
                    az -= self._accel_bias[2]
                    # That should yield linear_acceleration at rest: ax ≈ 0, ay ≈ 0, az ≈ +G0

                    # Do not subtract mag bias here as we don't have a real hard/soft iron calibration

                # Fill raw message (no orientation)
                imu_raw_msg.linear_acceleration.x = ax_raw
                imu_raw_msg.linear_acceleration.y = ay_raw
                imu_raw_msg.linear_acceleration.z = az_raw
                imu_raw_msg.angular_velocity.x = gx_raw
                imu_raw_msg.angular_velocity.y = gy_raw
                imu_raw_msg.angular_velocity.z = gz_raw
                imu_raw_msg.orientation_covariance[0] = -1.0

                # Fill mag message (convert to Teslas)
                mag_msg.magnetic_field.x = mx * 1e-6
                mag_msg.magnetic_field.y = my * 1e-6
                mag_msg.magnetic_field.z = mz * 1e-6

                # Convert temp raw -> Celsius, see datasheet pp.45,14
                temp_c = self.imu.tmpRaw / 333.87 + 21.0

                # Accumulate temp for averaging
                self._temp_sum_c += temp_c
                self._temp_count += 1
                publish_temp_now = (self._temp_count % self._temp_div) == 0

                if publish_temp_now:
                    avg_temp_c = self._temp_sum_c / float(self._temp_count)
                    temp_msg = sensor_msgs.msg.Temperature()
                    temp_msg.header.stamp = imu_raw_msg.header.stamp
                    temp_msg.header.frame_id = self.frame_id
                    temp_msg.temperature = round(avg_temp_c, 2)
                    temp_msg.variance = 0.0 # 0 means unknown
                    self.temp_pub.publish(temp_msg)
                    # Reset accumulator for next window
                    self._temp_sum_c = 0.0
                    self._temp_count = 0

            self.imu_raw_pub.publish(imu_raw_msg)
            self.mag_pub.publish(mag_msg)

        except Exception as e:
            # During shutdown, suppress noise; otherwise log
            if not self._shutting_down:
                self.logger.error(f"publish_cback exception: {e}")

    def destroy_node(self):
        self._shutting_down = True
        try:
            if hasattr(self, "pub_clk") and self.pub_clk is not None:
                self.pub_clk.cancel()
        except Exception:
            pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ICM20948Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl-C received, shutting down...")
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
