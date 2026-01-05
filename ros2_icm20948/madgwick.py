import math

class MadgwickAHRS:
    """
    Minimal Madgwick AHRS (IMU+Mag).
    - gyro in rad/s
    - accel in m/s^2 (will be normalized internally)
    - mag in Tesla (will be normalized internally)
    Quaternion is ENU body->world if your axes follow REP-103 (x fwd, y left, z up)
    """
    def __init__(self, beta=0.01):
        self.beta = beta

        # initial quaternion (w, x, y, z)
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

        # Low-pass accel filter:
        self._axf = self._ayf = 0.0
        self._azf = 9.81
        self._alpha_a = 0.2  # 0..1

        # Low-pass mag filter:
        self._mxf = self._myf = self._mzf = 0.0
        self._alpha_m = 0.2  # 0..1

    def _normalize3(self, x, y, z):
        n = math.sqrt(x*x + y*y + z*z)
        if n < 1e-12:
            return None
        return (x/n, y/n, z/n)

    def update(self, gx, gy, gz, ax, ay, az, mx=None, my=None, mz=None, dt=0.01):

        # Low-pass accel filter:
        self._axf = (1-self._alpha_a)*self._axf + self._alpha_a*ax
        self._ayf = (1-self._alpha_a)*self._ayf + self._alpha_a*ay
        self._azf = (1-self._alpha_a)*self._azf + self._alpha_a*az

        # Normalize accelerometer quaternion
        a = self._normalize3(self._axf, self._ayf, self._azf)
        if a is None or dt <= 0.0:
            # Only integrate gyro if accel invalid
            self._integrate_gyro(gx, gy, gz, dt)
            return

        ax, ay, az = a

        use_mag = (mx is not None and my is not None and mz is not None)

        if use_mag:
            # Low-pass mag filter (on good raw mag values):
            self._mxf = (1-self._alpha_m)*self._mxf + self._alpha_m*mx
            self._myf = (1-self._alpha_m)*self._myf + self._alpha_m*my
            self._mzf = (1-self._alpha_m)*self._mzf + self._alpha_m*mz

            # Normalize magnetometer quaternion:
            m = self._normalize3(self._mxf, self._myf, self._mzf)
            if m is None:
                use_mag = False  # oops...
            else:
                mx, my, mz = m
                # simple disturbance gate: if mag vector is nearly vertical, yaw is ill-conditioned
                if abs(mz) > 0.9:
                    use_mag = False

        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz  # initial quaternion (w, x, y, z)

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-qx*gx - qy*gy - qz*gz)
        qDot2 = 0.5 * ( qw*gx + qy*gz - qz*gy)
        qDot3 = 0.5 * ( qw*gy - qx*gz + qz*gx)
        qDot4 = 0.5 * ( qw*gz + qx*gy - qy*gx)

        # Gradient descent correction
        if use_mag:
            s1, s2, s3, s4 = self._grad_imu_mag(qw, qx, qy, qz, ax, ay, az, mx, my, mz)
        else:
            s1, s2, s3, s4 = self._grad_imu(qw, qx, qy, qz, ax, ay, az)

        # dynamic beta - reduce mag influence when stationary
        omega = math.sqrt(gx*gx + gy*gy + gz*gz)  # rad/s
        # Example thresholds (tune): 0.02 rad/s ≈ 1.15 deg/s
        if omega < 0.02:
            beta = self.beta * 0.2  # settling faster
        else:
            beta = self.beta

        # Normalize step magnitude
        s_norm = math.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
        if s_norm > 1e-12:
            s1, s2, s3, s4 = s1/s_norm, s2/s_norm, s3/s_norm, s4/s_norm
            qDot1 -= beta * s1
            qDot2 -= beta * s2
            qDot3 -= beta * s3
            qDot4 -= beta * s4

        # Integrate to yield quaternion
        qw += qDot1 * dt
        qx += qDot2 * dt
        qy += qDot3 * dt
        qz += qDot4 * dt

        # Normalize quaternion
        q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if q_norm > 1e-12:
            self.qw, self.qx, self.qy, self.qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm

    def _integrate_gyro(self, gx, gy, gz, dt):
        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz
        qDot1 = 0.5 * (-qx*gx - qy*gy - qz*gz)
        qDot2 = 0.5 * ( qw*gx + qy*gz - qz*gy)
        qDot3 = 0.5 * ( qw*gy - qx*gz + qz*gx)
        qDot4 = 0.5 * ( qw*gz + qx*gy - qy*gx)
        qw += qDot1 * dt
        qx += qDot2 * dt
        qy += qDot3 * dt
        qz += qDot4 * dt
        q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if q_norm > 1e-12:
            self.qw, self.qx, self.qy, self.qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm

    def _grad_imu(self, qw, qx, qy, qz, ax, ay, az):
        # Objective function and Jacobian for IMU-only
        _2qw = 2.0 * qw
        _2qx = 2.0 * qx
        _2qy = 2.0 * qy
        _2qz = 2.0 * qz
        _4qw = 4.0 * qw
        _4qx = 4.0 * qx
        _4qy = 4.0 * qy
        _8qx = 8.0 * qx
        _8qy = 8.0 * qy
        qwqw = qw * qw
        qxqx = qx * qx
        qyqy = qy * qy
        qzqz = qz * qz

        s1 = _4qw * qyqy + _2qy * ax + _4qw * qxqx - _2qx * ay
        s2 = _4qx * qzqz - _2qz * ax + 4.0 * qwqw * qx - _2qw * ay - _4qx + _8qx * qxqx + _8qx * qyqy + _4qx * az
        s3 = 4.0 * qwqw * qy + _2qw * ax + _4qy * qzqz - _2qz * ay - _4qy + _8qy * qxqx + _8qy * qyqy + _4qy * az
        s4 = 4.0 * qxqx * qz - _2qx * ax + 4.0 * qyqy * qz - _2qy * ay
        return s1, s2, s3, s4

    def _grad_imu_mag(self, qw, qx, qy, qz, ax, ay, az, mx, my, mz):
        """
        Canonical Madgwick 9-DOF gradient descent step (IMU + Mag).
        Inputs must already be normalized:
        accel (ax,ay,az) unit vector
        mag   (mx,my,mz) unit vector
        Returns gradient step (s0,s1,s2,s3) for quaternion (qw,qx,qy,qz).
        """

        # Rename to Madgwick paper / common reference notation
        q0, q1, q2, q3 = qw, qx, qy, qz

        # Precompute repeated quaternion products
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q3 = q2 * q3

        # Common factors
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3

        _2q0mx = 2.0 * q0 * mx
        _2q0my = 2.0 * q0 * my
        _2q0mz = 2.0 * q0 * mz
        _2q1mx = 2.0 * q1 * mx
        _2q1my = 2.0 * q1 * my
        _2q1mz = 2.0 * q1 * mz
        _2q2mx = 2.0 * q2 * mx
        _2q2my = 2.0 * q2 * my
        _2q2mz = 2.0 * q2 * mz
        _2q3mx = 2.0 * q3 * mx
        _2q3my = 2.0 * q3 * my
        _2q3mz = 2.0 * q3 * mz

        # Reference direction of Earth's magnetic field (hx, hy, then 2bx and 2bz)
        hx = (
            mx * q0q0
            - _2q0my * q3
            + _2q0mz * q2
            + mx * q1q1
            + _2q1my * q2
            + _2q1mz * q3
            - mx * q2q2
            - mx * q3q3
        )

        hy = (
            _2q0mx * q3
            + my * q0q0
            - _2q0mz * q1
            + _2q1mx * q2
            - my * q1q1
            + my * q2q2
            + _2q2mz * q3
            - my * q3q3
        )

        _2bx = math.sqrt(hx * hx + hy * hy)
        _2bz = (
            -_2q0mx * q2
            + _2q0my * q1
            + mz * q0q0
            + _2q1mx * q3
            - mz * q1q1
            + _2q2my * q3
            - mz * q2q2
            + mz * q3q3
        )

        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        # Objective function elements (f1..f6)
        f1 = 2.0 * (q1q3 - q0q2) - ax
        f2 = 2.0 * (q0q1 + q2q3) - ay
        f3 = 2.0 * (0.5 - q1q1 - q2q2) - az

        f4 = _2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx
        f5 = _2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my
        f6 = _2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz

        # Gradient (Jacobian^T * f), canonical Madgwick form
        s0 = (-_2q2 * f1) + (_2q1 * f2) + (-_2bz * q2 * f4) + ((-_2bx * q3 + _2bz * q1) * f5) + (_2bx * q2 * f6)
        s1 = (_2q3 * f1) + (_2q0 * f2) + (-4.0 * q1 * f3) + (_2bz * q3 * f4) + ((_2bx * q2 + _2bz * q0) * f5) + ((_2bx * q3 - _4bz * q1) * f6)
        s2 = (-_2q0 * f1) + (_2q3 * f2) + (-4.0 * q2 * f3) + ((-_4bx * q2 - _2bz * q0) * f4) + ((_2bx * q1 + _2bz * q3) * f5) + ((_2bx * q0 - _4bz * q2) * f6)
        s3 = (_2q1 * f1) + (_2q2 * f2) + ((-_4bx * q3 + _2bz * q1) * f4) + ((-_2bx * q0 + _2bz * q2) * f5) + (_2bx * q1 * f6)

        return s0, s1, s2, s3

    def quaternion_xyzw(self):
        # ROS uses x,y,z,w
        return (self.qx, self.qy, self.qz, self.qw)

    def initialize_from_accel_mag(self, ax, ay, az, mx=None, my=None, mz=None):
        # expects SI accel (m/s^2) and mag (any units), will normalize internally
        a = self._normalize3(ax, ay, az)
        if a is None:
            return False
        ax, ay, az = a

        # roll/pitch from accel, ENU with z-up (gravity points +z when stationary in your convention)
        # For a typical IMU reporting +Z up at rest: ax~0,ay~0,az~+1
        roll  = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

        yaw = 0.0
        use_mag = (mx is not None and my is not None and mz is not None)
        if use_mag:
            m = self._normalize3(mx, my, mz)
            if m is not None:
                mx, my, mz = m
                # tilt-compensate mag
                cr = math.cos(roll);  sr = math.sin(roll)
                cp = math.cos(pitch); sp = math.sin(pitch)

                # rotate mag into level frame
                mx2 = mx*cp + mz*sp
                my2 = mx*sr*sp + my*cr - mz*sr*cp

                # ENU: yaw=atan2(East, North) depends on your axis conventions.
                # Common choice: yaw = atan2(mx_level, my_level) or atan2(my_level, mx_level)
                # If x=East,y=North: heading (yaw) = atan2(E, N) = atan2(mx2, my2)
                yaw = math.atan2(mx2, my2)
                # yaw = math.atan2(-mx2, my2)  # ? REP-103 body axes (x forward, y left, z up)

        self._set_quaternion_from_rpy(roll, pitch, yaw)
        return True

    def _set_quaternion_from_rpy(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5);   sy = math.sin(yaw * 0.5)

        # yaw (z), pitch (y), roll (x) intrinsic, standard
        qw = cy*cp*cr + sy*sp*sr
        qx = cy*cp*sr - sy*sp*cr
        qy = sy*cp*sr + cy*sp*cr
        qz = sy*cp*cr - cy*sp*sr

        n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        self.qw, self.qx, self.qy, self.qz = qw/n, qx/n, qy/n, qz/n
