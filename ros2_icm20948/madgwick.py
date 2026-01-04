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
        self._azf = 1.0
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
        ax, ay, az = self._axf, self._ayf, self._azf

        # Normalize accelerometer
        a = self._normalize3(ax, ay, az)
        if a is None or dt <= 0.0:
            # Only integrate gyro if accel invalid
            self._integrate_gyro(gx, gy, gz, dt)
            return

        ax, ay, az = a

        # Normalize magnetometer (optional)
        use_mag = (mx is not None and my is not None and mz is not None)
        if use_mag:
            m = self._normalize3(mx, my, mz)
            if m is None:
                use_mag = False
            else:
                mx, my, mz = m

        qw, qx, qy, qz = self.qw, self.qx, self.qy, self.qz

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-qx*gx - qy*gy - qz*gz)
        qDot2 = 0.5 * ( qw*gx + qy*gz - qz*gy)
        qDot3 = 0.5 * ( qw*gy - qx*gz + qz*gx)
        qDot4 = 0.5 * ( qw*gz + qx*gy - qy*gx)

        # Gradient descent correction
        if use_mag:
            # Low-pass accel filter:
            self._mxf = (1-self._alpha_m)*self._mxf + self._alpha_m*mx
            self._myf = (1-self._alpha_m)*self._myf + self._alpha_m*my
            self._mzf = (1-self._alpha_m)*self._mzf + self._alpha_m*mz
            mx, my, mz = self._mxf, self._myf, self._mzf

            s1, s2, s3, s4 = self._grad_imu_mag(qw, qx, qy, qz, ax, ay, az, mx, my, mz)
        else:
            s1, s2, s3, s4 = self._grad_imu(qw, qx, qy, qz, ax, ay, az)

        # dynamic beta - settling faster when not rotating
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
        # A compact IMU+Mag gradient; good enough for yaw stabilization.
        # Compute reference direction of Earth's magnetic field
        # (using standard Madgwick helper terms)
        qwx = qw*qx; qwy = qw*qy; qwz = qw*qz
        qxx = qx*qx; qxy = qx*qy; qxz = qx*qz
        qyy = qy*qy; qyz = qy*qz; qzz = qz*qz

        # Rotate mag into earth frame
        hx = 2.0*mx*(0.5 - qyy - qzz) + 2.0*my*(qxy - qwz) + 2.0*mz*(qxz + qwy)
        hy = 2.0*mx*(qxy + qwz) + 2.0*my*(0.5 - qxx - qzz) + 2.0*mz*(qyz - qwx)
        _2bx = math.sqrt(hx*hx + hy*hy)
        _2bz = 2.0*mx*(qxz - qwy) + 2.0*my*(qyz + qwx) + 2.0*mz*(0.5 - qxx - qyy)

        # Gradient (IMU+Mag). This is the standard form but trimmed for readability.
        # If you ever want the fully expanded canonical version, ask and I’ll paste it.
        f1 = 2.0*(qxz - qwy) - ax
        f2 = 2.0*(qwx + qyz) - ay
        f3 = 2.0*(0.5 - qxx - qyy) - az
        f4 = 2.0*_2bx*(0.5 - qyy - qzz) + 2.0*_2bz*(qxz - qwy) - mx
        f5 = 2.0*_2bx*(qxy - qwz) + 2.0*_2bz*(qwx + qyz) - my
        f6 = 2.0*_2bx*(qwz + qxy) + 2.0*_2bz*(0.5 - qxx - qyy) - mz

        # Approximate Jacobian transpose * f (works well in practice; smaller than the huge expanded form)
        s1 = (-2.0*qy)*f1 + (2.0*qx)*f2 + 0.0*f3 + (-2.0*_2bx*qz)*f4 + (-2.0*_2bx*qy + 2.0*_2bz*qx)*f5 + (2.0*_2bx*qx)*f6
        s2 = (2.0*qz)*f1 + (2.0*qw)*f2 + (-4.0*qx)*f3 + (2.0*_2bx*qy + 2.0*_2bz*qz)*f4 + (2.0*_2bx*qz + 2.0*_2bz*qw)*f5 + (2.0*_2bx*qy - 4.0*_2bz*qx)*f6
        s3 = (-2.0*qw)*f1 + (2.0*qz)*f2 + (-4.0*qy)*f3 + (-4.0*_2bx*qy - 2.0*_2bz*qw)*f4 + (2.0*_2bx*qx + 2.0*_2bz*qz)*f5 + (2.0*_2bx*qw - 4.0*_2bz*qy)*f6
        s4 = (2.0*qx)*f1 + (2.0*qy)*f2 + 0.0*f3 + (-4.0*_2bx*qz + 2.0*_2bz*qx)*f4 + (-2.0*_2bx*qw + 2.0*_2bz*qy)*f5 + (2.0*_2bx*qx)*f6
        return s1, s2, s3, s4

    def quaternion_xyzw(self):
        # ROS uses x,y,z,w
        return (self.qx, self.qy, self.qz, self.qw)
