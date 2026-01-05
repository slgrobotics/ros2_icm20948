# -*- coding: utf-8 -*-
"""
    Copyright (c) 2015 Jonas Böer, jonas.boeer@student.kit.edu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#
# Original code:  see https://github.com/morgil/madgwick_py/blob/master/madgwickahrs.py
# ChatGPT.com review: https://chatgpt.com/s/t_695bf6b26e88819199e8094099819b7d
#                     https://chatgpt.com/s/t_695bf7a2e8248191b570af1b5464c396
#

import warnings
import numpy as np
from numpy.linalg import norm
from .quaternion import Quaternion


class MadgwickAHRS:
    samplePeriod = 1/256
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1  # A beta of 1 rad/s is huge for many consumer IMUs at 100–500 Hz unless your gyro noise is very high
    zeta = 0  # Gyro drift compensation

    def __init__(self, sampleperiod=1/256, quaternion=None, beta=0.05, zeta=0.0):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :param beta: Algorithm gain zeta
        :return:
        """
        self.samplePeriod = float(sampleperiod)
        self.quaternion = Quaternion(1,0,0,0) if quaternion is None else quaternion
        self.beta = float(beta)
        self.zeta = float(zeta)

    def setSamplePeriod(self, dt):
        self.samplePeriod = dt

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        accelerometer_raw = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalize accelerometer measurement & accel check
        a_norm = norm(accelerometer)
        if not np.isfinite(a_norm) or a_norm < 1e-12:
            warnings.warn("accelerometer is zero")
            # optionally: integrate gyro-only here
            return
        accelerometer /= a_norm

        # Normalize magnetometer measurement & mag check
        m_norm = norm(magnetometer)
        if not np.isfinite(m_norm) or m_norm < 1e-12:
            warnings.warn("magnetometer is zero; falling back to IMU")
            return self.update_imu(gyroscope, accelerometer_raw)  # or integrate with accel-correction
        magnetometer /= m_norm

        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])

        step = j.T.dot(f)
        step_norm = norm(step)
        if step_norm > 1e-12:
            step /= step_norm  # normalize step magnitude
        else:
            step[:] = 0.0  # or skip correction
        stepQuat = Quaternion(step[0], step[1], step[2], step[3])

        # Gyroscope compensation drift
        gyroscopeQuat = Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])

        gyroscopeQuat = gyroscopeQuat + (q.conj() * stepQuat) * 2 * self.samplePeriod * self.zeta * -1

        # Compute rate of change of quaternion
        qdot = (q * gyroscopeQuat) * 0.5 + (stepQuat * (-self.beta))

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalize quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalize accelerometer measurement & accel check
        a_norm = norm(accelerometer)
        if not np.isfinite(a_norm) or a_norm < 1e-12:
            warnings.warn("accelerometer is zero")
            # optionally: integrate gyro-only here
            return
        accelerometer /= a_norm

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step_norm = norm(step)
        if step_norm > 1e-12:
            step /= step_norm  # normalize step magnitude
        else:
            step[:] = 0.0  # or skip correction
        stepQuat = Quaternion(step[0], step[1], step[2], step[3])

        # Gyroscope compensation drift
        gyroscopeQuat = Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])

        gyroscopeQuat = gyroscopeQuat + (q.conj() * stepQuat) * 2 * self.samplePeriod * self.zeta * -1

        # Compute rate of change of quaternion
        qdot = (q * gyroscopeQuat) * 0.5 + (stepQuat * (-self.beta))

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalize quaternion

    def quaternion_xyzw(self):
        # Quaternion: self._q = np.array([w, x, y, z])
        # ROS uses x,y,z,w
        return (self.quaternion.q[1], self.quaternion.q[2], self.quaternion.q[3], self.quaternion.q[0])

    def initialize_from_accel_mag(self, ax, ay, az, mx=None, my=None, mz=None):
        # stubbed for now
        return

