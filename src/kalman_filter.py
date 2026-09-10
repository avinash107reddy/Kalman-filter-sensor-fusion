"""
1D Kalman filter for fusing IMU-like acceleration input with GPS-like
position measurements.

State vector: x = [position, velocity]
Motion model (predict step): driven by IMU acceleration input.
Measurement model (update step): driven by GPS position readings.
"""

import numpy as np


class KalmanFilter1D:
    def __init__(self, dt, x0, P0, sigma_a, sigma_gps):
        """
        Parameters
        ----------
        dt : float
            Timestep in seconds.
        x0 : array-like, shape (2,)
            Initial state guess [position, velocity].
        P0 : np.ndarray, shape (2, 2)
            Initial state covariance (uncertainty in x0).
        sigma_a : float
            Standard deviation of IMU acceleration noise (m/s^2).
            Used to build the process noise covariance Q.
        sigma_gps : float
            Standard deviation of GPS position noise (m).
            Used to build the measurement noise covariance R.
        """
        self.dt = dt

        # State transition matrix: constant-velocity model
        self.F = np.array([[1, dt],
                            [0, 1]])

        # Control input matrix: maps acceleration -> [position, velocity] change
        self.B = np.array([[0.5 * dt**2],
                            [dt]])

        # Measurement matrix: GPS observes position only
        self.H = np.array([[1, 0]])

        # Process noise covariance: acceleration uncertainty propagated
        # through B, same way real acceleration affects the state.
        self.Q = self.B @ self.B.T * sigma_a**2

        # Measurement noise covariance: GPS noise variance (scalar, as 1x1 matrix)
        self.R = np.array([[sigma_gps**2]])

        # State estimate and covariance (persistent, updated every call)
        self.x = np.array(x0, dtype=float).reshape(2, 1)
        self.P = np.array(P0, dtype=float)

    def predict(self, accel):
        """
        Predict step: propagate state and covariance forward using the
        IMU acceleration reading.

        Parameters
        ----------
        accel : float
            IMU acceleration reading (m/s^2) for this timestep.
        """
        self.x = self.F @ self.x + self.B * accel
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, gps_measurement):
        """
        Update step: correct the predicted state using a GPS position
        reading.

        Parameters
        ----------
        gps_measurement : float
            GPS position reading (m) for this timestep.
        """
        z = np.array([[gps_measurement]])

        # Innovation (residual): difference between measurement and prediction
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P