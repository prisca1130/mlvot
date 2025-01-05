import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])  # Control input (acceleration)

        # State matrix [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Control input model
        self.B = np.array([
            [(dt**2) / 2, 0],
            [0, (dt**2) / 2],
            [dt, 0],
            [0, dt]
        ])

        # Measurement mapping matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(4) * std_acc**2

        # Measurement noise covariance
        self.R = np.diag([x_sdt_meas**2, y_sdt_meas**2])

        # Prediction error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict the state and covariance
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2]  # Return position (x, y)

    def update(self, z):
        # Measurement residual
        y = z - (self.H @ self.x)
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update state
        self.x = self.x + K @ y
        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
