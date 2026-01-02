import numpy as np
from scipy.linalg import expm

class AdaptiveEKF:
    """
    Physics-based Extended Kalman Filter that
    estimates orientation and position using IMU measurements
    """
    def __init__(self):
        # State: [qw, qx, qy, qz, px, py, pz, vx, vy, vz, gx_bias, gy_bias, gz_bias]
        self.state = np.zeros(13)
        self.state[0] = 1.0  # Quaternion w component
        
        # Covariance matrix (uncertainty in estimates)
        self.P = np.eye(13) * 0.1
        
        # Process noise (motion model uncertainty)
        self.Q = np.eye(13) * 0.001  
        
        # Measurement noise (sensor uncertainty)
        self.R = np.eye(6) * 0.01    
        
        # Time step (100Hz sampling)
        self.dt = 0.01  
        
    def predict(self, gyro, accel):
        """Prediction step with gyro and accelerometer inputs"""
        q = self.state[0:4]
        
        gyro_corrected = gyro - self.state[10:13]
        
        Omega = np.array([
            [0, -gyro_corrected[0], -gyro_corrected[1], -gyro_corrected[2]],
            [gyro_corrected[0], 0, gyro_corrected[2], -gyro_corrected[1]],
            [gyro_corrected[1], -gyro_corrected[2], 0, gyro_corrected[0]],
            [gyro_corrected[2], gyro_corrected[1], -gyro_corrected[0], 0]
        ])
        
        q_dot = 0.5 * Omega @ q
        self.state[0:4] += q_dot * self.dt
        self.state[0:4] /= np.linalg.norm(self.state[0:4])  
        
        R = self._quat_to_rotmat(q)
        accel_world = R @ accel - np.array([0, 0, 9.81])
        
        self.state[7:10] += accel_world * self.dt  
        self.state[4:7] += self.state[7:10] * self.dt  
        
        F = self._compute_jacobian(gyro_corrected, accel_world)
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, accel_meas):
        """Update using accelerometer as inclinometer"""
        q = self.state[0:4]
        R = self._quat_to_rotmat(q)
        
        g_world = np.array([0, 0, 9.81])
        h = R.T @ g_world
        
        y = accel_meas - h
        
        H = self._measurement_jacobian()
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state += K @ y
        self.state[0:4] /= np.linalg.norm(self.state[0:4])  
        
        self.P = (np.eye(13) - K @ H) @ self.P
        
        self._adapt_noise(y, S)
        
    def _adapt_noise(self, innovation, innovation_cov):
        """Online noise adaptation"""
        alpha = 0.01  
        N = innovation @ innovation.T
        
        if N > 2 * np.trace(innovation_cov):
            self.Q *= (1 + alpha)
        else:
            self.Q *= (1 - alpha/10)
        
        self.Q = np.maximum(self.Q, np.eye(13) * 1e-6)  
        
    def _compute_jacobian(self, gyro, accel_world):
        """Compute state transition Jacobian"""
        q = self.state[0:4]
        
        # Simplified Jacobian for quaternion dynamics
        F = np.eye(13)
        
        # Quaternion state derivatives
        F[0:4, 0:4] = np.eye(4) + 0.5 * self.dt * np.array([
            [0, -gyro[0], -gyro[1], -gyro[2]],
            [gyro[0], 0, gyro[2], -gyro[1]],
            [gyro[1], -gyro[2], 0, gyro[0]],
            [gyro[2], gyro[1], -gyro[0], 0]
        ])
        
        # Position/velocity integration
        R = self._quat_to_rotmat(q)
        F[4:7, 7:10] = np.eye(3) * self.dt
        F[7:10, 7:10] = np.eye(3)
        
        return F
    
    def _measurement_jacobian(self):
        """Measurement Jacobian for accelerometer"""
        H = np.zeros((3, 13))
        q = self.state[0:4]
        
        # Partial derivatives of gravity w.r.t quaternion
        H[0, 0:4] = [2*q[1], 2*q[2], 2*q[3], 0]  
        H[1, 0:4] = [0, 2*q[0], 0, 2*q[3]]
        H[2, 0:4] = [2*q[0], 0, 2*q[1], 2*q[2]]
        
        return H
    
    def _quat_to_rotmat(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        # Normalize quaternion to prevent numerical instability
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-10:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])