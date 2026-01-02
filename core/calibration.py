import numpy as np
from collections import deque
from scipy.optimize import minimize

class SelfCalibrator:
    """
    Physics-based calibration that learns sensor biases
    and scale factors using real IMU measurements
    """
    def __init__(self, window_size=1000):
        self.window = deque(maxlen=window_size)
        self.is_calibrated = False
        self.calibration_data = {
            'accel_bias': np.zeros(3),
            'gyro_bias': np.zeros(3),
            'accel_scale': np.ones(3),
            'gyro_scale': np.ones(3),
            'gravity_magnitude': 9.81,
            'stationary_threshold': 0.05
        }
        
    def add_sample(self, accel, gyro):
        """Add new IMU sample to calibration window"""
        self.window.append((accel.copy(), gyro.copy()))
        
        if len(self.window) > 100:
            self._detect_stationary()
            
        if len(self.window) == self.window.maxlen:
            self._auto_calibrate()
    
    def _detect_stationary(self):
        """Detect when device is stationary using variance"""
        recent_accel = np.array([a for a, _ in list(self.window)[-50:]])
        recent_gyro = np.array([g for _, g in list(self.window)[-50:]])
        
        accel_var = np.var(recent_accel, axis=0)
        gyro_var = np.var(recent_gyro, axis=0)
        
        if (np.mean(accel_var) < self.calibration_data['stationary_threshold'] and 
            np.mean(gyro_var) < 0.001):
            
            self.stationary_accel = recent_accel.mean(axis=0)
            self.stationary_gyro = recent_gyro.mean(axis=0)
            
            self.calibration_data['gravity_magnitude'] = np.linalg.norm(self.stationary_accel)
    
    def _auto_calibrate(self):
        """Perform full 6-point calibration automatically"""
        print("Auto-calibration in progress...")
        
        all_accel = np.array([a for a, _ in self.window])
        all_gyro = np.array([g for _, g in self.window])
        
        self.calibration_data['gyro_bias'] = self.stationary_gyro
        
        def cost_function(params):
            bias = params[:3]
            scale = params[3:6]
            
            corrected = (all_accel - bias) * scale
            norms = np.linalg.norm(corrected, axis=1)
            
            return np.mean((norms - self.calibration_data['gravity_magnitude'])**2)
        
        initial_guess = np.concatenate([np.zeros(3), np.ones(3)])
        
        try:
            result = minimize(cost_function, initial_guess, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8})
            
            if result.success and np.all(np.isfinite(result.x)):
                self.calibration_data['accel_bias'] = result.x[:3]
                self.calibration_data['accel_scale'] = result.x[3:6]
                self.is_calibrated = True
                print("Auto-calibration complete!")
            else:
                print("Warning: Calibration optimization failed, using defaults")
                self.calibration_data['accel_bias'] = np.zeros(3)
                self.calibration_data['accel_scale'] = np.ones(3)
                
        except Exception as e:
            print(f"Calibration error: {e}, using defaults")
            self.calibration_data['accel_bias'] = np.zeros(3)
            self.calibration_data['accel_scale'] = np.ones(3)
            
    def get_calibration(self):
        return self.calibration_data