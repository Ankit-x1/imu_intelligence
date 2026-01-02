import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
import pywt

class MotionSignature:
    """
    Physics-based motion fingerprint that extracts
    meaningful features from IMU sensor data
    """
    def __init__(self, fs=100):
        self.fs = fs
        self.feature_names = []
        
    def extract(self, accel_window, gyro_window):
        """
        Extract 32-dimensional physics-based motion signature
        Features based on physical properties of motion
        """
        features = []
        
        # Time-domain physics features
        features.extend(self._physics_time_features(accel_window, 'accel'))
        features.extend(self._physics_time_features(gyro_window, 'gyro'))
        
        # Frequency-domain analysis (vibration characteristics)
        features.extend(self._physics_frequency_features(accel_window, 'accel'))
        
        # Statistical properties (motion consistency)
        features.extend(self._physics_statistical_features(accel_window))
        
        # Energy and power features (physical work)
        features.extend(self._physics_energy_features(accel_window, gyro_window))
        
        # Orientation and gravity features
        features.extend(self._physics_orientation_features(accel_window))
        
        return np.array(features)
    
    def _physics_time_features(self, data, sensor_type):
        """Extract time-domain physics features"""
        features = []
        
        features.append(np.sqrt(np.mean(data**2)))
        
        features.append(np.ptp(data))
        
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        features.append(peak / rms if rms > 0 else 0)
        
        features.append(np.sum(np.abs(data)) / len(data))
        
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features.append(zero_crossings / len(data))
        
        return features
    
    def _frequency_features(self, data, sensor_type):
        """Extract frequency-domain features"""
        freqs, psd = welch(data, fs=self.fs, nperseg=min(256, len(data)))
        
        features = []
        
        dominant_freq = freqs[np.argmax(psd)]
        features.append(dominant_freq)
        
        bands = [(0, 5), (5, 20), (20, 50), (50, 100)]
        total_power = np.sum(psd)
        
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[mask])
            features.append(band_power / total_power if total_power > 0 else 0)
        
        features.append(np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0)
        
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        features.append(geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0)
        
        return features
    
    def _statistical_features(self, data):
        """Statistical moments"""
        features = []
        
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            kurtosis = np.mean((data - mean)**4) / std**4
        else:
            kurtosis = 0
        features.append(kurtosis)
        
        if std > 0:
            skewness = np.mean((data - mean)**3) / std**3
        else:
            skewness = 0
        features.append(skewness)
        
        return features
    
    def _physics_energy_features(self, accel_window, gyro_window):
        """Energy and power features (physical work)"""
        features = []
        
        # Kinetic energy (1/2 * m * v^2)
        accel_magnitude = np.linalg.norm(accel_window, axis=1)
        gyro_magnitude = np.linalg.norm(gyro_window, axis=1)
        
        # Average energy over window
        features.append(np.mean(accel_magnitude**2))
        features.append(np.mean(gyro_magnitude**2))
        
        # Power spectral density (energy distribution)
        accel_fft = np.fft.rfft(accel_window, axis=0)
        gyro_fft = np.fft.rfft(gyro_window, axis=0)
        
        features.append(np.mean(np.abs(accel_fft)**2))
        features.append(np.mean(np.abs(gyro_fft)**2))
        
        # Mechanical work (force * displacement)
        if len(accel_window) > 1:
            accel_displacement = np.cumsum(accel_window, axis=0)
            work = np.sum(accel_window[:-1] * np.diff(accel_displacement, axis=0), axis=0)
            features.append(np.mean(work))
        
        return features
    
    def _physics_orientation_features(self, accel_window):
        """Orientation and gravity features"""
        features = []
        
        # Gravity vector estimation (when stationary)
        gravity_estimate = np.mean(accel_window, axis=0)
        features.extend(gravity_estimate)
        
        # Deviation from expected gravity (9.81 m/s²)
        gravity_magnitude = np.linalg.norm(gravity_estimate)
        features.append(abs(gravity_magnitude - 9.81))
        
        # Tilt angle from gravity components
        if gravity_magnitude > 0:
            tilt_x = np.arcsin(gravity_estimate[0] / gravity_magnitude)
            tilt_y = np.arcsin(gravity_estimate[1] / gravity_magnitude)
            features.extend([tilt_x, tilt_y])
        
        # Orientation change rate (stability)
        if len(accel_window) > 10:
            orientation_variance = np.var(accel_window[-10:], axis=0)
            features.append(orientation_variance)
        
        return features
    
    def _wavelet_features(self, signal):
        """Wavelet transform features"""
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        features = []
        
        for i, coeff in enumerate(coeffs):
            features.append(np.sum(coeff**2) / len(coeff))
            
        return features[:4]