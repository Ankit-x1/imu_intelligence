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
        
        # Time-domain physics features (5 + 5 = 10)
        features.extend(self._physics_time_features(accel_window, 'accel'))
        features.extend(self._physics_time_features(gyro_window, 'gyro'))
        
        # Frequency-domain analysis - only accel (7 features)
        features.extend(self._frequency_features_single_axis(accel_window))
        
        # Statistical properties - only accel magnitude (2 features)
        accel_mag = np.linalg.norm(accel_window, axis=1)
        features.extend(self._compute_statistical_features(accel_mag))
        
        # Energy and power features (5 features)
        features.extend(self._physics_energy_features(accel_window, gyro_window))
        
        # Orientation and gravity features (8 features)
        features.extend(self._physics_orientation_features_compact(accel_window))
        
        # Ensure exactly 32 features
        features = features[:32]
        
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
    
    def _frequency_features_single_axis(self, data):
        """Extract frequency-domain features from magnitude only"""
        # Use magnitude of acceleration for frequency analysis
        accel_mag = np.linalg.norm(data, axis=1)
        return self._compute_frequency_features(accel_mag)
    
    def _compute_frequency_features(self, data):
        """Compute frequency features for single axis"""
        # Ensure minimum data length
        if len(data) < 10:
            return [0.0] * 7  # Return zeros for insufficient data
        
        freqs, psd = welch(data, fs=self.fs, nperseg=min(256, len(data)//2))
        
        features = []
        
        # Handle empty PSD
        if len(psd) == 0 or np.sum(psd) == 0:
            return [0.0] * 7
        
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
        
        # Handle multi-dimensional data
        if len(data.shape) > 1:
            # Process each axis separately
            for axis in range(data.shape[1]):
                axis_data = data[:, axis]
                axis_features = self._compute_statistical_features(axis_data)
                features.extend(axis_features)
            return features
        else:
            # Single axis data
            return self._compute_statistical_features(data)
    
    def _compute_statistical_features(self, data):
        """Compute statistical features for single axis"""
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
    
    def _physics_orientation_features_compact(self, accel_window):
        """Compact orientation features (8 total)"""
        features = []
        
        # Gravity vector estimation (3 features)
        gravity_estimate = np.mean(accel_window, axis=0)
        features.extend(gravity_estimate)
        
        # Deviation from expected gravity (1 feature)
        gravity_magnitude = np.linalg.norm(gravity_estimate)
        features.append(abs(gravity_magnitude - 9.81))
        
        # Tilt angles (2 features)
        if gravity_magnitude > 0:
            tilt_x = np.arcsin(np.clip(gravity_estimate[0] / gravity_magnitude, -1, 1))
            tilt_y = np.arcsin(np.clip(gravity_estimate[1] / gravity_magnitude, -1, 1))
            features.extend([tilt_x, tilt_y])
        else:
            features.extend([0.0, 0.0])
        
        # Overall stability (2 features)
        if len(accel_window) > 10:
            orientation_variance = np.var(accel_window[-10:], axis=0)
            features.extend([np.mean(orientation_variance), np.max(orientation_variance)])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _wavelet_features(self, signal):
        """Wavelet transform features"""
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        features = []
        
        for i, coeff in enumerate(coeffs):
            features.append(np.sum(coeff**2) / len(coeff))
            
        return features[:4]