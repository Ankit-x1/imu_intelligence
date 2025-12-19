import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
import pywt

class MotionSignature:
    """
    Universal motion fingerprint that
    combines time, frequency, and wavelet features
    """
    def __init__(self, fs=100):
        self.fs = fs
        self.feature_names = []
        
    def extract(self, accel_window, gyro_window):
        """
        Extract 32-dimensional motion signature
        """
        features = []
        
        features.extend(self._time_features(accel_window, 'accel'))
        features.extend(self._time_features(gyro_window, 'gyro'))
        
        features.extend(self._frequency_features(accel_window, 'accel'))
        
        features.extend(self._statistical_features(accel_window))
        
        features.extend(self._wavelet_features(accel_window[:, 0]))  # X-axis
        
        return np.array(features)
    
    def _time_features(self, data, sensor_type):
        """Extract time-domain features"""
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
    
    def _wavelet_features(self, signal):
        """Wavelet transform features"""
        coeffs = pywt.wavedec(signal, 'db4', level=3)
        features = []
        
        for i, coeff in enumerate(coeffs):
            features.append(np.sum(coeff**2) / len(coeff))
            
        return features[:4]  