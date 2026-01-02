import time
import numpy as np
import json
import os
from datetime import datetime
from core.imu_driver import MPU6050
from core.kalman_filter import AdaptiveEKF
from core.calibration import SelfCalibrator
from ml.signature import MotionSignature
from ml.autoencoder import AnomalyDetector

class HermesIMU:
    """
    Main orchestrator
    """
    def __init__(self):
        mode = os.getenv('MODE', 'inference')
        
        self.imu = MPU6050()
        self.ekf = AdaptiveEKF()
        self.calibrator = SelfCalibrator()
        self.signature_extractor = MotionSignature(fs=100)
        self.anomaly_detector = AnomalyDetector()
        
        self.is_initialized = False
        self.motion_history = []
        self.signature_history = []
        
        self.stats = {
            'samples_processed': 0,
            'calibration_updates': 0,
            'anomalies_detected': 0,
            'avg_processing_time': 0
        }
        
        # Load ONNX model for inference mode
        if mode == 'inference':
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pretrained ONNX model for inference"""
        model_path = "models/anomaly_detector.onnx"
        if os.path.exists(model_path):
            if self.anomaly_detector.load_onnx(model_path):
                print(" ONNX model loaded for edge inference")
            else:
                print("  ONNX load failed, using PyTorch")
        else:
            print("  No pretrained model found, using PyTorch")
            print("Run training first: python training_protocol.py")
        
    def run(self):
        """Main loop - runs at 100Hz"""
        print("IMU System Starting...")
        
        # Warm-up period
        print("Calibrating... (keep device stationary)")
        warmup_samples = []
        for _ in range(500):  
            data = self.imu.read_raw()
            warmup_samples.append((data.accel, data.gyro))
            time.sleep(0.01)
        
        # Initial calibration
        accel_array = np.array([a for a, _ in warmup_samples])
        gyro_array = np.array([g for _, g in warmup_samples])
        self.calibrator.window.extend(list(zip(accel_array, gyro_array)))
        self.calibrator._auto_calibrate()
        
        # Apply calibration
        calib = self.calibrator.get_calibration()
        self.imu.set_calibration(
            calib['accel_bias'],
            calib['gyro_bias'],
            calib['accel_scale'],
            calib['gyro_scale']
        )
        
        print("Learning normal motion patterns...")
        self._collect_training_data()
        
        print("Agent active - monitoring motion...")
        
        window_size = 100  
        accel_window = []
        gyro_window = []
        
        while True:
            start_time = time.time()
            
            data = self.imu.read_raw()
            
            self.ekf.predict(data.gyro, data.accel)
            self.ekf.update(data.accel)
            
            accel_window.append(data.accel)
            gyro_window.append(data.gyro)
            
            if len(accel_window) >= window_size:
                signature = self.signature_extractor.extract(
                    np.array(accel_window[-window_size:]),
                    np.array(gyro_window[-window_size:])
                )
                
                anomaly_result = self.anomaly_detector.detect(signature)
                
                if anomaly_result['is_anomaly']:
                    self._log_anomaly(signature, anomaly_result)
                
                self.signature_history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'signature': signature.tolist(),
                    'anomaly_score': anomaly_result['anomaly_score'],
                    'orientation': self.ekf.state[0:4].tolist(),
                    'velocity': self.ekf.state[7:10].tolist()
                })
                
                if len(self.signature_history) > 1000:
                    self.signature_history = self.signature_history[-1000:]
            
            self.stats['samples_processed'] += 1
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                0.99 * self.stats['avg_processing_time'] + 
                0.01 * processing_time
            )
            
            time.sleep(max(0, 0.01 - processing_time))
    
    def _collect_training_data(self):
        """Collect normal operation data for training"""
        print("Move device in normal operating patterns for 30 seconds...")
        print("Show the device its typical motions and usage patterns")
        
        accel_windows = []
        gyro_windows = []
        
        start_time = time.time()
        window_size = 100
        current_accel = []
        current_gyro = []
        
        while time.time() - start_time < 30:
            data = self.imu.read_raw()
            
            current_accel.append(data.accel)
            current_gyro.append(data.gyro)
            
            if len(current_accel) >= window_size:
                # Store complete window with real accel and gyro data
                accel_windows.append(np.array(current_accel))
                gyro_windows.append(np.array(current_gyro))
                
                # Extract signature from real IMU data
                signature = self.signature_extractor.extract(
                    np.array(current_accel), 
                    np.array(current_gyro)
                )
                
                self.signature_history.append(signature)
                
                # Reset for next window
                current_accel = []
                current_gyro = []
                
                print(f"Collected {len(self.signature_history)} training samples...")
                
                if len(self.signature_history) >= 100:
                    print("Training data collection complete!")
                    break
            
            time.sleep(0.01)
        
        if len(self.signature_history) > 50:
            X_train = np.array(self.signature_history)
            self.anomaly_detector.build_model()
            self.anomaly_detector.train(X_train)
            print(f"Anomaly detector trained on {len(X_train)} real motion samples")
        else:
            print("Warning: Insufficient training data collected")
    
    def _log_anomaly(self, signature, anomaly_result):
        """Log anomaly with full context"""
        anomaly_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'signature': signature.tolist(),
            'anomaly_score': anomaly_result['anomaly_score'],
            'reconstruction_error': anomaly_result['reconstruction_error'],
            'orientation': self.ekf.state[0:4].tolist(),
            'velocity': self.ekf.state[7:10].tolist(),
            'system_state': self.stats.copy()
        }
        
        try:
            with open('anomalies.jsonl', 'a') as f:
                f.write(json.dumps(anomaly_log) + '\n')
        except Exception as e:
            print(f"Failed to log anomaly: {e}")
        
        self.stats['anomalies_detected'] += 1
        print(f" ANOMALY DETECTED! Score: {anomaly_result['anomaly_score']:.3f}")
    
    def get_status(self):
        """Get current system status"""
        return {
            'orientation': self.ekf.state[0:4].tolist(),
            'position': self.ekf.state[4:7].tolist(),
            'velocity': self.ekf.state[7:10].tolist(),
            'calibrated': self.calibrator.is_calibrated,
            'stats': self.stats,
            'latest_signature': self.signature_history[-1] if self.signature_history else None
        }

if __name__ == "__main__":
    hermes = HermesIMU()
    hermes.run()