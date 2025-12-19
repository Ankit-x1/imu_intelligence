import time
import numpy as np
import json
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
        
        training_data = []
        start_time = time.time()
        
        while time.time() - start_time < 30:
            data = self.imu.read_raw()
            
            # Simple window for signature            
            training_data.append(data.accel)
            
            if len(training_data) >= 100:
                window = np.array(training_data[-100:])
                signature = self.signature_extractor.extract(window, window*0.1)  # Mock gyro
                training_data = []  
                
                # Store signature
                if len(self.signature_history) < 100:  
                    self.signature_history.append(signature)
                else:
                    break
            
            time.sleep(0.01)
        
        if len(self.signature_history) > 50:
            X_train = np.array(self.signature_history)
            self.anomaly_detector.build_model()
            self.anomaly_detector.train(X_train)
            print(f"Anomaly detector trained on {len(X_train)} samples")
    
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
        
        with open('anomalies.jsonl', 'a') as f:
            f.write(json.dumps(anomaly_log) + '\n')
        
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