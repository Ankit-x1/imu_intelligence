#!/usr/bin/env python3
"""
Professional IMU Training Protocol
Real hardware data collection and validation
"""

import time
import numpy as np
import json
from datetime import datetime
from core.imu_driver import MPU6050
from ml.signature import MotionSignature
from ml.autoencoder import AnomalyDetector
import matplotlib.pyplot as plt

class IMUTrainer:
    """Professional IMU training system for real hardware"""
    
    def __init__(self):
        self.imu = MPU6050()
        self.signature_extractor = MotionSignature(fs=100)
        self.anomaly_detector = AnomalyDetector()
        
    def hardware_check(self):
        """Verify IMU is working properly"""
        print("=== HARDWARE VALIDATION ===")
        print("Testing IMU connection...")
        
        try:
            # Test reading
            samples = []
            for i in range(100):
                data = self.imu.read_raw()
                samples.append(data)
                time.sleep(0.01)
            
            # Check gravity reading
            accel_data = np.array([s.accel for s in samples])
            gravity_mag = np.linalg.norm(np.mean(accel_data, axis=0))
            error = abs(gravity_mag - 9.81) / 9.81 * 100
            
            print(f"Gravity measured: {gravity_mag:.2f} m/s²")
            print(f"Error: {error:.1f}%")
            
            if error < 10:
                print("✅ Hardware validation PASSED")
                return True
            else:
                print("❌ Hardware validation FAILED")
                return False
                
        except Exception as e:
            print(f"❌ Hardware error: {e}")
            return False
    
    def collect_training_data(self, duration=30, motion_type="normal"):
        """Collect real IMU training data"""
        print(f"\n=== COLLECTING {motion_type.upper()} DATA ===")
        print(f"Duration: {duration} seconds")
        print(f"Start moving the device in {motion_type} patterns NOW!")
        
        accel_windows = []
        gyro_windows = []
        signatures = []
        
        window_size = 100
        current_accel = []
        current_gyro = []
        
        start_time = time.time()
        samples_collected = 0
        
        while time.time() - start_time < duration:
            data = self.imu.read_raw()
            
            current_accel.append(data.accel)
            current_gyro.append(data.gyro)
            
            if len(current_accel) >= window_size:
                # Complete window collected
                accel_array = np.array(current_accel)
                gyro_array = np.array(current_gyro)
                
                # Extract signature
                signature = self.signature_extractor.extract(accel_array, gyro_array)
                
                accel_windows.append(accel_array)
                gyro_windows.append(gyro_array)
                signatures.append(signature)
                
                samples_collected += 1
                print(f"Samples collected: {samples_collected}")
                
                # Reset for next window
                current_accel = []
                current_gyro = []
            
            time.sleep(0.01)
        
        print(f"✅ Collected {len(signatures)} {motion_type} samples")
        return accel_windows, gyro_windows, signatures
    
    def train_anomaly_detector(self, normal_signatures):
        """Train anomaly detector on real normal data"""
        print("\n=== TRAINING ANOMALY DETECTOR ===")
        
        if len(normal_signatures) < 50:
            print("❌ Insufficient training data")
            return False
        
        X_train = np.array(normal_signatures)
        print(f"Training on {len(X_train)} real motion samples...")
        
        self.anomaly_detector.build_model()
        self.anomaly_detector.train(X_train)
        
        print("✅ Training complete")
        return True
    
    def validate_training(self, normal_signatures, anomaly_signatures):
        """Validate with real test data"""
        print("\n=== VALIDATION ===")
        
        # Test normal samples
        normal_scores = []
        for signature in normal_signatures[:10]:  # Test subset
            result = self.anomaly_detector.detect(signature)
            normal_scores.append(result['anomaly_score'])
        
        # Test anomaly samples
        anomaly_scores = []
        for signature in anomaly_signatures:
            result = self.anomaly_detector.detect(signature)
            anomaly_scores.append(result['anomaly_score'])
        
        print(f"Normal samples - Avg score: {np.mean(normal_scores):.3f}")
        print(f"Anomaly samples - Avg score: {np.mean(anomaly_scores):.3f}")
        
        # Check separation
        if np.mean(normal_scores) < 0.5 and np.mean(anomaly_scores) > 0.7:
            print("✅ Validation PASSED")
            return True
        else:
            print("❌ Validation FAILED")
            return False
    
    def save_training_data(self, normal_signatures, anomaly_signatures=None):
        """Save training data and export ONNX model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        training_data = {
            'timestamp': timestamp,
            'normal_signatures': [sig.tolist() for sig in normal_signatures],
            'anomaly_signatures': [sig.tolist() for sig in anomaly_signatures] if anomaly_signatures else [],
            'model_config': {
                'input_dim': 32,
                'latent_dim': 8
            }
        }
        
        filename = f"training_data_{timestamp}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2)
            print(f"✅ Training data saved to {filename}")
            
            # Export to ONNX for edge deployment
            self.anomaly_detector.export_onnx(f"models/anomaly_detector_{timestamp}.onnx")
            return filename
        except Exception as e:
            print(f"Failed to save training data: {e}")
            return None
    
    def visualize_data(self, accel_windows, gyro_windows, title="IMU Data"):
        """Visualize collected IMU data"""
        if not accel_windows or not gyro_windows:
            return
        
        # Combine all windows
        all_accel = np.vstack(accel_windows)
        all_gyro = np.vstack(gyro_windows)
        
        time_axis = np.arange(len(all_accel)) / 100.0  # 100Hz sampling
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Accelerometer data
        axes[0].plot(time_axis, all_accel[:, 0], label='Accel X', alpha=0.7)
        axes[0].plot(time_axis, all_accel[:, 1], label='Accel Y', alpha=0.7)
        axes[0].plot(time_axis, all_accel[:, 2], label='Accel Z', alpha=0.7)
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].set_title(f'{title} - Accelerometer')
        axes[0].legend()
        axes[0].grid(True)
        
        # Gyroscope data
        axes[1].plot(time_axis, all_gyro[:, 0], label='Gyro X', alpha=0.7)
        axes[1].plot(time_axis, all_gyro[:, 1], label='Gyro Y', alpha=0.7)
        axes[1].plot(time_axis, all_gyro[:, 2], label='Gyro Z', alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].set_title(f'{title} - Gyroscope')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        filename = f"{title.lower().replace(' ', '_')}.png"
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to {filename}")
        except Exception as e:
            print(f"Failed to save visualization: {e}")
        
        # Only show plot if display is available
        try:
            plt.show()
        except:
            print("Display not available, plot saved to file only")

def main():
    """Professional training workflow"""
    trainer = IMUTrainer()
    
    print("🚀 PROFESSIONAL IMU TRAINING PROTOCOL")
    print("=" * 50)
    
    # Step 1: Hardware validation
    if not trainer.hardware_check():
        print("❌ Hardware failed. Check connections and try again.")
        return
    
    # Step 2: Collect normal operation data
    print("\n" + "="*50)
    print("STEP 1: NORMAL OPERATION DATA")
    print("Move the device as it would normally be used.")
    print("Examples: normal walking, typical machine operation, etc.")
    
    input("Press Enter to start collecting normal data...")
    accel_normal, gyro_normal, sig_normal = trainer.collect_training_data(
        duration=30, motion_type="normal"
    )
    
    # Step 3: Collect anomaly data (optional but recommended)
    print("\n" + "="*50)
    print("STEP 2: ANOMALY DATA (Optional)")
    print("Move the device in unusual ways.")
    print("Examples: drops, impacts, unusual vibrations, etc.")
    
    collect_anomaly = input("Collect anomaly data? (y/n): ").lower() == 'y'
    accel_anomaly = gyro_anomaly = sig_anomaly = []
    
    if collect_anomaly:
        input("Press Enter to start collecting anomaly data...")
        accel_anomaly, gyro_anomaly, sig_anomaly = trainer.collect_training_data(
            duration=15, motion_type="anomaly"
        )
    
    # Step 4: Train anomaly detector
    print("\n" + "="*50)
    print("STEP 3: TRAINING")
    
    if trainer.train_anomaly_detector(sig_normal):
        # Step 5: Validation
        print("\n" + "="*50)
        print("STEP 4: VALIDATION")
        
        if sig_anomaly:
            trainer.validate_training(sig_normal, sig_anomaly)
        else:
            print("⚠️  No anomaly data for validation")
        
        # Step 6: Save data
        print("\n" + "="*50)
        print("STEP 5: SAVING DATA")
        trainer.save_training_data(sig_normal, sig_anomaly)
        
        # Step 7: Visualization
        print("\n" + "="*50)
        print("STEP 6: VISUALIZATION")
        trainer.visualize_data(accel_normal, gyro_normal, "Normal Operation")
        
        if sig_anomaly:
            trainer.visualize_data(accel_anomaly, gyro_anomaly, "Anomaly Patterns")
        
        print("\n🎉 TRAINING COMPLETE!")
        print("You can now run the main system: python main.py")
        
    else:
        print("❌ Training failed. Check data collection.")

if __name__ == "__main__":
    main()
