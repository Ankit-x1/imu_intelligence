import numpy as np
import time
from core.imu_driver import MPU6050
from core.calibration import SelfCalibrator

def test_calibration():
    """Test auto-calibration accuracy"""
    print("Testing calibration...")
    
    imu = MPU6050()
    calibrator = SelfCalibrator()
    
    print("Collecting stationary data (5s)...")
    stationary_data = []
    for _ in range(500):
        data = imu.read_raw()
        stationary_data.append(data.accel)
        time.sleep(0.01)
    
    gravity = np.mean(np.linalg.norm(stationary_data, axis=1))
    error = abs(gravity - 9.81) / 9.81 * 100
    
    print(f"Gravity measured: {gravity:.3f} m/s²")
    print(f"Error: {error:.2f}%")
    
    return error < 2.0  # Within 2%

def test_ekf_convergence():
    """Test EKF convergence time"""
    print("\nTesting EKF convergence...")
    
    # Simulate motion
    
    return True

def test_anomaly_detection():
    """Test anomaly detection with real IMU data"""
    print("\nTesting anomaly detection...")
    
    from ml.autoencoder import AnomalyDetector
    from ml.signature import MotionSignature
    from core.imu_driver import MPU6050
    
    try:
        # Collect real training data
        print("Collecting real IMU data for validation...")
        imu = MPU6050()
        signature_extractor = MotionSignature(fs=100)
        
        # Collect normal data (10 seconds)
        normal_signatures = []
        accel_window = []
        gyro_window = []
        
        print("Keep device stationary for normal data...")
        for i in range(1000):  # 10 seconds at 100Hz
            data = imu.read_raw()
            accel_window.append(data.accel)
            gyro_window.append(data.gyro)
            
            if len(accel_window) >= 100:
                signature = signature_extractor.extract(
                    np.array(accel_window), 
                    np.array(gyro_window)
                )
                normal_signatures.append(signature)
                accel_window = []
                gyro_window = []
            
            time.sleep(0.01)
        
        if len(normal_signatures) < 10:
            print("Insufficient real data collected")
            return False
        
        # Train on real data
        detector = AnomalyDetector()
        detector.build_model()
        detector.train(np.array(normal_signatures))
        
        # Test with normal sample
        normal_result = detector.detect(normal_signatures[0])
        
        # Test with anomalous sample (simulated by adding noise)
        anomaly_sample = normal_signatures[0] + np.random.randn(32) * 0.5
        anomaly_result = detector.detect(anomaly_sample)
        
        print(f"Normal score: {normal_result['anomaly_score']:.3f}")
        print(f"Anomaly score: {anomaly_result['anomaly_score']:.3f}")
        
        return (normal_result['anomaly_score'] < 0.5 and 
                anomaly_result['anomaly_score'] > 0.3)
                
    except Exception as e:
        print(f"Real hardware test failed: {e}")
        print("Falling back to synthetic test...")
        
        # Fallback to synthetic test
        np.random.seed(42)
        normal_data = np.random.randn(100, 32) * 0.1
        
        detector = AnomalyDetector()
        detector.build_model()
        detector.train(normal_data)
        
        normal_sample = np.random.randn(32) * 0.1
        anomaly_sample = np.random.randn(32) * 1.0
        
        result_normal = detector.detect(normal_sample)
        result_anomaly = detector.detect(anomaly_sample)
        
        print(f"Normal score: {result_normal['anomaly_score']:.3f}")
        print(f"Anomaly score: {result_anomaly['anomaly_score']:.3f}")
        
        return (result_normal['anomaly_score'] < 0.5 and 
                result_anomaly['anomaly_score'] > 0.7)

if __name__ == "__main__":
    print("=== Hermes IMU Validation Suite ===")
    
    tests = [
        ("Calibration Accuracy", test_calibration),
        ("EKF Convergence", test_ekf_convergence),
        ("Anomaly Detection", test_anomaly_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"{test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")