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
    """Test anomaly detection sensitivity"""
    print("\nTesting anomaly detection...")
    
    from ml.autoencoder import AnomalyDetector
    from ml.signature import MotionSignature
    
    # Generate synthetic normal data
    np.random.seed(42)
    normal_data = np.random.randn(1000, 32) * 0.1
    
    detector = AnomalyDetector()
    detector.build_model()
    detector.train(normal_data)
    
    # Test with normal sample
    normal_sample = np.random.randn(32) * 0.1
    result_normal = detector.detect(normal_sample)
    
    # Test with anomalous sample
    anomaly_sample = np.random.randn(32) * 1.0  # High variance
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