"""
Integration test for IMU Intelligence
A test to verify the hasrdware setting.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from core.imu_driver import MPU6050

def test_hardware_connection():
    """Test if your MPU6050 is properly connected"""
    print("Testing MPU6050 Hardware Connection...")
    
    try:
        imu = MPU6050()
        print(" MPU6050 initialized successfully")
        
        print("Reading IMU data for 2 seconds...")
        data_points = []
        start_time = time.time()
        
        while time.time() - start_time < 2:
            data = imu.read_raw()
            data_points.append(data.accel)
            time.sleep(0.01)
        
        data_array = np.array(data_points)
        
        mean_accel = np.mean(data_array, axis=0)
        std_accel = np.std(data_array, axis=0)
        
        print(f"\n Results (device should be stationary):")
        print(f"Mean acceleration: {mean_accel} m/s²")
        print(f"Expected gravity: [0.00, 0.00, 9.81] m/s²")
        print(f"Standard deviation: {std_accel}")
        
        gravity_magnitude = np.linalg.norm(mean_accel)
        error_percent = abs(gravity_magnitude - 9.81) / 9.81 * 100
        
        if error_percent < 10:
            print(f" Gravity measurement: {gravity_magnitude:.2f} m/s² (Error: {error_percent:.1f}%)")
            return True
        else:
            print(f" Gravity measurement off: {gravity_magnitude:.2f} m/s²")
            return False
            
    except Exception as e:
        print(f" Hardware test failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check wiring (3.3V, GND, SDA, SCL)")
        print("2. Run: sudo i2cdetect -y 1")
        print("3. Make sure I2C is enabled: sudo raspi-config")
        return False

def visualize_raw_data():
    """Plot raw IMU data to verify it's working"""
    imu = MPU6050()
    samples = 200  
    
    accel_data = []
    gyro_data = []
    timestamps = []
    
    print(f"Collecting {samples} samples...")
    
    for i in range(samples):
        data = imu.read_raw()
        accel_data.append(data.accel)
        gyro_data.append(data.gyro)
        timestamps.append(data.timestamp)
        time.sleep(0.01)
    
    accel_array = np.array(accel_data)
    gyro_array = np.array(gyro_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(timestamps, accel_array[:, 0], label='Accel X')
    axes[0].plot(timestamps, accel_array[:, 1], label='Accel Y')
    axes[0].plot(timestamps, accel_array[:, 2], label='Accel Z')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('MPU6050 Raw Acceleration Data')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(timestamps, gyro_array[:, 0], label='Gyro X')
    axes[1].plot(timestamps, gyro_array[:, 1], label='Gyro Y')
    axes[1].plot(timestamps, gyro_array[:, 2], label='Gyro Z')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_title('MPU6050 Raw Gyroscope Data')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    try:
        plt.savefig('imu_raw_data.png')
        print(" Saved plot to 'imu_raw_data.png'")
        plt.show()
    except:
        plt.savefig('imu_raw_data.png')
        print(" Saved plot to 'imu_raw_data.png' (display not available)")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("HERMES IMU - HARDWARE INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Basic connection
    if test_hardware_connection():
        print("\n" + "=" * 60)
        print("HARDWARE TEST PASSED! ")
        print("=" * 60)
        
        # Test 2: Visualize data
        visualize = input("\nGenerate visualization plot? (y/n): ")
        if visualize.lower() == 'y':
            visualize_raw_data()
    else:
        print("\n" + "=" * 60)
        print("HARDWARE TEST FAILED ")
        print("Check wiring and I2C configuration")
        print("=" * 60)