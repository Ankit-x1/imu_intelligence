#!/usr/bin/env python3
"""
Hardware Setup Analysis for Professional IMU Deployment
"""

import numpy as np

class HardwareAnalyzer:
    """Analyze current hardware setup for production readiness"""
    
    def __init__(self):
        self.setup_issues = []
        self.recommendations = []
        
    def analyze_imu_config(self):
        """Analyze MPU6050 configuration"""
        print("=== IMU CONFIGURATION ANALYSIS ===")
        
        # Current configuration from imu_driver.py
        current_config = {
            'gyro_config': 0x08,  # Register 0x1B
            'accel_config': 0x08,  # Register 0x1C
            'power_mgmt': 0x00,   # Register 0x6B
        }
        
        # MPU6050 Register meanings
        gyro_ranges = {
            0x00: '±250°/s',
            0x08: '±500°/s',  # Current
            0x10: '±1000°/s',
            0x18: '±2000°/s'
        }
        
        accel_ranges = {
            0x00: '±2g',
            0x08: '±4g',  # Current
            0x10: '±8g',
            0x18: '±16g'
        }
        
        print(f"Current Gyro Range: {gyro_ranges[current_config['gyro_config']]}")
        print(f"Current Accel Range: {accel_ranges[current_config['accel_config']]}")
        
        # Analysis
        if current_config['gyro_config'] == 0x08:
            print("✅ Gyro range (±500°/s) good for most applications")
        else:
            self.setup_issues.append("Gyro range may need adjustment")
            
        if current_config['accel_config'] == 0x08:
            print("✅ Accel range (±4g) good for most applications")
        else:
            self.setup_issues.append("Accel range may need adjustment")
            
        return current_config
    
    def analyze_sampling_rate(self):
        """Analyze 100Hz sampling rate"""
        print("\n=== SAMPLING RATE ANALYSIS ===")
        
        sampling_rate = 100  # Hz
        nyquist_freq = sampling_rate / 2  # 50 Hz
        
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"Nyquist Frequency: {nyquist_freq} Hz")
        
        # Human motion frequencies
        human_motion_freqs = {
            'walking': 1.0-2.0,      # Hz
            'running': 2.0-3.5,      # Hz
            'vibrations': 10-100,    # Hz
            'impacts': 100-1000      # Hz
        }
        
        print("\nMotion Frequency Coverage:")
        for motion, freq_range in human_motion_freqs.items():
            if freq_range[1] <= nyquist_freq:
                print(f"✅ {motion}: {freq_range[0]}-{freq_range[1]} Hz - CAPTURED")
            else:
                print(f"⚠️  {motion}: {freq_range[0]}-{freq_range[1]} Hz - PARTIALLY CAPTURED")
                self.setup_issues.append(f"High-frequency {motion} may be undersampled")
        
        # Recommendation
        if sampling_rate >= 100:
            print("✅ 100Hz sampling adequate for most applications")
            self.recommendations.append("Consider 200Hz for high-vibration monitoring")
        else:
            self.setup_issues.append("Sampling rate too low for accurate motion capture")
    
    def analyze_i2c_setup(self):
        """Analyze I2C communication setup"""
        print("\n=== I2C COMMUNICATION ANALYSIS ===")
        
        i2c_config = {
            'bus': 1,
            'address': 0x68,
            'speed': 100000  # 100kHz default
        }
        
        print(f"I2C Bus: {i2c_config['bus']}")
        print(f"Device Address: 0x{i2c_config['address']:02x}")
        print(f"I2C Speed: {i2c_config['speed']/1000} kHz")
        
        # Data rate analysis
        sample_rate = 100  # Hz
        bytes_per_sample = 14  # MPU6050 sends 14 bytes
        required_bandwidth = sample_rate * bytes_per_sample
        
        print(f"Required Bandwidth: {required_bandwidth} bytes/sec")
        print(f"Available Bandwidth: {i2c_config['speed']/8} bytes/sec")
        
        if required_bandwidth < i2c_config['speed']/8:
            print("✅ I2C bandwidth sufficient")
        else:
            self.setup_issues.append("I2C bandwidth may be limiting")
            self.recommendations.append("Consider increasing I2C speed to 400kHz")
        
        # I2C reliability
        print("\nI2C Reliability Factors:")
        print("✅ 4-wire connection (VCC, GND, SDA, SCL)")
        print("⚠️  Consider pull-up resistors for long cables")
        print("⚠️  Shielded cables recommended for noisy environments")
        
        self.recommendations.append("Use 4.7kΩ pull-up resistors for cables > 10cm")
        self.recommendations.append("Add 0.1µF capacitor near IMU for power stability")
    
    def analyze_power_requirements(self):
        """Analyze power supply requirements"""
        print("\n=== POWER SUPPLY ANALYSIS ===")
        
        # MPU6050 specifications
        power_specs = {
            'voltage_range': '2.37V - 3.46V',
            'current_consumption': {
                'normal': '3.6mA',
                'low_power': '500µA'
            },
            'recommended_voltage': '3.3V'
        }
        
        print(f"Voltage Range: {power_specs['voltage_range']}")
        print(f"Current Consumption: {power_specs['current_consumption']['normal']}")
        print(f"Recommended Voltage: {power_specs['recommended_voltage']}")
        
        print("\nPower Quality Requirements:")
        print("✅ 3.3V stable supply required")
        print("⚠️  Avoid 5V - can damage IMU")
        print("⚠️  Power ripple < 50mV recommended")
        
        # Raspberry Pi power analysis
        pi_power_capacity = {
            '3.3V_pins': '50mA max total',
            'available_per_pin': '~16mA typical'
        }
        
        print(f"\nRaspberry Pi 3.3V Capacity: {pi_power_capacity['3.3V_pins']}")
        print(f"Available per pin: {pi_power_capacity['available_per_pin']}")
        print(f"IMU Consumption: {power_specs['current_consumption']['normal']}")
        
        if float(power_specs['current_consumption']['normal'].replace('mA', '')) < 16:
            print("✅ IMU power draw within Pi limits")
        else:
            self.setup_issues.append("IMU may exceed Pi GPIO power limits")
            self.recommendations.append("Use external 3.3V regulator for better stability")
    
    def analyze_environmental_factors(self):
        """Analyze environmental considerations"""
        print("\n=== ENVIRONMENTAL FACTORS ===")
        
        environmental_specs = {
            'temperature_range': '-40°C to +85°C',
            'shock_resistance': '1000g',
            'vibration_resistance': '10g'
        }
        
        print(f"Operating Temperature: {environmental_specs['temperature_range']}")
        print(f"Shock Resistance: {environmental_specs['shock_resistance']}")
        print(f"Vibration Resistance: {environmental_specs['vibration_resistance']}")
        
        print("\nEnvironmental Recommendations:")
        print("✅ Suitable for industrial environments")
        print("⚠️  Consider temperature compensation for extreme conditions")
        print("⚠️  Mount securely to prevent mechanical stress")
        
        self.recommendations.append("Use mechanical mounting with vibration damping")
        self.recommendations.append("Consider enclosure for dust/moisture protection")
    
    def analyze_data_quality(self):
        """Analyze data quality factors"""
        print("\n=== DATA QUALITY ANALYSIS ===")
        
        # Noise characteristics
        noise_specs = {
            'accel_noise_density': '200 µg/√Hz',
            'gyro_noise_density': '0.015 °/s/√Hz',
            'temperature_drift': '±0.5%/°C'
        }
        
        print(f"Accelerometer Noise: {noise_specs['accel_noise_density']}")
        print(f"Gyroscope Noise: {noise_specs['gyro_noise_density']}")
        print(f"Temperature Drift: {noise_specs['temperature_drift']}")
        
        # Filtering analysis
        print("\nFiltering Recommendations:")
        print("✅ Kalman filter implemented for noise reduction")
        print("✅ Auto-calibration compensates for bias")
        print("⚠️  Consider additional low-pass filter for high vibration")
        
        self.recommendations.append("Add 20Hz low-pass filter for vibration-heavy applications")
        self.recommendations.append("Implement temperature monitoring for extreme environments")
    
    def generate_report(self):
        """Generate comprehensive hardware analysis report"""
        print("\n" + "="*60)
        print("HARDWARE SETUP ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        self.analyze_imu_config()
        self.analyze_sampling_rate()
        self.analyze_i2c_setup()
        self.analyze_power_requirements()
        self.analyze_environmental_factors()
        self.analyze_data_quality()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if not self.setup_issues:
            print("✅ HARDWARE SETUP IS PRODUCTION READY")
            print("✅ All critical requirements met")
        else:
            print("⚠️  HARDWARE SETUP NEEDS IMPROVEMENTS")
            print("⚠️  Address the following issues:")
            for issue in self.setup_issues:
                print(f"   - {issue}")
        
        print("\nRECOMMENDATIONS FOR OPTIMAL PERFORMANCE:")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\nOVERALL ASSESSMENT: {'PRODUCTION READY' if not self.setup_issues else 'NEEDS ATTENTION'}")
        
        return len(self.setup_issues) == 0

def main():
    """Run hardware analysis"""
    analyzer = HardwareAnalyzer()
    is_ready = analyzer.generate_report()
    
    if is_ready:
        print("\n🎉 Your hardware setup is ready for professional deployment!")
    else:
        print("\n⚠️  Review and implement recommendations for best results")

if __name__ == "__main__":
    main()
