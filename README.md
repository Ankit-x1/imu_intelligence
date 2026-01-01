# Autonomous Self calibrating IMU Edge Intelligence Module

## Overview
IMU intelligence is a universal IMU module that automatically learns any machine dynamics and detect anomalies without labeled data, and outputs a standarized motion fingerprint on a Edge Device.

## Features 
- Self calibrate in <30 seconds
- Extended Kalman filrer with online noise adaptation
- 32-dimensional fingerprint of machine behaviour
- Unsupervised anomaly detection, no labeled data needed 
- Real time dashboard

## Hardware Compatibility

### Supported IMU
- **MPU6050** (Default) - I2C address 0x68
- **IMU6950** - Requires driver modification (see notes below)

### Raspberry Pi Setup
```
MPU6050 -> Raspberry Pi
VCC    -> 3.3V
GND    -> GND  
SDA    -> GPIO 2 (SDA)
SCL    -> GPIO 3 (SCL)
```

## Installation

# Method 1: Using uv (Recommended for Raspberry Pi)
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and setup
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Raspberry Pi
uv pip install -r requirements.txt

# Method 2: Traditional pip
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence
python3 -m venv venv
source venv/bin/activate  # On Raspberry Pi
pip install -r requirements.txt

## Enable I2C on Raspberry Pi
sudo raspi-config
# Navigate to Interface Options -> I2C -> Enable
# Reboot after enabling

## Hardware Testing
# Test IMU connection and calibration
python test_integration.py

# Run validation suite (now uses real hardware)
python tests/validation.py

# Professional training protocol (recommended)
python training_protocol.py

## Running 
# Make sure virtual environment is activated
source .venv/bin/activate  # uv method
# or
source venv/bin/activate   # pip method

python main.py

# Dasboard (separate terminal)
python dashboard/web_ui.py
# Access at http://raspberry-pi-ip:5000

## Real-World Training Protocol

### Professional Data Collection (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run professional training protocol
python training_protocol.py
```

This will guide you through:
1. **Hardware validation** - Verify IMU is working
2. **Normal data collection** - 30 seconds of typical motion
3. **Anomaly data collection** - Optional 15 seconds of unusual motion
4. **Training** - Build anomaly detector from real data
5. **Validation** - Test with real hardware
6. **Visualization** - Generate plots of collected data
7. **Data saving** - Store training data for future use

### Quick Training (Built-in)
```bash
python main.py
# System will automatically:
# 1. Calibrate (30 seconds stationary)
# 2. Collect training data (30 seconds normal motion)
# 3. Train anomaly detector
# 4. Start monitoring
```

## Real-World Testing Guide

### 1. Hardware Validation
```bash
# Test basic connectivity
python test_integration.py

# Expected output (device stationary):
# Gravity measurement: ~9.81 m/s² (Error: <10%)
# Standard deviation: <0.5 m/s² on all axes
```

### 2. Calibration Testing
- Keep device completely stationary during first 30 seconds
- Verify gravity reading is close to 9.81 m/s²
- Check calibration bias values in logs

### 3. Motion Learning Phase
- Move device in normal operating patterns for 30 seconds
- System will learn normal motion signatures
- Training progress shown in console

### 4. Anomaly Detection Testing
- Normal motion: anomaly score < 0.5
- Unusual motion: anomaly score > 0.85
- Anomalies logged to `anomalies.jsonl`

### 5. Dashboard Monitoring
- Real-time orientation data
- Motion signature visualization  
- Anomaly history and scores
- System performance metrics

### 6. Production Deployment
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run with Docker (recommended)
docker-compose up -d

# Or run directly with virtual environment
python main.py > imu.log 2>&1 &
```

## Troubleshooting

### I2C Issues
```bash
# Check I2C devices
sudo i2cdetect -y 1
# Should show device at 0x68

# Check I2C speed
sudo raspi-config
# Advanced Options -> I2C -> 100kHz (default)
```

### Performance Issues
- Reduce sampling rate in `config/settings.yaml`
- Use Docker for resource isolation
- Monitor CPU usage with `htop`

### Calibration Problems
- Ensure device is perfectly level during startup
- Check for mechanical vibrations
- Verify stable power supply

## IMU6950 Compatibility Note
Current code supports MPU6050. For IMU6950:
1. Modify `core/imu_driver.py` with IMU6950 registers
2. Update I2C address (typically 0x69)
3. Adjust scaling factors for IMU6950 sensitivity
4. Test with `test_integration.py`
