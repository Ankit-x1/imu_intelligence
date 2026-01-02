## Physics-Based IMU Intelligence System

## Overview
IMU intelligence is a **physics-based** system that analyzes real IMU sensor data using fundamental physical principles. The system extracts meaningful features from accelerometer and gyroscope measurements based on actual motion physics, not black-box patterns.

## Physics-Based Features
- **Kinetic Energy**: ½mv² calculations from acceleration and velocity
- **Power Spectral Density**: Frequency-domain energy distribution analysis  
- **Orientation Estimation**: Gravity vector and tilt angle calculations
- **Mechanical Work**: Force × displacement integration
- **Statistical Moments**: Kurtosis and skewness of motion distributions
- **Vibration Characteristics**: Band-specific frequency analysis

## Features 
- Self calibrate in <30 seconds (physics-based)
- Extended Kalman filter with physical state estimation
- 32-dimensional physics-based motion fingerprint
- Unsupervised anomaly detection using physical feature reconstruction
- Real time dashboard
- **ONNX export for edge deployment**
- **Docker-based workflow**

## Docker-Based Workflow

## Docker-Based Workflow

### Development Environment
```bash
# Build and test without hardware
docker-compose --profile train build
docker-compose --profile train run pytest
```

### Training Workflow (Laptop)
```bash
# 1. Train with simulated data
docker-compose --profile train run python training_protocol.py

# 2. Export ONNX model
# Automatically exports to models/anomaly_detector.onnx

# 3. Validate model
docker-compose --profile train run python tests/validation.py
```

### Edge Deployment (Raspberry Pi)
```bash
# 1. Build inference image
docker-compose --profile inference build

# 2. Deploy with real hardware
docker-compose --profile inference up -d

# 3. Monitor dashboard
docker-compose --profile dashboard up -d
# Access at http://raspberry-pi-ip:5000
```

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

### Method 1: Using uv (Recommended for Raspberry Pi)
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

### Method 2: Traditional pip
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence
python3 -m venv venv
source venv/bin/activate  # On Raspberry Pi
pip install -r requirements.txt

## Enable I2C on Raspberry Pi
sudo raspi-config
# Navigate to Interface Options -> I2C -> Enable
# Reboot after enabling

## Docker Testing & Deployment

### 1. Container Testing (No Hardware)
```bash
# Test basic functionality
docker-compose --profile train run python -c "
import numpy as np
from ml.autoencoder import AnomalyDetector
from ml.signature import MotionSignature

# Test with synthetic data
print('Testing synthetic data pipeline...')
detector = AnomalyDetector()
detector.build_model()
detector.train(np.random.randn(100, 32) * 0.1)
print('✅ Synthetic test passed')
"

# Run pytest suite
docker-compose --profile train run pytest tests/ -v
```

### 2. Hardware Integration (Raspberry Pi)
```bash
# Test IMU connection
docker-compose --profile inference run python test_integration.py

# Expected output (device stationary):
# Gravity measurement: ~9.81 m/s² (Error: <10%)
# Standard deviation: <0.5 m/s² on all axes
```

### 3. Production Deployment
```bash
# Training (laptop) -> ONNX export -> Edge deployment
# Step 1: Train and export
python training_protocol.py

# Step 2: Copy ONNX to Pi
scp models/anomaly_detector_*.onnx pi@raspberry-pi:~/imu_intelligence/models/

# Step 3: Deploy on Pi
docker-compose --profile inference up -d
```

## Running 
### Training Mode (Development)
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # uv method
# or
source venv/bin/activate   # pip method

# Professional training with real hardware
python training_protocol.py
```

### Inference Mode (Edge)
```bash
# On Raspberry Pi with hardware
docker-compose --profile inference up -d

# Dashboard (separate terminal)
docker-compose --profile dashboard up -d
# Access at http://raspberry-pi-ip:5000
```

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
5. **ONNX export** - Export optimized model for edge
6. **Validation** - Test with real hardware
7. **Visualization** - Generate plots of collected data
8. **Data saving** - Store training data and ONNX model

### Quick Training (Built-in)
```bash
python main.py
# System will automatically:
# 1. Calibrate (30 seconds stationary)
# 2. Collect training data (30 seconds normal motion)
# 3. Train anomaly detector
# 4. Export ONNX model
# 5. Start monitoring
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
# Deploy with Docker (recommended)
docker-compose --profile inference up -d

# Or run directly with ONNX
python main.py  # Automatically loads ONNX if available
```

## ONNX Edge Deployment

### Model Export
```python
# Automatic export during training
anomaly_detector.export_onnx("models/anomaly_detector.onnx")

# Manual export
from ml.autoencoder import AnomalyDetector
detector = AnomalyDetector()
detector.build_model()
detector.train(training_data)
detector.export_onnx("models/my_model.onnx")
```

### Edge Inference Benefits
- **Faster inference** - Optimized for CPU
- **Smaller footprint** - No PyTorch dependency
- **Better performance** - Tailored for edge devices
- **Cross-platform** - Works on any device with ONNX Runtime

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

### Docker Issues
```bash
# Check container logs
docker-compose logs -f

# Rebuild if needed
docker-compose build --no-cache

# Clean up
docker-compose down -v
docker system prune -f
```

## IMU6950 Compatibility Note
Current code supports MPU6050. For IMU6950:
1. Modify `core/imu_driver.py` with IMU6950 registers
2. Update I2C address (typically 0x69)
3. Adjust scaling factors for IMU6950 sensitivity
4. Test with `test_integration.py`

## Edge vs Development Modes

### Development (Training)
- **PyTorch model** - Full training capabilities
- **Real IMU data** - Hardware required
- **ONNX export** - Automatic model optimization
- **Validation suite** - Comprehensive testing

### Edge (Inference)
- **ONNX Runtime** - Optimized inference
- **Faster startup** - No training overhead
- **Lower memory** - Smaller footprint
- **Real-time monitoring** - Production ready
