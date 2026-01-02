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

## Raspberry Pi Training & Deployment

### **Setup Raspberry Pi**
```bash
# Install uv (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and setup
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Enable I2C
sudo raspi-config
# Interface Options → I2C → Enable
sudo reboot
```

### **Hardware Setup**
```
MPU6050 -> Raspberry Pi
VCC    -> 3.3V
GND    -> GND  
SDA    -> GPIO 2 (SDA)
SCL    -> GPIO 3 (SCL)
```

## Step 1: Training on Raspberry Pi

### **Hardware Validation**
```bash
# Test IMU connection
python test_integration.py

# Expected output (device stationary):
# Gravity measurement: ~9.81 m/s² (Error: <10%)
# Standard deviation: <0.5 m/s²
```

### **Professional Training**
```bash
# Run physics-based training
python training_protocol.py
```

**Training Process:**
1. **Hardware Validation** - Tests MPU6050 connection
2. **Physics Calibration** - 30 seconds stationary (gravity estimation)
3. **Normal Data Collection** - 30 seconds typical motion
4. **Physics Feature Extraction** - Energy, orientation, vibration analysis
5. **Model Training** - Autoencoder learns physical patterns
6. **ONNX Export** - Optimized model for edge deployment

**Training Output:**
- `models/anomaly_detector_*.onnx` - Optimized edge model
- `training_data_*.json` - Physics-based training data
- Motion analysis plots

## Step 2: Deployment on Raspberry Pi

### **Deploy with Docker (Recommended)**
```bash
# Build inference image
docker-compose --profile inference build

# Deploy with real hardware
docker-compose --profile inference up -d

# Start dashboard
docker-compose --profile dashboard up -d
# Access at http://raspberry-pi-ip:5000
```

### **Or Direct Deployment**
```bash
# Run inference directly
source .venv/bin/activate
python main.py
```

**Deployment Process:**
1. **Model Loading** - Automatically loads ONNX if available
2. **Real-time Processing** - Physics-based feature extraction at 100Hz
3. **Anomaly Detection** - Identifies physical motion violations
4. **Continuous Monitoring** - 24/7 operation with logging

### **Dashboard Monitoring**
```bash
# Start dashboard
docker-compose --profile dashboard up -d
# Access at http://localhost:5000
```

**Dashboard Features:**
- Real-time physics data
- Anomaly timeline
- System performance
- Motion visualization


### **Integration Flow**
1. **IMU Driver** → Raw sensor data
2. **Calibration** → Physics-based bias correction
3. **Kalman Filter** → Orientation/position estimation
4. **Signature Extractor** → Physics features (energy, vibration)
5. **Autoencoder** → Anomaly detection + ONNX export
6. **Dashboard** → Real-time monitoring

## Quick Start Summary

```bash
# === RASPBERRY PI SETUP ===
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# === TRAINING ===
python training_protocol.py

# === DEPLOYMENT ===
docker-compose --profile inference up -d
docker-compose --profile dashboard up -d
```

## Troubleshooting

### **Hardware Issues**
```bash
# Check I2C connection
sudo i2cdetect -y 1
# Should show device at 0x68

# Test IMU
python test_integration.py
```

### **Training Issues**
- Ensure device is stationary during calibration
- Verify 30 seconds of normal motion data
- Check ONNX export completion

### **Deployment Issues**
```bash
# Check Docker logs
docker-compose logs -f

# Rebuild if needed
docker-compose build --no-cache
```

## Key Advantages

- **Physics-based**: Explainable, robust features
- **Edge-optimized**: Real-time, offline operation
- **Docker-smart**: Clean deployment profiles
- **uv-fast**: Quick dependency management
- **All-in-one**: Train and deploy on same device
