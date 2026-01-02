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

## Smart Docker-Based Workflow

### **Laptop (Training) → Raspberry Pi (Inference)**

## Step 1: Laptop Training Setup

### uv (Recommended - Fastest)**
```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and setup
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Step 2: Training on Laptop

### **Hardware Connection Required**
```bash
# Connect MPU6050 to laptop via USB-I2C adapter
# Or use Raspberry Pi for training, then deploy model

# Test hardware connection
python test_integration.py
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

## Step 3: Raspberry Pi Deployment

### **Setup Raspberry Pi**
```bash
# On Raspberry Pi
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence

# Install uv (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Setup environment
uv venv
source .venv/bin/activate
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
# Copy trained model from laptop
scp models/anomaly_detector_*.onnx pi@raspberry-pi:~/imu_intelligence/models/

# Run inference
source .venv/bin/activate
python main.py
```

## File Integration Analysis

### **System Architecture**
```
📁 Core Components
├── core/imu_driver.py      # MPU6050 hardware interface
├── core/kalman_filter.py   # Physics-based state estimation
├── core/calibration.py     # Self-calibration using gravity
├── ml/signature.py         # Physics-based feature extraction
├── ml/autoencoder.py       # Anomaly detection + ONNX export
├── main.py                 # Inference orchestrator
├── training_protocol.py    # Professional training workflow
├── test_integration.py     # Hardware validation
└── dashboard/web_ui.py      # Real-time monitoring
```

### **Integration Flow**
1. **IMU Driver** → Raw sensor data
2. **Calibration** → Physics-based bias correction
3. **Kalman Filter** → Orientation/position estimation
4. **Signature Extractor** → Physics features (energy, vibration)
5. **Autoencoder** → Anomaly detection + ONNX export
6. **Dashboard** → Real-time monitoring

## Smart Training Options

### **Option 1: Laptop Training (Recommended)**
- **Hardware**: Connect MPU6050 via USB-I2C adapter
- **Advantage**: Powerful laptop for training
- **Process**: Train → Export ONNX → Deploy to Pi

### **Option 2: Raspberry Pi Training**
- **Hardware**: Connect directly to Pi GPIO
- **Advantage**: No adapter needed
- **Process**: Train on Pi → Keep model locally

### **Option 3: Simulation Training**
- **Hardware**: No IMU needed (synthetic data)
- **Advantage**: Test workflow without hardware
- **Process**: Use synthetic physics data → Test deployment

## Testing & Validation

### **Hardware Testing**
```bash
# Test IMU connection
python test_integration.py

# Expected output (stationary):
# Gravity measurement: ~9.81 m/s² (Error: <10%)
# Standard deviation: <0.5 m/s²
```

### **Physics Validation**
```bash
# Test physics-based features
python tests/validation.py

# Test with real hardware
python training_protocol.py --validate-only
```

### **Docker Testing**
```bash
# Test without hardware
docker-compose --profile train run pytest

# Test with hardware
docker-compose --profile inference run python test_integration.py
```

## Production Deployment

### **Monitoring Dashboard**
```bash
# Start dashboard
docker-compose --profile dashboard up -d

# Features:
- Real-time physics data
- Anomaly timeline
- System performance
- Motion visualization
```

### **Edge Benefits**
- **Real-time Processing**: 100Hz physics calculations
- **Offline Operation**: No cloud dependency
- **Privacy**: Local data processing
- **Reliability**: Self-calibrating, adaptive

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

## Quick Summary

**Smart Workflow:**
1. **Laptop**: `uv venv` → `python training_protocol.py` → ONNX export
2. **Raspberry Pi**: `uv venv` → `docker-compose --profile inference up -d`
3. **Monitor**: `docker-compose --profile dashboard up -d`

**Key Advantages:**
- **Physics-based**: Explainable, robust features
- **Edge-optimized**: Real-time, offline operation
- **Docker-smart**: Clean deployment profiles
- **uv-fast**: Quick dependency management
