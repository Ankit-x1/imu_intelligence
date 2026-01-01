# IMU Connection Guide

## MPU6050 to Raspberry Pi Connection

### Physical Wiring
```
MPU6050 Pin    -> Raspberry Pi GPIO
VCC (3.3V)     -> Pin 1 (3.3V)
GND            -> Pin 6 (GND)
SDA            -> Pin 3 (GPIO 2/SDA)
SCL            -> Pin 5 (GPIO 3/SCL)
```

### Pinout Diagram
```
Raspberry Pi GPIO Header:
     3V3  (1) (2)  5V
   GPIO2  (3) (4)  5V
   GPIO3  (5) (6)  GND
   GPIO4  (7) (8)  GPIO14
     GND  (9) (10) GPIO15
```

### I2C Configuration
```bash
# Enable I2C interface
sudo raspi-config
# Select: Interface Options -> I2C -> Yes -> Finish -> Reboot

# Verify I2C is working
sudo i2cdetect -y 1
# Should show device at address 0x68
```

## IMU6950 Adaptation (If Needed)

### Required Changes
1. **Update I2C Address** (usually 0x69 for IMU6950)
2. **Modify Register Map** in `core/imu_driver.py`
3. **Adjust Scaling Factors** for IMU6950 sensitivity
4. **Update Configuration** in `config/settings.yaml`

### Example Driver Modification
```python
# In core/imu_driver.py
class IMU6950:
    def __init__(self, bus=1, address=0x69):  # Changed address
        # IMU6950 specific initialization
        self.bus.write_byte_data(self.address, 0x6B, 0x00)
        # Update register values for IMU6950
```

## Installation with Virtual Environment

### Method 1: Using uv (Recommended)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and setup
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Raspberry Pi
uv pip install -r requirements.txt
```

### Method 2: Traditional pip
```bash
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence
python3 -m venv venv
source venv/bin/activate  # On Raspberry Pi
pip install -r requirements.txt
```

## Testing Procedure

### 1. Basic Connectivity Test
```bash
# Activate virtual environment first
source .venv/bin/activate  # uv method
# or
source venv/bin/activate   # pip method

python test_integration.py
```

### 2. Expected Results (Stationary Device)
- Gravity magnitude: ~9.81 m/s²
- Error: <10%
- Standard deviation: <0.5 m/s²

### 3. Calibration Verification
- Device should be level during startup
- First 30 seconds: calibration phase
- Verify bias values are reasonable

### 4. Motion Learning Test
- Move device normally for 30 seconds
- System learns motion patterns
- Check console for training progress

### 5. Anomaly Detection Test
- Normal motion: score < 0.5
- Unusual motion: score > 0.85
- Check `anomalies.jsonl` for logged events

## Troubleshooting

### Common Issues
1. **I2C Device Not Found**
   - Check wiring connections
   - Verify I2C is enabled
   - Try different I2C bus (0 vs 1)

2. **Incorrect Readings**
   - Check power supply stability
   - Verify 3.3V power (not 5V)
   - Ensure device is stationary during calibration

3. **Performance Issues**
   - Reduce sampling rate in config
   - Check CPU usage
   - Use Docker for isolation

### Debug Commands
```bash
# Check I2C devices
sudo i2cdetect -y 1

# Monitor system resources
htop

# Check I2C speed
sudo raspi-config -> Advanced Options -> I2C
```

## Production Deployment

### Docker Setup
```bash
# Build and run with Docker
docker-compose up -d

# Check logs
docker-compose logs -f
```

### System Service
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run as background service
python main.py > imu.log 2>&1 &

# Monitor logs
tail -f imu.log
```

### Dashboard Access
```bash
# Activate virtual environment in separate terminal
source .venv/bin/activate

# Main system
python main.py

# Dashboard
python dashboard/web_ui.py
# Web interface: http://raspberry-pi-ip:5000
```
