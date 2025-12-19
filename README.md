# Autonomous Self calibrating IMU Edge Intelligence Module

## Overview
IMU intelligence is a universal IMU module that automatically learns any machine dynamics and detect anomalies without labeled data, and outputs a standarized motion fingerprint on a Edge Device.

## Features 
- Self calibrate in <30 seconds
- Extended Kalman filrer with online noise adaptation
- 32-dimensional fingerprint of machine behaviour
- Unsupervised anomaly detection, no labeled data needed 
- Real time dashboard

## Installation

# Clone Repo 
git clone https://github.com/Ankit-x1/imu_intelligence
cd imu_intelligence
pip install -r requirements.txt

## Enable I2C on Raspberry Pi
sudo raspi-config

## Running 
python main.py

# Dasboard
python dashboard/web_ui.py