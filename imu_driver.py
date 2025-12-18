import smbus2
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class IMUData:
    timestamp: float
    accel: np.ndarray  
    gyro: np.ndarray   
    temp: float        
    
class MPU6050:
    def __init__(self, bus=1, address=0x68):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        
        self.bus.write_byte_data(self.address, 0x6B, 0x00)  # Wake up
        self.bus.write_byte_data(self.address, 0x1B, 0x08)  # Gyro ±500°/s
        self.bus.write_byte_data(self.address, 0x1C, 0x08)  # Accel ±4g
        
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_scale = np.ones(3)
        
    def read_raw(self) -> Tuple[np.ndarray, np.ndarray, float]:
        data = self.bus.read_i2c_block_data(self.address, 0x3B, 14)
        
        accel = np.array([
            self._twos_complement(data[0] << 8 | data[1]) / 8192.0 * 9.81,
            self._twos_complement(data[2] << 8 | data[3]) / 8192.0 * 9.81,
            self._twos_complement(data[4] << 8 | data[5]) / 8192.0 * 9.81
        ])
        
        temp = self._twos_complement(data[6] << 8 | data[7]) / 340.0 + 36.53
        
        gyro = np.array([
            self._twos_complement(data[8] << 8 | data[9]) / 65.5,
            self._twos_complement(data[10] << 8 | data[11]) / 65.5,
            self._twos_complement(data[12] << 8 | data[13]) / 65.5
        ])
        
        accel = (accel - self.accel_bias) * self.accel_scale
        gyro = (gyro - self.gyro_bias) * self.gyro_scale
        
        return IMUData(time.time(), accel, np.radians(gyro), temp)
    
    def _twos_complement(self, val):
        return val - 65536 if val >= 32768 else val
    
    def set_calibration(self, accel_bias, gyro_bias, accel_scale, gyro_scale):
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.accel_scale = accel_scale
        self.gyro_scale = gyro_scale