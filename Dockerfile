FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    i2c-tools \
    python3-smbus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV I2C_BUS=1
ENV PYTHONPATH=/app

# Expose data volume
VOLUME ["/app/data"]

# Default command for inference mode
CMD ["python", "main.py"]