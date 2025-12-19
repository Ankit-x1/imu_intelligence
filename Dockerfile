FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    i2c-tools \
    python3-smbus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV I2C_BUS=1

CMD ["python", "main.py"]