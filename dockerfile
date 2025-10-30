# Sử dụng Python 3.10 để tương thích mediapipe-runtime
FROM python:3.10-slim

# Cài thư viện hệ thống cần thiết cho OpenCV/MediaPipe
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Cập nhật pip và cài dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cổng Flask/Gunicorn dùng
EXPOSE 5000

# Lệnh chạy app
CMD gunicorn app:app --bind 0.0.0.0:$PORT
