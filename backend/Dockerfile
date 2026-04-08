FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.2" \
    python-multipart==0.0.20 \
    insightface==0.7.3 \
    onnxruntime==1.21.1 \
    opencv-python-headless==4.11.0.86 \
    numpy==1.26.4 \
    Pillow==11.2.1 \
    httpx==0.28.1

# Download models into an image cache, then hydrate the runtime volume on startup.
ENV MODEL_CACHE_DIR=/opt/model-cache
COPY download_models.sh start.sh ./
RUN chmod +x download_models.sh start.sh && MODEL_DIR=$MODEL_CACHE_DIR bash download_models.sh

COPY main.py swapper.py face_masker.py ./

EXPOSE 8000
CMD ["./start.sh"]
