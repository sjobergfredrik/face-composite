#!/bin/sh
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/opt/model-cache}"

mkdir -p "$MODEL_DIR"

if [ -d "$MODEL_CACHE_DIR" ]; then
  cp -an "$MODEL_CACHE_DIR"/. "$MODEL_DIR"/
fi

if [ ! -d "$MODEL_DIR/models/buffalo_l" ] || \
   [ ! -f "$MODEL_DIR/inswapper_128.onnx" ] || \
   [ ! -f "$MODEL_DIR/hyperswap_1c_256.onnx" ] || \
   [ ! -f "$MODEL_DIR/xseg_1.onnx" ] || \
   [ ! -f "$MODEL_DIR/bisenet_resnet_34.onnx" ]; then
  echo "Downloading missing models into $MODEL_DIR..."
  MODEL_DIR="$MODEL_DIR" bash /app/download_models.sh
fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
