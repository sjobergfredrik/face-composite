#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/app/models}"
mkdir -p "$MODEL_DIR"

download() {
  local url="$1"
  local dest="$2"
  if [ -f "$dest" ]; then
    echo "✓ $(basename "$dest") (exists)"
  else
    echo "↓ Downloading $(basename "$dest")..."
    curl -L --progress-bar --retry 3 -o "$dest" "$url"
    echo "✓ $(basename "$dest") ($(du -h "$dest" | cut -f1))"
  fi
}

echo "=== Downloading models ==="

# InsightFace buffalo_l (face detection + recognition)
echo ""
echo "InsightFace buffalo_l..."
python3 -c "
import os
os.environ['INSIGHTFACE_HOME'] = '$MODEL_DIR'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='$MODEL_DIR')
app.prepare(ctx_id=0, det_size=(640, 640))
print('buffalo_l ready')
"

# inswapper_128
echo ""
download "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" \
  "$MODEL_DIR/inswapper_128.onnx"

# hyperswap_1c_256
echo ""
download "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx" \
  "$MODEL_DIR/hyperswap_1c_256.onnx"

# Face occlusion mask model (detects hair, hands, glasses overlapping face)
echo ""
echo "Face occluder (xseg_1)..."
download "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx" \
  "$MODEL_DIR/xseg_1.onnx"

# Face parser model (semantic segmentation: skin, eyes, nose, mouth, etc.)
echo ""
echo "Face parser (bisenet_resnet_34)..."
download "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx" \
  "$MODEL_DIR/bisenet_resnet_34.onnx"

echo ""
echo "=== All models downloaded ==="
