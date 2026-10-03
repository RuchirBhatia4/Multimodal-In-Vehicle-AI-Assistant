#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, model weights (~5 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
mkdir -p models
[ -f models/face_landmarker.task ] || curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["mlx-community/whisper-large-v3-turbo", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"]:
    snapshot_download(repo)
from sentence_transformers import SentenceTransformer
SentenceTransformer("clip-ViT-B-32")
import os
os.chdir("models")
from ultralytics import YOLO
YOLO("yolo11n.pt")
PY
[ -f .env ] || cp .env.example .env
echo "Setup complete. Run: scripts/run.sh"
