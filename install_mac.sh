#!/usr/bin/env bash
set -euo pipefail

# Chatterbox TTS - Apple Silicon Mac installation script
# Requires: conda activate chatterbox (Python 3.11)
# Note: Skips pkuseg (only needed for Chinese) — English works fine without it

echo "=== Upgrading pip/setuptools ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch ==="
pip install torch==2.6.0 torchaudio==2.6.0

echo "=== Installing Chatterbox dependencies ==="
pip install transformers==4.46.3 diffusers==0.29.0 \
  conformer==0.3.2 resemble-perth==1.0.1 safetensors==0.5.3 \
  librosa==0.11.0 pykakasi==2.3.0

echo "=== Installing s3tokenizer ==="
pip install onnx==1.16.2
pip install --no-deps s3tokenizer==0.2.0

echo "=== Installing chatterbox-tts (without pkuseg — Chinese only dep) ==="
pip install --no-deps chatterbox-tts

echo "=== Installing pyloudnorm + omegaconf (runtime deps) ==="
pip install pyloudnorm omegaconf

echo "=== Installing FastAPI server deps ==="
pip install fastapi==0.110.1 uvicorn==0.29.0 python-multipart==0.0.9

echo ""
echo "=== Done! Start the server with: ==="
echo "uvicorn server:app --host 0.0.0.0 --port 3200"
