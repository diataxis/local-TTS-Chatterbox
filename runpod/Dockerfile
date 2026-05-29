# Extends the RunPod ComfyUI base image with Chatterbox TTS server.
# ComfyUI runs on port 8188, Chatterbox TTS on port 3200.

ARG BASE_IMAGE=runpod/comfyui:latest
FROM ${BASE_IMAGE}

# ==========================================
# Chatterbox TTS — isolated in its own venv
# to avoid numpy conflicts with ComfyUI/Impact Pack.
# Chatterbox needs numpy<1.26, ComfyUI needs numpy>=2.0.
# ==========================================
RUN python3 -m venv /opt/chatterbox-venv --system-site-packages && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir --no-deps chatterbox-tts && \
    /opt/chatterbox-venv/bin/pip install --no-cache-dir "numpy==1.26.4" \
    librosa==0.11.0 \
    s3tokenizer \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    resemble-perth==1.0.1 \
    conformer==0.3.2 \
    safetensors==0.5.3 \
    spacy-pkuseg \
    pykakasi==2.3.0 \
    pyloudnorm \
    omegaconf \
    fastapi \
    python-multipart \
    torchcodec \
    uvicorn[standard]

# ==========================================
# ComfyUI custom nodes — uses system Python (numpy 2.x is fine here)
# ==========================================
RUN cd /opt/comfyui-baked/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    pip install --no-cache-dir -r requirements.txt && \
    PYTHONPATH=/opt/comfyui-baked python install.py || echo "install.py warning (non-fatal)" && \
    cd /opt/comfyui-baked/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && \
    pip install --no-cache-dir -r requirements.txt

# Download face detection + SAM models for FaceDetailer
RUN mkdir -p /opt/comfyui-baked/models/ultralytics/bbox && \
    mkdir -p /opt/comfyui-baked/models/ultralytics/segm && \
    mkdir -p /opt/comfyui-baked/models/sams && \
    curl -sL -o /opt/comfyui-baked/models/ultralytics/bbox/face_yolov8m.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" && \
    curl -sL -o /opt/comfyui-baked/models/ultralytics/segm/face_yolov8m-seg_2.pt \
      "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m-seg_2.pt" && \
    curl -sL -o /opt/comfyui-baked/models/sams/sam_vit_b_01ec64.pth \
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Copy TTS server and voice presets
COPY server.py /opt/chatterbox-tts/server.py
COPY voices/ /opt/chatterbox-tts/voices/

# HuggingFace model cache: symlink to workspace so models persist across pod restarts.
# The actual symlink is created at runtime in start-all.sh (workspace volume isn't
# available at build time).

# Startup wrapper: launches TTS server, then hands off to the original ComfyUI start.sh
COPY runpod/start-all.sh /start-all.sh
RUN chmod +x /start-all.sh

EXPOSE 3200

ENTRYPOINT ["/start-all.sh"]
