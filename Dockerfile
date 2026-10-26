# FMV Intelligence Pipeline — GPU-enabled Docker container
# Phase 2+ (C++ hot path + Python sidecar)
#
# Build:  docker build -t fmv-pipeline:latest .
# Run:    docker run --gpus all --runtime=nvidia \
#           -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
#           -v $(pwd)/output:/app/output \
#           fmv-pipeline:latest /app/build/fmv_pipeline <source>

# ── Base: NVIDIA CUDA 12.4 with cuDNN ─────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System build tools & libraries ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    cmake ninja-build pkg-config git wget curl \
    build-essential ca-certificates \
    # FFmpeg libs (for libavcodec/libavformat — NVDEC build done separately)
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    # Geospatial
    libgdal-dev libproj-dev \
    # Kafka output
    librdkafka-dev \
    # OpenCV dependencies
    libgtk-3-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    # Python
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python sidecar dependencies ───────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── TensorRT ──────────────────────────────────────────────────────────────────
# Option A (recommended): pull the TensorRT NGC container as a multi-stage base
#   FROM nvcr.io/nvidia/tensorrt:24.08-py3 AS trt_base
#   COPY --from=trt_base /usr/local/tensorrt /usr/local/tensorrt
#
# Option B: install via NVIDIA apt repo (uncomment and adjust version):
# RUN apt-get install -y tensorrt libnvinfer-dev libnvonnxparsers-dev

# ── OpenCV with CUDA support ──────────────────────────────────────────────────
# TODO: Build OpenCV from source with -DWITH_CUDA=ON for cv::cuda::GpuMat support
# RUN git clone --depth 1 -b 4.10.0 https://github.com/opencv/opencv.git /opt/opencv
# RUN cmake -B /opt/opencv/build -S /opt/opencv -GNinja \
#     -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCUDA_ARCH_BIN="8.6;8.9;9.0" \
#     -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
# RUN cmake --build /opt/opencv/build && cmake --install /opt/opencv/build

# ── Copy source and build C++ pipeline ───────────────────────────────────────
WORKDIR /app
COPY . .

# Build C++ hot path (requires TensorRT and CUDA-enabled OpenCV — see above)
# RUN cmake -B /app/build -S /app/src -GNinja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DTENSORRT_ROOT=/usr/local/tensorrt \
#     -DOpenCV_DIR=/opt/opencv/build
# RUN cmake --build /app/build

# ── Runtime environment ───────────────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# POC: Python pipeline entry point
# Phase 2: switch ENTRYPOINT to /app/build/fmv_pipeline
ENTRYPOINT ["python3", "fmv_pipeline.py"]
CMD ["--help"]
