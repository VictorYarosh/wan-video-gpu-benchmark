# Wan 2.1 video generation benchmark.
# Runs on any Linux host with an NVIDIA GPU + NVIDIA Container Toolkit.
# On macOS the image can be built (--platform linux/amd64) but not run on a GPU.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

# Model weights are downloaded at first run into /cache (mount a volume to keep them).
VOLUME ["/cache", "/app/results"]

ENTRYPOINT ["python", "src/bench.py"]
CMD ["--mode", "single"]
