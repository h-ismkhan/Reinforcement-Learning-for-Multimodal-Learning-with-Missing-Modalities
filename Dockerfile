# syntax=docker/dockerfile:1
#
# One Dockerfile, two flavours:
#   CPU (default) – demo, tests, API:
#       docker build -t rl4m4 .
#   GPU – real CMU-MOSI runs (needs NVIDIA Container Toolkit on the host):
#       docker build -t rl4m4:gpu \
#         --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
#         --build-arg INSTALL_MEDIA_LIBS=true .
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG INSTALL_MEDIA_LIBS=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    BUNDLE_DIR=/app/artifacts/bundle

WORKDIR /app

# ffmpeg / libsndfile are only needed to decode real MOSI audio & video.
RUN if [ "$INSTALL_MEDIA_LIBS" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Dependencies first (better layer caching), source code last.
COPY requirements.txt requirements-serve.txt ./
RUN pip install torch torchaudio --index-url ${TORCH_INDEX} \
 && pip install -r requirements.txt -r requirements-serve.txt

COPY rl.py mosi_reg.py demo_synthetic.py serve.py run_mosi.py ./

# Run as non-root; make the output / HF-cache dirs writable.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/artifacts /opt/hf-cache \
 && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 0

# Default: dataset-free end-to-end demo, writes to /app/artifacts
CMD ["python", "demo_synthetic.py", "--out-dir", "artifacts"]
