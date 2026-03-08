FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ONNX Runtime: max graph optimization, parallel execution
ENV ONNXRUNTIME_SESSION_GRAPH_OPTIMIZATION_LEVEL=99
ENV ONNXRUNTIME_EXECUTION_MODE=1

# All model caches under one tree for easy bind-mounting
ENV HF_HOME=/data/cache/huggingface
ENV MOONSHINE_VOICE_CACHE=/data/cache/moonshine

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user; pre-create cache dirs with correct ownership
RUN groupadd -r app && useradd -r -g app -d /app app \
    && mkdir -p /data/cache /data/logs \
    && chown -R app:app /app /data

COPY src/*.py /app/

USER app

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ws", "websockets", "--loop", "uvloop", "--http", "httptools", \
     "--no-access-log"]
