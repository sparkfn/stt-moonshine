FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/*.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
