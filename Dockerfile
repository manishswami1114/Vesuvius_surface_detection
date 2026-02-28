FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# copy source
COPY src/ ./src/
COPY configs/ ./configs/
WORKDIR /workspace/src

ENV PYTHONPATH=/workspace/src

# default command: show help
CMD ["python3", "-m", "vesuvius.cli", "--help"]
