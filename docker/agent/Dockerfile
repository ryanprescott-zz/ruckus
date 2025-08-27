# Multi-stage Dockerfile for RUCKUS Agent with GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /ruckus

# Copy project files
COPY . /ruckus/

# Install Python dependencies
WORKDIR /ruckus/common
RUN pip install -e .

WORKDIR /ruckus/agent
RUN pip install -e .

# Install additional ML dependencies
RUN pip install \
    torch \
    torchvision \
    torchaudio \
    transformers \
    tokenizers \
    datasets \
    accelerate \
    scipy \
    numpy \
    pandas \
    scikit-learn \
    nltk \
    rouge-score \
    sacrebleu \
    pynvml \
    psutil \
    GPUtil

# Install vLLM (requires CUDA)
RUN pip install vllm

# Create models directory
RUN mkdir -p /ruckus/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash ruckus
RUN chown -R ruckus:ruckus /ruckus
USER ruckus

# Set working directory for agent
WORKDIR /ruckus/agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose agent port
EXPOSE 8081

# Default command
CMD ["python", "-m", "ruckus_agent.main"]