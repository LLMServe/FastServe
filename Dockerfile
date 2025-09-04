# Dockerfile for fastserve
# Build with: docker build -t fastserve .

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
COPY . /app
WORKDIR /app

RUN rm -rf build \
    && apt update \
    && apt install -y python3 pip python-is-python3 cmake build-essential git openmpi-bin libopenmpi-dev \
    && apt-get clean all \
    && pip3 install mpmath==1.2.1 tabulate==0.9.0 transformers==4.31.0 psutil==5.9.5 regex==2023.6.3 pandas==2.1.0 \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set up the entry point
ENTRYPOINT ["bash"]
