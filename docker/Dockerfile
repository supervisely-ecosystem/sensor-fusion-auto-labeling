FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    git \
    ffmpeg \
    libgeos-dev \
    libsm6 \
    libexiv2-dev \
    libxrender-dev \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir packaging==22.0

RUN pip3 install --no-cache-dir open3d==0.16.0
RUN python3 -m pip install --no-cache-dir supervisely==6.73.330
RUN pip3 install --no-cache-dir jsonschema==4.20.0
RUN pip3 install --no-cache-dir ultralytics==8.3.100
RUN pip3 install --no-cache-dir python-json-logger==2.0.7