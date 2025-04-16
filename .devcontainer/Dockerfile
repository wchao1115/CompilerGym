# Base Image
FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install prerequisites
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libtinfo5 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create a default environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Activate environment by default
SHELL ["conda", "run", "-n", "comgym", "/bin/bash", "-c"]

# Clone a specific branch of gym to apply a patch
RUN git clone -b v0.21.0 --single-branch https://github.com/openai/gym.git /tmp/gym
COPY gym_0.21.0_patch/setup.py /tmp/gym/setup.py
RUN pip install -e /tmp/gym