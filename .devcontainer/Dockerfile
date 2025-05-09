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

# Clone a specific branch of gym to apply a patch. OpenAI gym is a core dependency of the library.
RUN git clone -b v0.21.0 --single-branch https://github.com/openai/gym.git /tmp/gym
COPY gym_0.21.0_patch/setup.py /tmp/gym/setup.py
RUN pip install -e /tmp/gym

# To build Compiler-Gym locally, follow these steps:
#
# ``` bash
#   .devcontainer/build_prep.sh # prepare the build prereqs e.g. Bazel, Clang, etc.
#   make install
# ```
#
# Examples:
# To run any example after building Compiler-Gym, you need to install the examples dependencies:
#
# ``` bash
#   cd examples
#   make install
# ```
#
# Then, go to the Testing (beaker icon on the left menu pane). If you set everything correctly 
# with both the Compiler-Gym and the examples dependencies installed, you should be able to see
# the tests and run them.
#
# Here is a suggested launch.json and settings.json edits for VSCode to use when 
# debugging anything under the examples directory.
#
# (launch.json)
# {
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#            "name": "Python Debugger: Current File",
#            "type": "debugpy",
#            "request": "launch",
#            "cwd": "${workspaceFolder}/examples",
#            "program": "${file}",
#            "console": "integratedTerminal",
#            "justMyCode": false
#         }
#     ]
# }
#
# (settings.json)
# {
#     "python.testing.framework": "pytest",
#     "python.testing.pytestEnabled": true,
#     "python.testing.autoTestDiscoverOnSaveEnabled": true,
#     "python.testing.cwd": "${workspaceFolder}/examples",
#     "python.testing.pytestArgs": ["example_compiler_gym_service"]
# }
