# Base Image
FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install prerequisites as root user
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
    chown -R vscode:vscode /opt/conda && \
    conda clean -afy

# Switch to vscode user to install conda and the rest.
USER vscode
WORKDIR /home/vscode

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Copy Conda environment and create it
COPY environment.yml /home/vscode/environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Create a workspace directory
RUN mkdir -p /home/vscode/gym

# Clone gym repo and patch it
RUN git config --global http.postBuffer 524288000  # 500MB
RUN git clone -b v0.21.0 --single-branch --depth 1 https://github.com/openai/gym.git gym
COPY gym_0.21.0_patch/setup.py ./gym/setup.py

# Ensure subsequent shell runs use the specified conda environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate comgym" >> /home/vscode/.bashrc


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
