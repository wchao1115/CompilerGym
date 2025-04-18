#!/bin/bash
sudo apt update
sudo apt install -y clang clang-format golang libjpeg-dev libtinfo5 m4 make patch zlib1g-dev tar bzip2 wget patchelf
mkdir -pv ~/.local/bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O ~/.local/bin/bazel
wget https://github.com/hadolint/hadolint/releases/download/v1.19.0/hadolint-Linux-x86_64 -O ~/.local/bin/hadolint
chmod +x ~/.local/bin/bazel ~/.local/bin/hadolint
go install github.com/bazelbuild/buildtools/buildifier@latest
GO111MODULE=on go install github.com/uber/prototool/cmd/prototool@dev
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
