# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
# FROM ubuntu:22.04

# Install git, compiler, cmake
RUN apt-get update && apt-get -y install \
build-essential cmake git git-lfs && \
# Install pangolin pre-reqs
apt-get -y install libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev \
libc++-dev libglew-dev libeigen3-dev cmake g++ ninja-build \
libjpeg-dev libpng-dev \
libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev

# Install other ORB-SLAM3 and CUDA dependencies
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get -y install libopencv-dev libopencv-core-dev libeigen3-dev libboost-serialization-dev libssl-dev 

# RUN apt-get -y install nvidia-cuda-toolkit nvidia-cuda-dev nvidia-cuda-gdb

# Install CUDA Toolkit 12.6
RUN apt-get -y install cuda-toolkit-12-6 cuda-gdb-12-6

# ORB-SLAM3 Stuff
# Install pangolin
RUN apt-get -y install python3-dev python3-setuptools
RUN git clone --branch v0.6 --recursive https://github.com/stevenlovegrove/Pangolin.git && \
cd Pangolin && \
cmake -B build -GNinja && \
cmake --build build && \
cd build && ninja install