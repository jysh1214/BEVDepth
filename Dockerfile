# Download the nuscenes data set and put them to '$PWD/data'.
#   - v1.0-test_meta.tgz
#   - v1.0-test_blobs.tgz
#   - v1.0-trainval_meta.tgz
#   - v1.0-trainval01_blobs.tgz
#
# Build the docker image:
#   docker build -t bevdepth_image .
#
# Run the container:
#   docker run --name bevdepth_container --shm-size 64gb --gpus all --mount src=$PWD,target=/home/BEVDepth,type=bind -it bevdepth_image /bin/bash

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y cmake \
  ninja-build \
  libssl-dev \
  git \
  wget \
  vim \
  # for python3.9
  libffi-dev \
  zlib1g-dev \
  # for BEVDepth
  lzma \
  liblzma-dev \
  libbz2-dev

# Install Python3.9 for torch1.9.0.
ENV CPYTHON_ROOT="/home/cpython"
WORKDIR /home
RUN git clone https://github.com/python/cpython.git
WORKDIR $CPYTHON_ROOT
RUN git checkout v3.9.21
RUN ./configure
RUN make -j $(nproc)
RUN make install

# Create Python3.9 virtual environment and
# install the dependencies.
ENV VENV_PATH="/home/pyvenv"
RUN python3 -m venv $VENV_PATH
ENV PATH="$PATH:$VENV_PATH/bin"
RUN pip install numpy==1.23.5
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/open-mmlab/mmdetection3d.git@v1.0.0rc4
RUN pip install mmdet==2.26.0
# We encounter the error: 'RuntimeError: DeformConv is not compiled with GPU support'
# so we build mmcv from source instead of pip installing it
# RUN pip install mmcv-full==1.7.0
RUN pip install mmsegmentation==0.30.0

# Install BEVDepth's dependencies.
WORKDIR /tmp
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /home
CMD ["/bin/sh"]

# Activate Python virtual environment:
#   source /home/pyvenv/bin/activate

# Install mmcv.
# ENV MMCV_ROOT="/home/mmcv"
# WORKDIR /home
# RUN git clone https://github.com/open-mmlab/mmcv.git
# WORKDIR $MMCV_ROOT
# RUN git checkout v1.7.0
# RUN MMCV_WITH_OPS=1 pip install -e .

# Build BEVDepth:
#   cd $BEVDEPTH_ROOT
#   rm -rf build
#   python3 setup.py develop

# Prepare data:
#   cd data
#   mkdir nuscenes
#   tar zxvf v1.0-test_meta.tgz
#   tar zxvf v1.0-test_blobs.tgz
#   tar zxvf v1.0-trainval_meta.tgz
#   tar zxvf v1.0-trainval01_blobs.tgz
#
#   python3 scripts/gen_info.py

# TROUBLE SHOOTING:
# 
# vim /home/pyvenv/lib/python3.9/site-packages/networkx/algorithms/dag.py:23
# Change 'from fractions import gcd' to 'from math import gcd'
#
# vim /home/pyvenv/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:284
# Add 'accelerator = "gpu"'

# Download weights:
#   wget https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_24e_2key.pth
#
# Demo:
#   python3 bevdepth/exps/nuscenes/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py --ckpt_path bev_depth_lss_r50_256x704_128x128_24e_2key.pth -e -b 1 --gpus 1
