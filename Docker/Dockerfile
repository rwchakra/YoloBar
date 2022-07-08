# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
# FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04 

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

RUN set -x && \
    apt-get update -y && \
    apt-get upgrade -y && \
    apt install -y --no-install-recommends \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        python3-dev \
        python3-pip \ 
        python3-setuptools \
        cmake \
        swig \
        wget \
        unzip \
        screen \
        nano \
        lxc \
        ca-certificates \
        libfftw3-dev \
        expat \
        libgomp1 \
        protobuf-compiler 

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# install dependencies
## # CUDA 11.1
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

##CUDA 11.3
# RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html



# install detectron2
## CUDA 11.1 torch 1.9
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

##CUDA 11.3torch 1.10
# RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

## set environment variables for FastAPI 
# Possible error-> 
    # RuntimeError: Click will abort further execution because Python was configured to use ASCII as encoding for the environment. Consult https://click.palletsprojects.com/unicode-support/ for mitigation steps.

    # This system supports the C.UTF-8 locale which is recommended. You might be able to resolve your issue by exporting the following environment variables:

    #     export LC_ALL=C.UTF-8
    #     export LANG=C.UTF-8

ENV  LC_ALL=C.UTF-8 
ENV  LANG=C.UTF-8
# install libraries
WORKDIR /yolobar
COPY requirements.txt .
RUN pip3 install -r ./requirements.txt


# gRpc code
#COPY ./RestAPI ./RestAPI
# COPY ./scripts ./scripts
#COPY ./src ./src

# notebook code
#COPY ./notebooks ./notebooks
# COPY train.sh .
# rest api entry point 
#COPY run_restapi.sh .
# COPY start_tensorboard.sh .
# COPY evaluate.sh .
# COPY start_evaluation.sh .

# COPY detectron_serving.py detectron_serving.py

# CMD "python3"  "detectron_serving.py"

