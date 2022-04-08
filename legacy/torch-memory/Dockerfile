FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

# RUN apt-get update && apt-get -y install cmake

RUN export PATH="/usr/local/cuda/bin:$PATH"
RUN export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
RUN export CUDA_HOME="/usr/local/cuda"
RUN apt-get -y install --reinstall gcc 

RUN conda install astunparse numpy pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
