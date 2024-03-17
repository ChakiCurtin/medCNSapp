ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="2.0.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install openmim==0.3.9"]
RUN ["/bin/bash", "-c", "mim install mmengine==0.10.1"]
RUN ["/bin/bash", "-c", "mim install mmcv==2.0.1"]
# check to see if mmsegmenation can be run through mim install or pip
RUN ["/bin/bash", "-c", "pip install mmsegmentation==1.2.1"]
RUN ["/bin/bash", "-c", "mim install mmdet==3.2.0"]
RUN ["/bin/bash", "-c", "mim install mmyolo==0.6.0"]
RUN ["/bin/bash", "-c", "pip install ftfy==6.1.2"]
RUN ["/bin/bash", "-c", "pip install regex"]
RUN ["/bin/bash", "-c", "pip install pandas==2.0.3"]
RUN ["/bin/bash", "-c", "pip install numpy==1.24.3"]
RUN ["/bin/bash", "-c", "pip install matplotlib"]
RUN ["/bin/bash", "-c", "pip install seaborn==0.13.0"]
RUN ["/bin/bash", "-c", "pip install streamlit==1.28.0"]
RUN ["/bin/bash", "-c", "pip install plotly==5.18.0"]
RUN ["/bin/bash", "-c", "pip install streamlit-image-comparison==0.0.4"]
RUN ["/bin/bash", "-c", "pip install st-clickable-images==0.0.3"]
RUN ["/bin/bash", "-c", "pip install Jinja2==3.1.2"]
RUN ["/bin/bash", "-c", "pip install git+https://github.com/facebookresearch/segment-anything.git"]
RUN ["/bin/bash", "-c", "pip install gdown"]
RUN ["/bin/bash", "-c", "pip install torchstain==1.3.0"]

#Install application
RUN git clone -b main https://github.com/ChakiCurtin/medCNSapp.git /mmyolo_sam_st
WORKDIR /mmyolo_sam_st
ENV FORCE_CUDA="1"
#RUN pip install -r requirements.txt # No need to install requirements when installing container-wide
WORKDIR /mmyolo_sam_st
# DOCKER RUN CMD
CMD ["streamlit","run", "home.py"]
EXPOSE 8501