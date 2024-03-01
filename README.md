# Application for clinical research into Cell Nucli Analysis

Tested both locally and through docker on Ubuntu 22.04.3 LTS (WSL2) with NVIDIA RTX 3070Ti. 

Requires Python 3.8

**Currently can only run with GPU** (working on CPU support)

## Features
- Analysis of tissue images pre and post segmentation

- Evaluate a wide range of image segmentation models (semantic and instance) and an end-to-end pipeline with object detector and the Segment Anything Model (SAM)

- Visual analysis of prediction from models as outlines on the image and clearly see differencees from prediction and ground truth

- support for newer models from both mmsegmentation and mmdetection (with mmyolo coming soon)

- Extensive information on the top 4 datasets currently used in single cell nuclei analysis (MoNuSeg, MoNuSAC, CryoNuSeg, TNBC)

- Visually compare the differences in the top 2 augmentation techniques used in training models (Stain Normalistion and Contrast Limited Adaptive Histogram Equalisation)


## Key Technology Used

- Web interface: [streamlit](https://streamlit.io/)
- model implementations: 
    - [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main)
    - [MMDetection](https://github.com/open-mmlab/mmdetection)
    - [MMYOLO](https://github.com/open-mmlab/mmyolo)
    - [SAM](https://github.com/facebookresearch/segment-anything)
- Image displaying: [Plotly](https://plotly.com/)


## Installation

### Docker (Easy)

1. Either clone the whole repository or just download the dockerfile provided

2. Make sure docker is install properly and able to use docker as a non-root user (if you are on linux) [docker_post_install](https://docs.docker.com/engine/install/linux-postinstall/#:~:text=If%20you%20don)
```bash
# Instructions copied from docker site: 
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
# test to make sure it works
docker run hello-world
```
3. Build the image from the dockerfile (**it is ~ 16GB**)
```bash
docker build . -f dockerfile -t medcnsapp
```
4. Run the built image as a container (with NVIDIA enabled)(detached)
```bash
docker run -dp 127.0.0.1:8501:8501 --runtime=nvidia medcnsapp
# or if you want to have it attached
docker run -p 127.0.0.1:8501:8501 --runtime=nvidia medcnsapp
```
5. To stop the container:
```bash
# see the container ID of the running container under medcnsapp
docker ps -a
# stop the container
docker stop >ID<
```

### Source
1. clone the full repository to a directory of your choice (or download as zip)
```bash
git clone https://github.com/ChakiCurtin/medCNSapp.git
cd mmyolo_sam_app/
```
2. Some form of conda or venv is recommended for managing dependencies
    - recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. create conda environment
```bash
conda create --name medcnsapp python=3.8 -y
conda activate medcnsapp
```
3. Install [Pytorch](https://pytorch.org/get-started/locally/) according to the instructions
```bash
# As of 12 Jul 2023
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```
4. Install required dependencies
```bash
pip3 install -r requirements.txt
```
5. Install rest of the required dependencies
    - [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)
    - [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html)
    - [MMYOLO](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html)
```bash
mim install mmengine==0.10.1
mim install mmcv==2.0.1
mim install mmsegmentation==1.2.1
mim install mmdet==3.2.0
mim install mmyolo==0.6.0
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Usage
If docker was used in the installation process, in a new tab or webpage, type "http://localhost:8501/" to view the application. This is the main interface you will be interacting with.

If you have installed from source, type:
```bash
python3 -m streamlit run home.py --server.headless true --browser.gatherUsageStats false
```

## To Uninstall

## Docker
1. Stop the container

2. delete the container
```bash
# Find container ID of the stopped container
docker ps -a
# Remove the container 
docker container rm <ID>
# Remove the full docker image
docker image rm <ID>
```

## Source
1. If you used a virtual environment (Conda or VenV), just remove the environment and all dependencies will be deleted
2. Delete the repository directory