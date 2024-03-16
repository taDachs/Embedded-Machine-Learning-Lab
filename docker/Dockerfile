FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

COPY torch-1.8.1a0+56b43f4-cp36-cp36m-linux_aarch64.whl /root/
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN apt-get install -y libopenblas-base libopenmpi-dev

RUN apt install -y python3-pip
RUN apt install -y libomp5

RUN python3 -m pip install pip --upgrade 
RUN python3 -m pip install numpy
RUN python3 -m pip install /root/torch-1.8.1a0+56b43f4-cp36-cp36m-linux_aarch64.whl

RUN python3 -m pip install jupyterlab matplotlib thop
RUN python3 -m pip install nbimporter tqdm

#install onnx
COPY onnxruntime_gpu-1.9.0-cp36-cp36m-linux_aarch64.whl/ /
WORKDIR / 
RUN pip3 install onnxruntime_gpu-1.9.0-cp36-cp36m-linux_aarch64.whl

#install opencv
COPY opencv_python-4.5.5.62-cp36-cp36m-linux_aarch64.whl /
WORKDIR /
RUN pip3 install opencv_python-4.5.5.62-cp36-cp36m-linux_aarch64.whl

RUN python3 -m pip install ipywidgets
RUN python3 -m pip install onnx
RUN python3 -m pip install torchinfo

RUN git clone https://github.com/NVIDIA-AI-IOT/jetcam
WORKDIR /jetcam/
RUN python3 setup.py install
WORKDIR /

RUN adduser --home /home/jetson --shell /bin/bash --uid 1000 --disabled-password --gecos "" jetson
RUN adduser jetson video

RUN pip3 install pyaml pandas
