FROM dustynv/ros:noetic-pytorch-l4t-r35.1.0
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA
RUN apt-get update && \

    #-> Install general usage dependencies
    echo "Installing general usage dependencies ..." && \
    apt-get install -y \
    git \
    cmake \
    wget \
    tar \
    libx11-dev \
    xorg-dev \
    libssl-dev \
    build-essential \
    libusb-1.0-0-dev
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

WORKDIR /packages
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

#RUN git clone https://github.com/facebookresearch/detectron2.git && \
COPY detectron2 detectron2
RUN  cd detectron2 && ls && \
  pip install -e .


#RUN git clone https://github.com/facebookresearch/pytorch3d.git && \
COPY pytorch3d pytorch3d
RUN    cd pytorch3d && FORCE_CUDA=1 pip install -e .
RUN pip install cython scipy seaborn
WORKDIR /omni3d_ws

RUN pip install open3d
RUN apt-get install -y tmux


#RUN git clone https://github.com/facebookresearch/omni3d.git

