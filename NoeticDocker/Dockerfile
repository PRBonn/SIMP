FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3 \
    python3-dev \
    python3.9 \
    python3.9-dev \
    python3-pip \
    cmake \
    wget \
    tar \
    libx11-dev \
    xorg-dev \
    libssl-dev \
    build-essential \
    tmux \
    libusb-1.0-0-dev && apt-get clean && rm -rf /var/lib/apt/lists/*



# Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*
RUN apt-add-repository universe
RUN apt-add-repository multiverse
RUN apt-add-repository restricted


RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install opencv-python

WORKDIR /packages
RUN git clone https://github.com/facebookresearch/detectron2.git &&  cd detectron2 && ls && pip install -e .
RUN pip install fvcore iopath
RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
RUN tar xzf 1.10.0.tar.gz
ENV CUB_HOME=$PWD/cub-1.10.0
#RUN git clone https://github.com/facebookresearch/pytorch3d.git
#RUN cd pytorch3d && FORCE_CUDA=1 pip install -e .
#RUN pip install cython scipy seaborn
#RUN git clone https://github.com/facebookresearch/omni3d.git
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
RUN pip install cython scipy seaborn


# Install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update && apt-get install -y --allow-downgrades --allow-remove-essential --allow-change-held-packages \
libpcap-dev \
libopenblas-dev \
gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
ros-noetic-desktop-full python3-rosdep python3-rosinstall-generator python3-vcstool build-essential && \
apt-get clean && rm -rf /var/lib/apt/lists/*
# Configure ROS
RUN rosdep init && rosdep update 
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc


# Install our code

RUN apt-get update && \
    apt-get install -y nlohmann-json3-dev libgtest-dev libeigen3-dev ros-noetic-map-server vim && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/PRBonn/SIMP.git
RUN cd SIMP/ncore && mkdir build && cd build && cmake .. -DBUILD_TESTING=1 && make -j12 
RUN cd SIMP/ros1_ws/src && git clone https://github.com/FullMetalNicky/omni3d
RUN pip install open3d
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd SIMP/ros1_ws; catkin_make'
RUN cd SIMP/ros1_ws/src/omni3d_ros/ && mkdir -p trained_models/22_12_20_server_224/ && cd trained_models/22_12_20_server_224/ \
&& curl -LO https://www.ipb.uni-bonn.de/html/projects/simp/model_recent.pth

