# Ubuntu with CUDA
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        pkg-config \
        unzip

# zsh (from [1]) ----------------
RUN apt-get update && apt-get install -y wget git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc

# X window
RUN apt-get update && apt-get install -y \
    vim \
    libsdl2-2.0-0 \ 
    cmake \
    xorg-dev freeglut3-dev \
    libglu1-mesa-dev \
    x11-apps

# robomimic dependencies ----------------
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    python3-pip

# Install Mambaforge -> robotsuite env
COPY conda_environment.yaml /root/

ENV MAMBA_DIR=/opt/mamba
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p $MAMBA_DIR \
    && rm Miniforge3-Linux-x86_64.sh

# Initialize Mamba environment
ENV PATH="$MAMBA_DIR/bin:$PATH"
RUN mamba init zsh && \
    mamba env create -f /root/conda_environment.yaml

RUN echo 'mamba init' >> /root/.zshrc
RUN echo 'mamba activate rdt' >> /root/.zshrc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
    
# finalize image  ----------------
RUN apt-get clean && \
        rm -rf /var/lib/apt/lists/*
    
RUN source /root/.zshrc
    
WORKDIR /root
CMD ["zsh"]