FROM jupyter/base-notebook:python-3.9.12

LABEL name="pymc"
LABEL description="Environment for PyMC version 4"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git-core git mercurial subversion \
    build-essential \
    byobu \
    curl \
    htop \
    libcupti-dev \
    libfreetype6-dev \
    libzmq3-dev \
    pkg-config \
    rsync \
    software-properties-common \
    unzip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

COPY /conda-envs/environment-dev-py39.yml .
RUN mamba env create -f environment-dev-py39.yml && \
    /bin/bash -c ". activate pymc-dev-py39 && \
    mamba install -c conda-forge -y pymc" && \
    conda clean --all -f -y

# Fix PkgResourcesDeprecationWarning
RUN pip install --upgrade --user setuptools==58.3.0

#Setup working forder
RUN mkdir /home/jovyan/workspace
WORKDIR /home/jovyan/workspace

# For running from bash
SHELL ["/bin/bash","-c"]
RUN echo "conda activate pymc-dev-py39" >> ~/.bashrc && \
    source ~/.bashrc

# For running from jupyter notebook
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "pymc-dev-py39", "jupyter","notebook","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root"]
