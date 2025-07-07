FROM jupyter/base-notebook:python-3.9.12

LABEL name="pymc"
LABEL description="Environment for PyMC version 4"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Switch to jovyan to avoid container runs as root
USER $NB_UID

COPY /conda-envs/environment-dev.yml .
RUN mamba env create -f environment-dev.yml && \
    /bin/bash -c ". activate pymc-dev && \
    mamba install -c conda-forge -y pymc" && \
    conda clean --all -f -y

# Fix PkgResourcesDeprecationWarning
RUN pip install --upgrade --user setuptools==58.3.0

#Setup working folder
WORKDIR /home/jovyan/work

# For running from bash
SHELL ["/bin/bash","-c"]
RUN echo "conda activate pymc-dev" >> ~/.bashrc && \
    source ~/.bashrc

# For running from jupyter notebook
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "pymc-dev", "jupyter","notebook","--ip=0.0.0.0","--port=8888","--no-browser"]
