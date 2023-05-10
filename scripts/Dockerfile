# Stage 1: Base image
FROM jupyter/base-notebook:python-3.9.12 as base
LABEL name="pymc" description="Environment for PyMC"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
USER $NB_UID

# Stage 2: Install Conda dependencies
FROM base as conda
COPY --chown=$NB_UID:users /conda-envs/environment-dev.yml .
RUN mamba env create -f environment-dev.yml && \
    conda clean --all -f -y

# Stage 3: Install PyMC
FROM conda as pymc
RUN /bin/bash -c ". activate pymc-dev && \
    mamba install -c conda-forge -y pymc"

# Stage 4: Final stage
FROM base
COPY --from=pymc --chown=$NB_UID:users /opt/conda /opt/conda

# Fix PkgResourcesDeprecationWarning
RUN pip install --upgrade --user setuptools==58.3.0

# Setup working folder
WORKDIR /home/jovyan/work

# For running from bash
SHELL ["/bin/bash","-c"]
RUN echo "conda activate pymc-dev" >> ~/.bashrc && \
    source ~/.bashrc

# For running from jupyter notebook
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "pymc-dev", "jupyter","notebook","--ip=0.0.0.0","--port=8888","--no-browser"]
