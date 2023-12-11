FROM ghcr.io/mamba-org/micromamba-devcontainer:latest

COPY --chown=${MAMBA_USER}:${MAMBA_USER} conda-envs/environment-dev.yml /tmp/environment-dev.yml
RUN : \
    && micromamba install --yes --name base --file /tmp/environment-dev.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment-dev.yml \
    && sudo chmod -R a+rwx /opt/conda \
;

# Run subsequent commands in an activated Conda environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
