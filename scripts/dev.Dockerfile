FROM ghcr.io/mamba-org/micromamba-devcontainer:git-a49ee3e

COPY --chown=${MAMBA_USER}:${MAMBA_USER} conda-envs/environment-dev.yml /tmp/environment-dev.yml
RUN : \
    && micromamba install --yes --name base --file /tmp/environment-dev.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment-dev.yml \
    && sudo chmod a+rwx /opt/conda \
;

ARG MAMBA_DOCKERFILE_ACTIVATE=1

COPY --chown=${MAMBA_USER}:${MAMBA_USER} .pre-commit-config.yaml /fake-repo/.pre-commit-config.yaml
RUN : \
    && cd /fake-repo \
    && git init \
    && pre-commit install-hooks \
    && sudo rm -rf /fake-repo \
    && sudo mkdir --mode=777 /opt/.pre-commit-cache-prebuilt \
    && mv ~/.cache/pre-commit/* /opt/.pre-commit-cache-prebuilt \
    && chmod a+rwx /opt/.pre-commit-cache-prebuilt \
;
