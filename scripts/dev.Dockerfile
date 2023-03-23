FROM ghcr.io/mamba-org/micromamba-devcontainer:git-e04d158

COPY --chown=${MAMBA_USER}:${MAMBA_USER} conda-envs/environment-dev.yml /tmp/environment-dev.yml
RUN : \
    && micromamba install --yes --name base --file /tmp/environment-dev.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment-dev.yml \
    && sudo chmod -R a+rwx /opt/conda \
;

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ENV PRE_COMMIT_HOME=/opt/.pre-commit-cache-prebuilt
COPY --chown=${MAMBA_USER}:${MAMBA_USER} .pre-commit-config.yaml /fake-repo/.pre-commit-config.yaml
RUN : \
    && sudo mkdir --mode=777 /opt/.pre-commit-cache-prebuilt \
    && cd /fake-repo \
    && git init \
    && pre-commit install-hooks \
    && sudo rm -rf /fake-repo \
    && sudo chmod -R a+rwx /opt/.pre-commit-cache-prebuilt \
;
