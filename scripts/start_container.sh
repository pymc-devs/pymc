#! /bin/bash

PORT=${PORT:-8888}
SRC_DIR=${SRC_DIR:-`pwd`}
NOTEBOOK_DIR=${NOTEBOOK_DIR:-$SRC_DIR/notebooks}
TOKEN=$(openssl rand -hex 24)

# note that all paths are relative to the build context, so . represents
# SRC_DIR to Docker
docker build \
    -t pymc3 \
    -f $SRC_DIR/scripts/Dockerfile \
    --build-arg SRC_DIR=. \
    $SRC_DIR

docker run -d \
    -p $PORT:8888 \
    -v $SRC_DIR:/home/jovyan/ \
    -v $NOTEBOOK_DIR:/home/jovyan/work/ \
    --name pymc3 pymc3 \
    start-notebook.sh --NotebookApp.token=${TOKEN}

if [[ $* != *--no-browser* ]]; then
  python -m webbrowser "http://localhost:${PORT}/?token=${TOKEN}"
fi
