#! /bin/bash

PORT=${PORT:-8888}
SRC_DIR=${SRC_DIR:-`pwd`}
NOTEBOOK_DIR=${NOTEBOOK_DIR:-$SRC_DIR/notebooks}
TOKEN=$(openssl rand -hex 24)
CONTAINER_NAME=${CONTAINER_NAME:-pymc3}

# stop and remove previous instances of the pymc3 container to avoid naming conflicts
if [[ $(docker ps -aq -f name=${CONTAINER_NAME}) ]]; then
   echo "Shutting down and removing previous instance of ${CONTAINER_NAME} container..."
   docker rm -f ${CONTAINER_NAME}
fi

# note that all paths are relative to the build context, so . represents
# SRC_DIR to Docker
docker build \
    -t ${CONTAINER_NAME} \
    -f $SRC_DIR/scripts/Dockerfile \
    --build-arg SRC_DIR=. \
    $SRC_DIR

docker run -d \
    -p $PORT:8888 \
    -v $SRC_DIR:/home/jovyan/ \
    -v $NOTEBOOK_DIR:/home/jovyan/work/ \
    --name ${CONTAINER_NAME} ${CONTAINER_NAME} \
    start-notebook.sh --NotebookApp.token=${TOKEN}

if [[ $* != *--no-browser* ]]; then
  python -m webbrowser "http://localhost:${PORT}/?token=${TOKEN}"
fi
