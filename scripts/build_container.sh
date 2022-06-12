#! /bin/bash

SRC_DIR=${SRC_DIR:-`pwd`}
CONTAINER_NAME=${CONTAINER_NAME:-pymc}

# stop and remove previous instances of the pymc container to avoid naming conflicts
if [[ $(docker ps -aq -f name=${CONTAINER_NAME}) ]]; then
   echo "Shutting down and removing previous instance of ${CONTAINER_NAME} container..."
   docker rm -f ${CONTAINER_NAME}
fi

# note that all paths are relative to the build context, so . represents SRC_DIR to Docker
docker build \
    -t ${CONTAINER_NAME} \
    -f $SRC_DIR/scripts/Dockerfile $SRC_DIR
