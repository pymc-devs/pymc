#! /bin/bash

PORT=${PORT:-8888}
SRC_DIR=${SRC_DIR:-`pwd`}
CONTAINER_NAME=${CONTAINER_NAME:-pymc}

# stop and remove previous instances of the pymc container to avoid naming conflicts
if [[ $(docker ps -aq -f name=${CONTAINER_NAME}) ]]; then
   echo "Shutting down and removing previous instance of ${CONTAINER_NAME} container..."
   docker rm -f ${CONTAINER_NAME}
fi

docker run -it -p $PORT:8888 -v $SRC_DIR:/home/jovyan/workspace --rm  --name ${CONTAINER_NAME} ${CONTAINER_NAME}
