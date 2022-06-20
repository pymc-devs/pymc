#! /bin/bash

COMMAND="${1:-jupyter}"
SRC_DIR=${SRC_DIR:-`pwd`}
CONTAINER_NAME=${CONTAINER_NAME:-pymc}
PORT=${PORT:-8888}

# stop and remove previous instances of the pymc container to avoid naming conflicts
if [[ $(docker ps -aq -f name=${CONTAINER_NAME}) ]]; then
   echo "Shutting down and removing previous instance of ${CONTAINER_NAME} container..."
   docker rm -f ${CONTAINER_NAME}
fi

# $COMMAND can be either `build` or `bash` or `jupyter`
if [[ $COMMAND = 'build' ]]; then
   docker build \
      -t ${CONTAINER_NAME} \
      -f $SRC_DIR/scripts/Dockerfile $SRC_DIR

elif [[ $COMMAND = 'bash' ]]; then
   docker run -it -v $SRC_DIR:/home/jovyan/work --rm  --name ${CONTAINER_NAME} ${CONTAINER_NAME} bash
else
   docker run -it -p $PORT:8888 -v $SRC_DIR:/home/jovyan/work --rm  --name ${CONTAINER_NAME} ${CONTAINER_NAME}
fi
