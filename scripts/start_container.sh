#! /bin/bash

PORT=${PORT:-8888}
SRC_DIR=${SRC_DIR:-`pwd`}
NOTEBOOK_DIR=${NOTEBOOK_DIR:-$SRC_DIR/notebooks}

docker build -t pymc3 $SRC_DIR/scripts
docker run -d \
    -p $PORT:8888 \
    -v $SRC_DIR:/home/jovyan/pymc3 \
    -v $NOTEBOOK_DIR:/home/jovyan/work/ \
    --name pymc3 pymc3 \
    start-notebook.sh --NotebookApp.token=''
