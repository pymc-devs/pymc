FROM jupyter/scipy-notebook

MAINTAINER Austin Rochford <austin.rochford@gmail.com>

USER $NB_USER

RUN wget -O /tmp/requirements.txt https://raw.githubusercontent.com/pymc-devs/pymc3/master/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN conda install --quiet --yes icu=56.1

ENV PYTHONPATH $PYTHONPATH:/home/jovyan/pymc3
