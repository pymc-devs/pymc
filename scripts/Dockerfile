FROM continuumio/miniconda3

MAINTAINER Austin Rochford <austin.rochford@gmail.com>

USER $NB_USER

RUN wget -O /tmp/create_testenv.sh https://raw.githubusercontent.com/pymc-devs/pymc3/master/scripts/create_testenv.sh
RUN sh /tmp/create_testenv.sh --global

ENV PYTHONPATH $PYTHONPATH:$HOME/pymc3
