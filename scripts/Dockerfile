FROM jupyter/minimal-notebook

MAINTAINER Austin Rochford <austin.rochford@gmail.com>

USER $NB_USER

COPY create_testenv.sh /tmp/create_testenv.sh
RUN /bin/bash /tmp/create_testenv.sh --global --no-setup

#  matplotlib nonsense
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
ENV MPLBACKEND=Agg
# Import matplotlib the first time to build the font cache.
RUN python -c "import matplotlib.pyplot"

ENV PYTHONPATH $PYTHONPATH:"$HOME"/pymc3
