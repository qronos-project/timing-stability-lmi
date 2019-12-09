FROM localhost/hyst_baseimage
# to remove reachability-analysis-related dependencies, replace the above line with:
# FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN sed 's@archive.ubuntu.com@ftp.fau.de@' -i /etc/apt/sources.list
RUN apt-get update && apt-get -qy install jupyter python3-pip python3-numpy
# Note: python3-gmpy2 is recommended by python3-mpmath.
RUN apt-get -qy install python3-mpmath python3-gmpy2 python3-repoze.lru
# and libblas to avoid https://github.com/bodono/scs-python/issues/5
RUN apt-get -qy install libatlas-base-dev
# TeX support in Jupyter Notebooks:
RUN apt-get -qy install texlive texlive-xetex
# build requirements for slycot and python-control (and cvxopt?)
RUN apt-get -qy install gfortran libopenblas-dev liblapack-dev cmake

# The following lines are commented out because python-control is currently not used by this code:
#RUN pip3 install numpy
#RUN pip3 install scikit-build # WORKAROUND BUG https://github.com/python-control/Slycot/issues/3
#RUN ln -s /usr/local/lib/python3.6/dist-packages/numpy /usr/lib/python3/dist-packages/numpy  # WORKAROUND BUG
#RUN pip3 install slycot

# Install Python dependencies which are not available as Ubuntu package:
ADD src/qronos/requirements* /
# For reproducibility, fixed versions are installed (generated using {pip,pip3} freeze).
# For development, uncomment the following line (newest versions) and comment out the one below (fixed versions):
# RUN pip3 install -r /requirements.txt
RUN pip3 install -r /requirements-frozen-py3.txt
# Regenerate requirements-frozen-py3.txt by running `pip3 freeze -l > qronos/requirements-frozen-py3.txt` inside the container, *after it was built with the non-frozen requirements.txt*

WORKDIR /src
RUN adduser --gecos "" --disabled-password user
USER user
CMD jupyter-notebook --ip=0.0.0.0 --port=8888  2>&1 | sed s/0.0.0.0/localhost/
