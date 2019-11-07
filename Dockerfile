FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN sed 's@archive.ubuntu.com@ftp.fau.de@' -i /etc/apt/sources.list
RUN apt-get update && apt-get -qy install jupyter python3-pip python-pip python3-numpy python-numpy
# Note: python-gmpy2 is recommended by python-mpmath.
RUN apt-get -qy install python-mpmath python3-mpmath python-gmpy2 python3-gmpy2 python-repoze.lru python3-repoze.lru
# we manually need to install numpy, due to this bug: https://github.com/embotech/ecos-python/issues/8
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

# Python2:
# For unknown reasons, cvxpy/setup.py installs the wrong version of numpy (too new for python2). As a workaround, we explicitly install numpy first.
#
# For development, uncomment the following line (newest versions) and comment out the one below (fixed versions):
# RUN pip install "numpy>=1.15" && pip install -r /requirements.txt
RUN pip install "$(cat /requirements-frozen-py2.txt | grep numpy)" && pip install -r /requirements-frozen-py2.txt
# Regenerate requirements-frozen-py2.txt by running `pip freeze -l > qronos/requirements-frozen-py2.txt` inside the container, *after it was built with the non-frozen requirements.txt*

WORKDIR /src
RUN adduser --gecos "" --disabled-password user
USER user
CMD jupyter-notebook --ip=0.0.0.0 --port=8888  2>&1 | sed s/0.0.0.0/localhost/
