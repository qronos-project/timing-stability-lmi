FROM localhost/hyst_baseimage
# to remove reachability-analysis-related dependencies, replace the above line with:
# FROM ubuntu:18.04

# Workaround to run reachability analysis on older CPUs: (SpaceEx ships with a buggy GMP lib that causes SIGILL errors on CPUs with certain subsets of AVX. Only seems to affect CPUs made before 2015.)
RUN cp -sf /usr/lib/x86_64-linux-gnu/libgmp* /tools/spaceex/spaceex_exe/lib/

ENV DEBIAN_FRONTEND=noninteractive
RUN sed 's@archive.ubuntu.com@ftp.fau.de@' -i /etc/apt/sources.list
RUN apt-get update && apt-get -qy install python3-pip python3-numpy python3-deprecation
# python3-mpmath omitted, see below for custom mpmath version
# Note: python3-gmpy2 is recommended by python3-mpmath.
RUN apt-get -qy install python3-gmpy2 python3-repoze.lru
# and libblas to avoid https://github.com/bodono/scs-python/issues/5
RUN apt-get -qy install libatlas-base-dev
# build requirements for slycot and formerly python-control (and cvxopt?)
RUN apt-get -qy install gfortran libopenblas-dev liblapack-dev cmake

# Install Python dependencies which are not available as Ubuntu package:
ADD src/qronos/requirements* /

# Custom mpmath version until all changes are merged and released upstream
RUN apt-get -qy remove python3-mpmath
ADD mpmath /mpmath
RUN rm -rf /mpmath/.git
RUN SETUPTOOLS_SCM_PRETEND_VERSION=999.999.42 pip3 install -e /mpmath

# Newer pip's resolver is less broken than the older one.
RUN pip3 install pip==20.3.1
RUN pip3 check
# For reproducibility, fixed versions are installed (generated using pip3 freeze).
# For development, uncomment the following 'non-frozen' line (newest versions) and comment out the 'frozen' one below (fixed versions):
RUN pip3 install -r /requirements.txt # nonfrozen, will install latest available versions
# RUN pip3 install -r /requirements-frozen-py3.txt # frozen, fixed versions
# Regenerate requirements-frozen-py3.txt by running `pip3 freeze -l | grep -v -- ^-e > qronos/requirements-frozen-py3.txt` inside the container, *after it was built with the non-frozen requirements.txt*

WORKDIR /src
# configurable user and group IDs for running as non-root. 1000 is the default value. To override, use "docker --build-arg", or "CONTAINER_UID=1234 docker-compose ...", or the provided run_*.sh shell scripts.
ARG CONTAINER_UID=1000
ARG CONTAINER_GID=1000
RUN echo "Using CONTAINER_UID=$CONTAINER_UID and CONTAINER_GID=$CONTAINER_GID"
RUN addgroup user --gid $CONTAINER_GID
RUN adduser --gecos "" --disabled-password --uid $CONTAINER_UID --gid $CONTAINER_GID user
RUN chown $CONTAINER_UID -R /hyst
USER user
CMD bash
