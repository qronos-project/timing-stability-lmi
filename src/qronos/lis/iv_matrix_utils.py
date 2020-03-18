#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for working with mpmath matrices.

NOTE: This is a low-level interface --- use generic_matrix.IntervalMatrix where possible!
"""

import mpmath as mp
from mpmath import iv
import numpy
from .memoize_simple import matrix_memoize_simple

def matrix_abs_max(M):
    """ entrywise maximum of absolute value of interval matrix """
    return M.apply(lambda x: abs(x).b)

def iv_matrix_mid_as_mp(M):
    """
    entrywise midpoint of interval matrix,
    as (non-interval) mpmath matrix.

    If the input is a mpmath matrix, it is returned unchanged.
    If the input is a 2-dimensional numpy.ndarray, it is converted to mpmath.
    """
    if isinstance(M, numpy.ndarray):
        return mp.matrix(M)
    Y=mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            Y[i,j] = mp.mpi(M[i,j]).mid
    return Y


def iv_matrix_to_mp(M):
    """
    convert zero-width interval matrix to mp matrix

    (will raise an Exception if the interval of any element is not zero-width)
    """
    Y=mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        for j in range(M.cols):
            Y[i,j] = mp.convert(M[i,j])
    return Y


def iv_matrix_to_numpy_ndarray(M):
    """
    convert zero-width interval matrix to numpy ndarray

    (will raise an Exception if the interval of any element is not zero-width)
    """
    return numpy.array(iv_matrix_to_mp(M).tolist(), dtype=float)

def iv_matrix_mid_to_numpy_ndarray(M):
    """
    convert midpoint of interval matrix to numpy ndarray
    """
    if isinstance(M, numpy.ndarray):
        return M
    Y=numpy.zeros((M.rows, M.cols))
    for i in range(M.rows):
        for j in range(M.cols):
            Y[i,j] = float(mp.mpf(mp.mpi(M[i,j]).mid))
    return Y

def numpy_ndarray_to_mp_matrix(M):
    """
    convert numpy ndarray to zero-width interval matrix
    """
    return mp.matrix(M.tolist())

#@repoze.lru.lru_cache(64) # doesn't work with mutable values
#@mp.memoize # We can't use mpmath.memoize because it doesn't work properly (really slow, don't know why). Also it fails on python3.
@matrix_memoize_simple
def iv_matrix_inv(M):
    """
    interval matrix inverse, with caching

    NOTE: If you change mpmath's resolution after the first call to this function,
    you may get cached old output with the old resolution.
    """
    assert isinstance(M, (mp.matrix, iv.matrix))
    M = iv.matrix(M)
    return M**(-1)