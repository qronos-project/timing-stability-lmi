#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Memoization (caching) for functions using numpy.ndarray and mpmath.matrix.
"""

import numpy
import mpmath

def matrix_memoize_simple(func):
    """
    A simple decorator for caching the last argument.
    Only supports functions with one argument, which must be a mpmath.matrix or 2-dimensional numpy.ndarray.

    This is a quick replacement for repoze.lru.lru_cache(1)
    """
    def equal_matrix(a, b):
        """
        Test equality of two matrices of the type as given above.
        """
        if a is None or b is None:
            return False
        if type(a) != type(b):
            return False
        equal = (a == b)
        if isinstance(equal, numpy.ndarray):
            # numpy comparison returns a boolean matrix
            return equal.all()
        else:
            # mpmath comparison returns a boolean (scalar)
            return equal

    func._cache = (None, None)
    def wrapped_function(arg):
        assert isinstance(arg, (numpy.ndarray, mpmath.matrix, mpmath.iv.matrix)), "illegal type {}".format(type(arg))
        cached_arg, cached_result = func._cache
        if equal_matrix(cached_arg, arg):
            return cached_result
        else:
            result = func(arg)
            arg = 1 * arg # create copy of argument, because mp matrices are mutable
            func._cache = (arg, result)
            return result
    return wrapped_function
