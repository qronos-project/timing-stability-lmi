#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Utilities for handling both mpmath.iv and numpy matrices in a consistent fashion.
This is required since they behave differently and there is no implicit or easy conversion.
"""

import numpy as np
from mpmath import iv
import mpmath
from .iv_matrix_utils import iv_matrix_to_numpy_ndarray, iv_matrix_mid_as_mp

def check_datatype(datatype):
    """
    Check if the given datatype is valid.
    Valid values are `numpy` or `mpmath.iv` (just give the module as argument). Specify `None` for the default (numpy).
    
    Return the datatype, where `None` is replaced with the actual default value `mpmath.iv`.
    """
    assert datatype in [np, iv, None]
    if datatype == None:
        return np
    else:
        return datatype


# TODO: The return type of these functions is not strictly consistent for the 'numpy' case: Sometimes np.ndarray, sometimes np.matrix.
#       (See the note in convert() below.)

def convert(M, datatype):
    datatype = check_datatype(datatype)
    if datatype == iv:
        if isinstance(M, mpmath.matrix):
            return iv.matrix(M)
        else:
            return iv.matrix(M.tolist())
    elif datatype == np:
        if isinstance(M, (mpmath.matrix, iv.matrix)):
            return iv_matrix_to_numpy_ndarray(iv_matrix_mid_as_mp(M))
        else:
            # TODO For consistency, switch to np.ndarray, and change the whole code to use "@" instead of "mpmath *" or "numpy matmul". However, this is only possible after dropping Python 2 support, which will be done after the code in ../reachability has been migrated. Currently, that code still has Python2-only dependencies.
            return np.matrix(M)

def zeros(a, b, datatype=None):
    '''
    a x b matrix from given module
    datatype: numpy (default) or mpmath.iv
    '''
    datatype = check_datatype(datatype)
    if datatype == np:
        return np.zeros((a, b));
    elif datatype == iv:
        return iv.zeros(a, b);

def eye(n, datatype=None):
    '''
    n x n unity matrix from given module
    datatype: numpy (default) or mpmath.iv
    '''
    datatype = check_datatype(datatype)
    return datatype.eye(n)

def blockmatrix(M, blocklengths, datatype=None):
    '''
    build a square block-matrix like np.block, where
    0 is replaced by zeroes(...) of appropriate dimension,
    1 is replaced by eye(...) of appropriate dimension

    blocklengths is an array of the length of each block. The matrices on the diagonal must be square.
    
    datatype: numpy (default) or mpmath.iv

    Example:
    blockmatrix([[A, B], [0, C]], [a,b]) = np.block([[A, B], [zeroes(b,a), C]]).
    with matrices A,B,C of shape (a,a), (a,b), and (b,b) respectively.
    '''
    datatype = check_datatype(datatype)
    assert isinstance(M, list)
    assert len(M) == len(blocklengths)
    for i in M:
        assert isinstance(i, list), "M must be a list of lists of matrices"
        assert len(i) == len(blocklengths), "each row of M must have as many entries as there are blocks"
    output = zeros(sum(blocklengths), sum(blocklengths), datatype)
    for i in range(len(blocklengths)):
        for j in range(len(blocklengths)):
            block_value = M[i][j]
            if type(block_value) == type(0) and block_value == 0:
                # replace integer 0 with zeros(...) of appropriate dimension
                block_value = zeros(blocklengths[i], blocklengths[j], datatype)
            if type(block_value) == type(1) and block_value == 1:
                # replace integer 1 with zeros(...) of appropriate dimension
                assert blocklengths[i] == blocklengths[j], "1-blocks (identity matrix) are only allowed on the diagonal"
                block_value = eye(blocklengths[i], datatype)
            output[sum(blocklengths[0:i]):sum(blocklengths[0:i+1]), sum(blocklengths[0:j]):sum(blocklengths[0:j+1])] = convert(block_value, datatype)
    return output
