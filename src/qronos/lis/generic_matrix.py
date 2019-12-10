#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Utilities for handling both mpmath.iv and numpy matrices in a consistent fashion.
This is required since they behave differently and there is no implicit or easy conversion.

Note that by design, numpy uses '@' for matrix multiplication and '*' for elementwise.
However, mpmath.iv uses '*' for matrix multiplication. We patch it so that '@' also can be used for matrix multiplication.

DO NOT USE '*' for matrix*matrix as it can be ambiguous.

In the long-term future, this whole file should be made obsolete by any of these possible solutions:
- augmenting mpmath.iv with proper numpy conversion support
- augmenting mpmath.fp and replacing numpy with mpmath.fp.matrix
"""

import numpy as np
from mpmath import iv
import mpmath
from .iv_matrix_utils import iv_matrix_mid_to_numpy_ndarray
from .norms import iv_spectral_norm, iv_P_norm_expm, iv_P_norm, approx_P_norm, approx_P_norm_expm
import scipy.sparse.linalg as scipy_sparse_linalg
import scipy.linalg

from deprecation import deprecated
from abc import ABC, abstractmethod

# FIXME - monkey-patching so that we can use the "@" operator for mpmath.iv.matrix - this should be submitted as a patch in mpmath
iv.matrix.__matmul__ = iv.matrix.__mul__


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

@deprecated()
def convert(M, datatype):
    datatype = check_datatype(datatype)
    if datatype == iv:
        if isinstance(M, mpmath.matrix):
            return iv.matrix(M)
        else:
            return iv.matrix(M.tolist())
    elif datatype == np:
        if isinstance(M, (mpmath.matrix, iv.matrix)):
            return iv_matrix_mid_to_numpy_ndarray(M)
        else:
            return np.array(M)

@deprecated()
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

@deprecated()
def eye(n, datatype=None):
    '''
    n x n unity matrix from given module
    datatype: numpy (default) or mpmath.iv
    '''
    datatype = check_datatype(datatype)
    return datatype.eye(n)

@deprecated()
def expm(M, datatype=None):
    '''
    matrix exponential of M in given datatype

    datatype: numpy (default) or mpmath.iv
    '''
    datatype = check_datatype(datatype)
    if datatype == iv:
        return iv.expm(convert(M, datatype))
    else:
        return scipy.linalg.expm(convert(M, datatype))

@deprecated()
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

def approx_max_abs_eig(M) -> float:
    """
    Return max(abs(lambda_i)), where lambda_i is eigenvalue of M

    Result is only approximate due to finite numerical precision.
    """
    M = convert(M, np)
    return abs(scipy_sparse_linalg.eigs(M, k=1, which='LM')[0][0])

class AbstractMatrix(ABC):
    '''
    Abstract wrapper for numpy.ndarray and mpmath.iv.matrix
    to create matrices of the desired type with a consistent interface.

    For example,
    the following code works with both
    d = IntervalMatrix
    and
    d = NumpyMatrix
    :

    n = 2
    m1 = np.eye(n)
    m2 = iv.eye(n)
    d.eye(n) + d.zeros(n,n) + d.convert(m1) + d.convert(m2)

    For d=IntervalMatrix, all computations are valid interval bounds.
    '''

    @staticmethod
    def from_type(datatype):
        if datatype == iv:
            return IntervalMatrix
        elif datatype == np:
            return NumpyMatrix

    @staticmethod
    @abstractmethod
    def convert(M):
        '''
        convert matrix to current type
        '''

    @staticmethod
    def convert_scalar(x):
        '''
        convert matrix to scalar type
        '''
        return float(x)

    @staticmethod
    @abstractmethod
    def zeros(a, b):
        '''
        a x b matrix full of zeros
        '''

    @staticmethod
    @abstractmethod
    def eye(n):
        '''
        n x n unity matrix
        '''

    @staticmethod
    @abstractmethod
    def expm(M):
        '''
        matrix exponential
        '''

    @staticmethod
    @abstractmethod
    def spectral_norm(M):
        '''
        spectral_norm
        '''

    @classmethod
    def blockmatrix(cls, M, blocklengths):
        '''
        build a square block-matrix like np.block, where
        0 is replaced by zeroes(...) of appropriate dimension,
        1 is replaced by eye(...) of appropriate dimension

        blocklengths is an array of the length of each block. The matrices on the diagonal must be square.

        Example:
        NumpyMatrix.blockmatrix([[A, B], [0, C]], [a,b]) == np.block([[A, B], [zeroes(b,a), C]]).
        with matrices A,B,C of shape (a,a), (a,b), and (b,b) respectively.
        '''
        assert isinstance(M, list)
        assert len(M) == len(blocklengths)
        for i in M:
            assert isinstance(i, list), "M must be a list of lists of matrices"
            assert len(i) == len(blocklengths), "each row of M must have as many entries as there are blocks"
        output = cls.zeros(sum(blocklengths), sum(blocklengths))
        for i in range(len(blocklengths)):
            for j in range(len(blocklengths)):
                block_value = M[i][j]
                if type(block_value) == type(0) and block_value == 0:
                    # replace integer 0 with cls.zeros(...) of appropriate dimension
                    block_value = cls.zeros(blocklengths[i], blocklengths[j])
                if type(block_value) == type(1) and block_value == 1:
                    # replace integer 1 with cls.zeros(...) of appropriate dimension
                    assert blocklengths[i] == blocklengths[j], "1-blocks (identity matrix) are only allowed on the diagonal"
                    block_value = cls.eye(blocklengths[i])
                output[sum(blocklengths[0:i]):sum(blocklengths[0:i+1]), sum(blocklengths[0:j]):sum(blocklengths[0:j+1])] = cls.convert(block_value)
        return output

    @staticmethod
    @abstractmethod
    def P_norm(M, P_sqrt_T):
        """
        P_norm(M), defined as max_{x in R^n} sqrt(((M x).T P (M x)) / (x.T P x))

        with P_sqrt_T.T * P_sqrt_T = P,   where x.T P x typically is a Lyapunov function
        """

    @staticmethod
    @abstractmethod
    def P_norm_expm(P_sqrt_T, M1, A, M2, tau):
        """
        Upper bound on P_norm(M1 (expm(A*t) - I) M2, P_sqrt_T)  for |t| < tau

        @param P_sqrt_T: see P_norm()
        """

class NumpyMatrix(AbstractMatrix):
    @staticmethod
    def convert(M):
        if isinstance(M, (mpmath.matrix, iv.matrix)):
            return iv_matrix_mid_to_numpy_ndarray(M)
        else:
            return np.array(M)

    @staticmethod
    def zeros(a, b):
        return np.zeros((a, b))

    @staticmethod
    def eye(n):
        return np.eye(n)

    @staticmethod
    def expm(M):
        return scipy.linalg.expm(NumpyMatrix.convert(M))

    @staticmethod
    def spectral_norm(M):
        return scipy.linalg.norm(NumpyMatrix.convert(M), 2)

    @staticmethod
    def P_norm(M, P_sqrt_T):
        M = NumpyMatrix.convert(M)
        P_sqrt_T = NumpyMatrix.convert(P_sqrt_T)
        return approx_P_norm(M, P_sqrt_T)

    @staticmethod
    def P_norm_expm(P_sqrt_T, M1, A, M2, tau):
        P_sqrt_T = NumpyMatrix.convert(P_sqrt_T)
        M1 = NumpyMatrix.convert(M1)
        A = NumpyMatrix.convert(A)
        M2 = NumpyMatrix.convert(M2)
        tau = float(tau)
        return approx_P_norm_expm(P_sqrt_T, M1, A, M2, tau)

class IntervalMatrix(AbstractMatrix):
    @staticmethod
    def convert(M):
        if isinstance(M, mpmath.matrix):
            return iv.matrix(M)
        else:
            return iv.matrix(M.tolist())

    @staticmethod
    def convert_scalar(x):
        return iv.convert(x)

    @staticmethod
    def zeros(a, b):
        return iv.zeros(a, b)

    @staticmethod
    def eye(n):
        return iv.eye(n)

    @staticmethod
    def expm(M):
        return iv.expm(IntervalMatrix.convert(M))

    @staticmethod
    def spectral_norm(M):
        return iv_spectral_norm(M)

    @staticmethod
    def P_norm(M, P_sqrt_T):
        return iv_P_norm(IntervalMatrix.convert(M), IntervalMatrix.convert(P_sqrt_T))

    @staticmethod
    def P_norm_expm(P_sqrt_T, M1, A, M2, tau):
        return iv_P_norm_expm(IntervalMatrix.convert(P_sqrt_T), M1, A, M2, tau).b