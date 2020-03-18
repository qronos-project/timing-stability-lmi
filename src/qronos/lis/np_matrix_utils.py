"""
utility functions for working with numpy

NOTE: scheduled for deprecation, to be integrated in generic_matrix.py
"""

import numpy as np
def blkrepeat(M, repetitions):
    '''
    block-diagonal repetition of matrix M.
    M is converted to np.array.

    blkrepeat(M,3)= blkdiag(M,M,M) = [M,
                                        M,
                                           M]
    '''
    return np.kron(np.eye(repetitions), np.asarray(M))