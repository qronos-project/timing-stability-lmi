import numpy as np
from . import generic_matrix
def blkrepeat(M, repetitions):
    '''
    block-diagonal repetition of matrix M.
    M is converted to np.array.

    blkrepeat(M,3)= blkdiag(M,M,M) = [M,
                                        M,
                                           M]
    '''
    return np.kron(np.eye(repetitions), np.asarray(M))

def blockmatrix(M, blocklengths):
    '''
    build a square block-matrix like np.block, where
    0 is replaced by zeroes(...) of appropriate dimension.

    blocklengths is an array of the length of each block. The matrices on the diagonal must be square.

    Example:
    blockmatrix([[A, B], [0, C]], [a,b]) = np.block([[A, B], [zeroes(b,a), C]]).
    with matrices A,B,C of shape (a,a), (a,b), and (b,b) respectively.
    '''
    return generic_matrix.blockmatrix(M, blocklengths, np)