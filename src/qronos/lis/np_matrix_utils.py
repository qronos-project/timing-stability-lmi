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

def blockmatrix(M, blocklengths):
    '''
    build a square block-matrix like np.block, where
    0 is replaced by zeroes(...) of appropriate dimension.

    blocklengths is an array of the length of each block. The matrices on the diagonal must be square.

    Example:
    blockmatrix([[A, B], [0, C]], [a,b]) = np.block([[A, B], [zeroes(b,a), C]]).
    with matrices A,B,C of shape (a,a), (a,b), and (b,b) respectively.
    '''
    assert isinstance(M, list)
    assert len(M) == len(blocklengths)
    for i in M:
        assert isinstance(i, list), "M must be a list of lists of matrices"
        assert len(i) == len(blocklengths), "each row of M must have as many entries as there are blocks"

    output = np.zeros((sum(blocklengths), sum(blocklengths)));
    for i in range(len(blocklengths)):
        for j in range(len(blocklengths)):
            block_value = M[i][j]
            if type(block_value) == type(0) and block_value == 0:
                # replace integer 0 with np.zeros(...) of appropriate dimension
                block_value = np.zeros((blocklengths[i], blocklengths[j]))
            output[sum(blocklengths[0:i]):sum(blocklengths[0:i+1]), sum(blocklengths[0:j]):sum(blocklengths[0:j+1])] = block_value
    return output