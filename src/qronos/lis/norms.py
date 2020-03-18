#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Robust stability of general discrete-time systems using the P-norm

This is a low-level interface. Please use generic_matrix if possible.
"""

import scipy.linalg
import numpy
import mpmath as mp
from mpmath import iv
import math
import random
from .memoize_simple import matrix_memoize_simple

# FIXME - monkey-patching so that we can use the "@" operator for mpmath.iv.matrix - this should be submitted as a patch in mpmath
iv.matrix.__matmul__ = iv.matrix.__mul__

from .iv_matrix_utils import iv_matrix_mid_to_numpy_ndarray, numpy_ndarray_to_mp_matrix, iv_matrix_to_numpy_ndarray, iv_matrix_mid_as_mp, iv_matrix_inv



def iv_spectral_norm_rough(M):
    """
    Fast but rough interval bound of spectral norm.

    0 <= spectral_norm(M) <= sqrt(sum of all m[i,k]^2)
    """
    norm = iv.norm(M, 2)
    return iv.mpf([0, norm.b])


def iv_spectral_norm(M):
    """
    Good interval bound of spectral norm of a (interval) matrix

    Theorem 3.2 from

    Siegfried M. Rump. “Verified bounds for singular values, in particular for the
    spectral norm of a matrix and its inverse”. In: BIT Numerical Mathematics 51.2
    (Nov. 2010), pp. 367–384. DOI : 10.1007/s10543-010-0294-0.
    """
    M = iv.matrix(M)
    # imprecise SVD of M (no requirement on precision)
    # _, _, V_T = mp.svd(iv_matrix_mid_to_mp(M)) # <-- configurable precision
    _, _, V_T = scipy.linalg.svd(iv_matrix_mid_to_numpy_ndarray(M)) # <-- faster
    # in [Rump 2010], SVD is defined as M = U @ S @ V.T,
    # in mpmath it is M = U @ S @ V (not transposed) for U,S,V=svd(M)
    # in scipy it is effectively the same as in mpmath, M = U @ S @ Vh for U,S,Vh = svd(M)
    V = numpy_ndarray_to_mp_matrix(V_T).T
    # now, everything is named as in [Rump2010], except that here, A is called M.
    # all following computations are interval bounds
    V = iv.matrix(V)
    B = M @ V
    BTB = B.T @ B
    # split BTB such that DE = diagonal D + rest E
    D = BTB @ 0
    E = BTB @ 0
    for i in range(BTB.rows):
        for j in range(BTB.cols):
            if i == j:
                D[i,j] = BTB[i,j]
            else:
                E[i,j] = BTB[i,j]
    # upper bound of spectral norm of I - V.T @ V
    alpha = iv_spectral_norm_rough(iv.eye(len(M)) - V.T @ V)
    # upper bound of spectral norm of E
    epsilon = iv_spectral_norm_rough(E)
    # maximum of D[i,i]  (which are always >= 0)
    d_max = iv.norm(D, mp.inf)
    if alpha.b >= 1:
        # this shouldn't happen - even an imprecise SVD will roughly have V.T @ V = I.
        raise scipy.linalg.LinAlgError("Something's numerically wrong - the singular vectors are far from orthonormal")
        # should this ever happen in reality, a valid return value would be:
        # return iv.mpf([0, mp.inf])
    try:
        lower_bound = iv.sqrt((d_max - epsilon) / (1 + alpha)).a
    except mp.libmp.libmpf.ComplexResult:
        lower_bound=0;
    # note that d_max, epsilon,alpha are intervals, so everything in the following computation is interval arithmetic
    return iv.mpf([lower_bound, iv.sqrt((d_max + epsilon) / (1 - alpha)).b])





# BUG in mpmath: None == mp.zeros(1) errors



def iv_P_norm(M, P_sqrt_T):
    """
    interval bound on P_norm(M), defined as max_{x in R^n} sqrt(((M x).T P (M x)) / (x.T P x))

    with P_sqrt_T.T @ P_sqrt_T = P,   where x.T P x typically is a Lyapunov function
    """
    P_sqrt_T = iv.matrix(P_sqrt_T)
    M = iv.matrix(M)
    # FIXME: To a workaround a bug in mpmath we currently add a zero interval to the matrix
    # BUG in mpmath: iv.matrix(mp.eye(2)) * (iv.ones(2) + iv.mpf([1, 2]))   errors
    #
    # P_norm(M) = spectral norm (maximum singular value) of P_sqrt_T A P_sqrt_T**(-1)
    return iv_spectral_norm((iv.matrix(mp.zeros(len(M))) + iv.matrix(P_sqrt_T)) * M * iv_matrix_inv(iv.matrix(P_sqrt_T)))


def approx_P_norm(M, P_sqrt_T):
    """
    approximation of P_norm(M)

    @see iv_P_norm()
    """
    M = iv_matrix_mid_to_numpy_ndarray(M)
    P_sqrt_T = iv_matrix_mid_to_numpy_ndarray(P_sqrt_T)
    return scipy.linalg.norm(numpy.matmul(P_sqrt_T, numpy.matmul(M, np_matrix_inv(P_sqrt_T))), 2)


def approx_P_sqrt_T(A, tolerance=1e-4):
    """
    Compute P_sqrt_T such that approximately iv_P_norm(M, P_sqrt_T) < max(abs(eig(A)))

    Will raise an AssertionError if approximately max(abs(eig(A))) >= 1.

    @param tolerance: relative factor for numerical tolerance. Decrease with caution. Increasing the tolerance will increase robustness.
    """
    A = iv_matrix_mid_as_mp(A)
    eigv_A, _= mp.eig(A)
    max_eigv_A = max([abs(i) for i in eigv_A])
    assert max_eigv_A < (1 - tolerance)
    A = iv_matrix_to_numpy_ndarray(A)
    # eigenvalue scaling of A (without this, we could only guarantee iv_P_norm(M, P_sqrt_T) < 1)
    A = A / float(max_eigv_A) * (1 - tolerance)
    Q = numpy.eye(len(A))
    # scipy solves AXA^H - X + Q = 0 for X.
    # We need the solution of A.T P A - P = -Q, so A must be transposed!
    P = scipy.linalg.solve_discrete_lyapunov(a=A.T, q=Q)
    # check validity of the solution
    # Note that there is no guaranteed tolerance bound. For practical reasons we choose the given tolerance.
    assert numpy.allclose(numpy.matmul(numpy.matmul(A.T, P), A) - P, -Q, atol=tolerance), "Discrete lyapunov solution inaccurate. Try increasing the tolerance."
    # numpy.linalg.cholesky returns lower-triangular L such that L*L.T = P
    # Here, L.T is called P_sqrt_T.
    P_sqrt_T = numpy.linalg.cholesky(P).T
    return numpy_ndarray_to_mp_matrix(P_sqrt_T)

IV_NORM_EVAL_ORDER = 10

@matrix_memoize_simple
def _iv_matrix_powers(A):
    """
    return the first IV_NORM_EVAL_ORDER+1 powers of A:
    [I, A, A**2, ..., A**(IV_NORM_EVAL_ORDER)]
    """
    assert isinstance(A, iv.matrix)
    A = iv.matrix(A) + mp.mpi(0,0) # workaround bug
    A_pow = [iv.eye(len(A))]
    for i in range(IV_NORM_EVAL_ORDER+1):
        A_pow.append(A @ A_pow[-1])
    return A_pow

def iv_P_norm_expm(P_sqrt_T, M1, A, M2, tau):
    """
    Bound on P_norm( M1 (expm(A*t) - I) M2)  for |t| < tau

    using the theorem in arXiv:1911.02537, section "Norm bounding of summands"

    @param P_sqrt_T: see iv_P_norm()
    """
    P_sqrt_T = iv.matrix(P_sqrt_T)
    M1 = iv.matrix(M1)
    A = iv.matrix(A)
    M2 = iv.matrix(M2)
    # coerce tau to maximum
    tau = abs(iv.mpf(tau)).b
    # P-norms
    M1_p = iv_P_norm(M=M1, P_sqrt_T=P_sqrt_T)
    M2_p = iv_P_norm(M=M2, P_sqrt_T=P_sqrt_T)
    A_p = iv_P_norm(M=A, P_sqrt_T=P_sqrt_T)
    # A_pow[i] = A ** i
    A_pow = _iv_matrix_powers(A)
    # Work around bug in mpmath, see comment in iv_P_norm()
    zero = iv.matrix(mp.zeros(len(A)))
    M1 = zero + M1
    # terms from [arXiv:1911.02537]
    M1_Ai_M2_p = lambda i: iv_P_norm(M=M1 @ A_pow[i] @ M2, P_sqrt_T=P_sqrt_T)
    gamma = lambda i: 1 / math.factorial(i) * (M1_Ai_M2_p(i) - M1_p * A_p ** i * M2_p)
    max_norm = sum([gamma(i) * (tau ** i) for i in range(1, IV_NORM_EVAL_ORDER + 1)]) + M1_p * M2_p * (iv.exp(A_p * tau) - 1)
    # the lower bound is always 0 (for t=0)
    return mp.mpi([0, max_norm.b])


@matrix_memoize_simple
def np_matrix_inv(M):
    """
    interval matrix inverse, with caching
    """
    assert isinstance(M, numpy.ndarray)
    return scipy.linalg.inv(M)

def approx_P_norm_expm(P_sqrt_T, M1, A, M2, tau):
    """
    approximate max ( P_norm( M1 (expm(A*t) - I) M2 )  for |t| < tau )

    @param P_sqrt_T: see iv_P_norm()
    """
    P_sqrt_T = iv_matrix_mid_to_numpy_ndarray(P_sqrt_T)
    M1 = iv_matrix_mid_to_numpy_ndarray(M1)
    A = iv_matrix_mid_to_numpy_ndarray(A)
    M2 = iv_matrix_mid_to_numpy_ndarray(M2)
    # coerce tau to maximum
    tau = float(mp.mpf(abs(iv.mpf(tau)).b))
    max_norm = 0
    for t in numpy.linspace(-tau, tau, 100):
        matrix = numpy.matmul(M1, numpy.matmul(scipy.linalg.expm(A*t) - numpy.eye(len(A)), M2))
        max_norm = max(max_norm, approx_P_norm(M=matrix, P_sqrt_T=P_sqrt_T))
    return max_norm

if __name__ == "__main__":
    random.seed(1234565567)
    print("Example: random matrix")
    for i in range(1):
        c = 2
        A = mp.randmatrix(20) - 0.5
        eigv_A, _= mp.eig(iv_matrix_mid_as_mp(A))
        A = 0.5 * A / max([abs(i) for i in eigv_A])
        

        eigv_A, _= mp.eig(iv_matrix_mid_as_mp(A))
        #print('eigenvalues(A) = ', eigv_A)
        print('spectral radius(A) = ', max([abs(i) for i in eigv_A]))
        print('interval spectral_norm(A) = ', iv_spectral_norm(A))
        P_sqrt_T = approx_P_sqrt_T(A)
        print('interval P_norm(A) = ',iv_P_norm(A, P_sqrt_T))
        print('interval P_norm(...expm(...)) = ', iv_P_norm_expm(P_sqrt_T, M1=mp.eye(len(A)), A=A, M2=mp.eye(len(A)), tau=0.01))
        print('sampled P_norm(...expm(...)) = ', approx_P_norm_expm(P_sqrt_T, M1=mp.eye(len(A)), A=A, M2=mp.eye(len(A)), tau=0.01))
        print('')

