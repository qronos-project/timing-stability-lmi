#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Computation of Common Quadratic Lyapunov Function (CQLF) based on Linear Matrix Inequalities (LMI)
as discussed in:
Gaukler et al. (2019/2020): Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing. Submitted for publication.
"""

import logging
from . import norms
from .generic_matrix import NumpyMatrix
try:
    import cvxpy as cp
except ImportError:
    cp = None
    logging.warning("cvxpy not found. Continuing with ugly workaround. The results will be horrible.")
    pass
import numpy as np
from .memoize_simple import matrix_memoize_simple

precond_cache_A = None
precond_cache_R_inv = None

@matrix_memoize_simple
def preconditioning(A):
    R_inv, _ = cqlf(A, [], rho=1, beta=0, R_inv=None)
    return R_inv

def cqlf(A, Delta_list, rho, beta, R_inv='auto'):
    """
    Return (P_sqrt_T, gamma) for common quadratic lyapunov function x^T P x
    where P = P_sqrt_T.T * P_sqrt_T
    and gamma is an indicator for the numerical accuracy of the solution (< 1e-7 typically means that it is useless)
    
    for the system x[k+1] = (A + Delta[i[k]]) x[k], i[k] uncertain in 0 ... len(Ad_list) - 1.
    
    (Note that this system description is slightly simplified, it assumes that the weighting factors rho, beta -- see below -- are chosen correctly.).
    
    
    A: n-by-n np.array or iv.matrix
    
    Delta: list of (n-by-n np.array or iv.matrix)
    
    rho: float: desired contraction bound for sqrt(V(A x)/V(x)). Must be larger than joint spectral radius of Ad_list.
        Increasing rho decreases the above bound, but typically improves robustness, i.e., robust_stability.iv_P_norm(Delta, P_sqrt_T) for any Delta
    
    beta: float: desired upper bound for P_norm(Delta[i])
        Similar to rho, increasing improves robustness.
    
    R_inv: n-by-n np.array or iv.matrix or None or 'auto':
        Optional preconditioning matrix R^-1.
        Improves numerical stability by transforming from A to R^-1 A R  before solving the LMI, and transforming back afterwards.
        Output is the same (but with lower numerical error if R is chosen wisely).
        'auto': automatically determine R_inv
    """
#    eig = abs(np.linalg.eig(Ad)[0])
#    print('eig', eig)
#    print('Ad', Ad)
#    
#    assert(all(abs(np.linalg.eig(Ad)[0])<1))
    if cp is None:
        logging.warning("cvxpy not found. Continuing with ugly workaround. The results will be horrible.")
        # just compute a quadratic lyapunov function for the first given matrix
        return (norms.approx_P_sqrt_T(A, (1-rho)/2), float('NaN'))

    # Load and convert parameters
    A = NumpyMatrix.convert(A)
    nd = len(A)
    if R_inv is None:
        R_inv = np.eye(nd)
        R = np.eye(nd)
    elif type(R_inv) == type('auto') and R_inv == 'auto':
        print("Solving CQLF for preconditioning (result may be cached)")
        R_inv = preconditioning(A)
        print("Preconditioning done. Now solving actual CQLF")
        return cqlf(A, Delta_list, rho, beta, R_inv=R_inv)
    else:
        R_inv = NumpyMatrix.convert(R_inv)
        R = np.linalg.inv(R_inv)
    Delta_list = [NumpyMatrix.convert(D) for D in Delta_list]
    
    # Apply transformation
    A = R_inv @ A @ R
    Delta_list = [R_inv @ Delta @ R for Delta in Delta_list]

    # Solve LMI
    P = cp.Variable((nd, nd), PSD=True)
    gamma = cp.Variable(1, nonneg=True)
    
    constraints = [P << np.eye(nd), # prevent infinite P
                   P >> gamma * np.eye(nd)] # prevent singular P
    constraints.append(A.T @ P @ A - (rho ** 2) * P << 0) # CQLF should be decreasing by desired factor rho
    for Delta in Delta_list:
        constraints.append(Delta.T * P * Delta - (beta ** 2) * P << 0)
    objective = cp.Maximize(gamma)
    prob = cp.Problem(objective, constraints)
    
    # CVXOPT with kktsolver='robust' turned out to be most robust open-source option available
    prob.solve(solver=cp.CVXOPT, verbose=True, kktsolver='robust')
    assert(prob.status == "optimal"), "no solution: problem status is " + prob.status
    
    
    P=P.value
#    print("(transformed) P", repr(P))
    gamma=gamma.value
    print("(transformed) robustness gamma:", gamma)
    try:
        P_sqrt_T = np.linalg.cholesky(P).T
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Cholesky decomposition of P failed. This should not happen.")
        # if this case happens, a valid (but useless) replacement result would be:
        # return (np.eye(nd), float('inf'))
    
    # Transform back
    # For simplicity, the following explanation assumes Delta=0, rho=1.
    # We have solved A_transformed.T P A_transformed < P for P,
    #    with A_transformed = R^-1 A R.
    # We are looking for P_orig, where A.T P_orig A < P_orig.
    # 
    # A_transformed.T P A_transformed < P
    # <=> (R^-1 A R).T P (R^-1 A R) < P
    # <=> R.T A.T R^-1.T P R^-1 A R < P.
    # Note that for invertible M: A < 0 <=> M.T A M < 0.
    # Apply this to the previous equation: R^-1.T * (...) * R^-1 < R^-1.T * (...) * R^-1
    # <=> A.T R^-1.T P R^-1 A < R^-1.T P R^-1
    # therefore, P_orig = R^-1.T P R^-1 = (P_sqrt_T R^-1).T  (P_sqrt_T R^-1)
    # and P_orig_sqrt_T = P_sqrt_T R^-1.
    # Particularly, if R approximates P_sqrt_T (where P = P_sqrt_T.T * P_sqrt_T),
    # then approximately P_orig = I, so P_orig is well-conditioned.
    P_sqrt_T = P_sqrt_T @ R_inv
#    print("P_sqrt (after inverse transform)", repr(P_sqrt_T))
    
    return (P_sqrt_T, gamma)
