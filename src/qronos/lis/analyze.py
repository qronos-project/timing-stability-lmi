#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Stability analysis based on Linear Impulsive Systems and Linear Matrix Inequalities.
See Gaukler et al. (2019/2020): Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing. IFAC 2020 / https://arxiv.org/abs/1911.02537
"""
from .. import examples
from .lis import LISControlLoop
import mpmath as mp
import numpy as np
from mpmath import iv
from .iv_matrix_utils import numpy_ndarray_to_mp_matrix
from .generic_matrix import check_datatype, approx_max_abs_eig, AbstractMatrix, NumpyMatrix
from . import lmi_cqlf
import itertools

import sys
from datetime import datetime

def analyze(s, datatype=None):
    """
    Run CQLF-based analysis (Gaukler et al. 2019/2020, see https://arxiv.org/abs/1911.02537)

    @param LISControlLoop s system to analyze
    @param datatype: Datatype for analysis:
        either mpmath.iv (exact analysis using interval arithmetic)
        or numpy (approximate analysis)
    @return dictionary of:
        P_sqrt_T: CQLF matrix as defined in generic_matrix.P_norm()
        rho_approx: fast approximation of rho
        time_approx: time for computing only rho_approx
        rho: proven value of rho (only if dtype==mpmath.iv)
        time: time for computing rho and rho_approx
    """
    datatype = check_datatype(datatype or iv)
    d = AbstractMatrix.from_type(datatype)
    print("")
    print("")
    print("Analyzing system: {}".format(s))
    print("Analysis is exact (interval arithmetic): {}".format(datatype == iv))
    print("Please note that some of the above parameters, eg. spaceex_... are not used in this LMI-based analysis, they are only present for reachability analysis.")
    # TODO: move the spaceex_... parameters to some sub-structure, e.g. sys.param.reach.XXX
    time_start = datetime.now()
    l = LISControlLoop(s, datatype)
    result = {'time_approx': None, 'time': None, 'n': l.n}
    A = l.Ak_nominal
    rho_ideal = approx_max_abs_eig(A)
    print('spectral radius(A) = ', rho_ideal)
    print('spectral_norm(A) = ', d.spectral_norm(A))

    # "Pressure factor" to reduce P_norm(A) at the cost of numerical robustness and the other goals (see other factors)
    tighteningRho = 0.8 # 0 ... 1, typically 0.8
    precondition=True # use preconditioning?

    rho = 1 - (1 - rho_ideal) * tighteningRho # target rho to be shown by CQLF

    # "Pressure factor" to reduce P_norm(Delta_A_...) at the cost of numerical robustness
    # Usually, increasing helps up to some limit, at which the numerical robustness is so low that everything "blows up"
    # 0 ... infinity, should typically be at least (m+p+m*p)/(1-rho), with rho=max(abs(eigenvalues(A)))
    # This is an initial guess which will later be used for a heuristic search.
    # 0.25 is a factor approximating the "sparsity", i.e., roughly the ratio of Delta_A_... terms which are zero or smaller than the theoretical bound.
    beta = 0.25 * (s.m + s.p + s.m * s.p) / (1 - rho)
    print('beta=', beta)

    Delta_list = []
    import itertools
    dtu_dty_combinations = list(itertools.product([l.sys.delta_t_u_min*0, l.sys.delta_t_u_min, l.sys.delta_t_u_max], [l.sys.delta_t_y_min*0, l.sys.delta_t_y_min, l.sys.delta_t_y_max]))
    for (dtu, dty) in dtu_dty_combinations:
        if all(dtu==0) and all(dty==0):
            continue
        Delta_list.append(l.Ak_delta_to_nominal(dtu, dty, datatype=np))


    print("")
    print("Solving CQLF LMI")
    P_list = []
    delta = 2
    num_iterations = 3
    for iterations in range(num_iterations):
        print("Iteration {} of {}: ".format(iterations, num_iterations) + "delta={}, ".format(delta) + "beta={}".format(beta))
        P_sqrt_T, gamma = lmi_cqlf.cqlf(A=l.Ak_nominal, Delta_list=Delta_list, rho=rho, beta=1/beta, R_inv='auto' if precondition else None)
        print("Solving CQLF for beta=", beta, " resulted in robustness gamma=", gamma)
        P_sqrt_T = numpy_ndarray_to_mp_matrix(P_sqrt_T)
        print("first approximation:")
        (rho_approx, pnorm_approx) = l.rho_total(P_sqrt_T, datatype=np, verbose=True)
        print("approx rho=", rho_approx)
        P_list.append((beta, rho_approx, gamma, P_sqrt_T))

        if pnorm_approx > 1:
            print("finding CQLF failed, retrying with lower beta. Probably this system is impossible to verify.")
            beta /= pnorm_approx
            beta = max(beta, 1) # must be at least 1
            continue
        # For improvement at the cost of more iterations, this heuristic can be replaced e.g. with a line search
        if gamma < 1e-5:
            # Reduce pressure if numerical robustness of the solution is bad
            delta *= 0.45
        # Guess beta such that (for delta=1) roughly rho_approx = 1, assuming that beta linearly scales the timing-dependent part of rho_approx.
        beta *= delta * (rho_approx - pnorm_approx) / (1 - pnorm_approx)
    print("Results:")
    for i in P_list:
        (beta, rho_approx, gamma, P_sqrt_T) = i
        print('rho (approx.):', rho_approx, ' for  beta:', beta, ' with robustness gamma:', gamma)
        p_sqrt_eigv = np.diag(NumpyMatrix.convert(P_sqrt_T))
        print('eigenvalue spread of P_sqrt_T: (loosely indicates higher excentricity):', max(p_sqrt_eigv)/min(p_sqrt_eigv))
    # Choose best result
    (beta, rho_approx, gamma, P_sqrt_T) = min(P_list, key=lambda x: x[1])
    result['P_sqrt_T'] = P_sqrt_T
    print("Best rho (approx): ", rho_approx)
    result['rho_approx'] = rho_approx
    result['time_approx'] = (datetime.now() - time_start).total_seconds()
    if rho_approx > 1:
        print("Cannot show stability. Returning approximate result for instability, skipping verification.")
        return result
    if datatype == np:
        print("Only inexact computation was requested. Not performing exact analysis")
        return result
    assert datatype == iv
    print("Exact results:")
    (rho_total, _) = l.rho_total(P_sqrt_T)
    print('Overall rho: ', rho_total)
    # The following is a simple but slightly pessimistic variant for rounding up from mpmath.mpi to float
    result['rho'] = float(mp.mpf(rho_total.b)) * 1.0000001
    result['time'] = (datetime.now() - time_start).total_seconds()


    return result

def analyze_cqlf_timing_range(system, datatype=None, P_sqrt_T=None, num_points=150):
    """
    How does rho scale if the timing is changed but P stays constant?

    @param DigitalControlLoop system
    @param datatype: see analyze() (optional)
    @param P_sqrt_T: CQLF-matrix determined by analyze() (optional)
    @param num_points: discretization steps (higher value means less pessimism)

    returns (scaling_list, rho_list, offset, slope),
         where scaling_list[i] is the factor by which the timing was scaled (0: perfect timing, 2: twice as much as given in the system model, ...)
         and rho_list[i] the corresponding rho.

         scaling_list[0]==0, so rho_list[0] is the "nominal" rho (for perfect timing).

         offset and slope are determined such that
         rho(scale) <= offset + slope * scale,
         even for scale values inbetween the raster points of scaling_list,
         using the fact that rho(scale1) <= rho(scale2) if 0 <= scale1 <= scale2.
         This requires that delta_t=0 is included in the interval of allowed delta_t.
    """
    datatype = datatype or np
    if P_sqrt_T is None:
        P_sqrt_T = analyze(system, np)['P_sqrt_T']
    l = LISControlLoop(system, datatype)
    assert all(system.delta_t_u_min <= 0) and all(system.delta_t_u_max >= 0) \
        and all(system.delta_t_y_min <= 0) and all(system.delta_t_y_max >= 0), \
        "delta t=0 must be included in the timing intervals"
    max_abs_timing = max(abs(np.hstack((system.delta_t_u_max, system.delta_t_u_min, system.delta_t_y_max, system.delta_t_y_min))))
    print(max_abs_timing)
    rho_list = np.zeros(num_points)
    scaling_list = np.zeros(num_points)
    i = 0
    for scaling in np.linspace(0, system.T/2/max_abs_timing, num=num_points):
        rho_total, rho_nominal=l.rho_total(P_sqrt_T=P_sqrt_T, scale_delta_t=scaling, datatype=datatype)
        print(scaling, rho_total, (rho_total-rho_nominal)/scaling if scaling != 0 else float('NaN'))
        scaling_list[i] = scaling
        rho_list[i] = rho_total
        i = i + 1
    offset = rho_list[1]
    slope = np.max(np.diff(rho_list)) / np.min(np.diff(scaling_list))
    print(f"rho(scale) <= offset + slope * scale, where offset={offset}, slope={slope}")
    return (rho_list, scaling_list, offset, slope)

def analyze_cqlf_skip(system, P_sqrt_T=None):
    """
    rho=P_norm(Ak) for given P, ideal timing, but skipped events.

    Returns the individual rho for every combination of skipping or not skipping events.
    Return value is a dictionary: {(skip_ctrl, skip_u, skip_y): rho, ...}
    """
    lis = LISControlLoop(system, np)
    if P_sqrt_T is None:
        P_sqrt_T = analyze(system)['P_sqrt_T']
    def boolean_vectors(length):
        return itertools.product([True, False], repeat=length)

    rho={}
    for skip_ctrl in [True, False]:
        for skip_u in boolean_vectors(lis.sys.m):
            for skip_y in boolean_vectors(lis.sys.p):
                delta = lis.Ak_delta_to_nominal(skip_ctrl=skip_ctrl, skip_u=skip_u, skip_y=skip_y);
                pnorm_a_new = NumpyMatrix.P_norm(delta + lis.Ak_nominal, P_sqrt_T)
                rho[(skip_ctrl, skip_u, skip_y)] = pnorm_a_new
    return rho

def analyze_cqlf_skip_condition(system, P_sqrt_T=None, condition=None):
    """
    maximum rho=P_norm(Ak) for given P, ideal timing, but skipped events.

    Which events may be skipped is restricted by a condition:

    condition(skip_ctrl: bool, skip_u: iterable[bool], skip_y: iterable[bool]):
        boolean function, returns True if this combination of skips may occur.
        (skip_... has the same meaning as in lis.Ak_delta_to_nominal).
        example: condition = lambda skip_ctrl, skip_u, skip_y: skip_ctrl and not any(skip_u) and all(skip_y)
    """
    rho = analyze_cqlf_skip(system, P_sqrt_T)
    return max([rho_i for ((skip_ctrl, skip_u, skip_y), rho_i) in rho.items() if condition(skip_ctrl, skip_u, skip_y)])



if __name__ == "__main__":
    if "range" in sys.argv:
        analyze_cqlf_timing_range(examples.example_C_quadrotor_attitude_one_axis())
    elif "skip" in sys.argv:
        tmp = analyze_cqlf_skip(examples.example_C_quadrotor_attitude_one_axis())
        print(" --- ")
        print("Stability under skips, with perfect timing:")
        print("skip(ctrl, u, y): rho")
        for (skip, rho) in tmp.items():
            print(f"{skip}: {rho}")
    else:
        print(analyze(examples.example_C_quadrotor_attitude_one_axis()))
