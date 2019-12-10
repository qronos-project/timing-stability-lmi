#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Stability analysis based on Linear Impulsive Systems and Linear Matrix Inequalities.
See Gaukler et al. (2019/2020): Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing. Submitted for publication.
"""
from .. import examples
from .lis import LISControlLoop
import mpmath as mp
import numpy as np
from mpmath import iv
from .iv_matrix_utils import numpy_ndarray_to_mp_matrix
from .generic_matrix import convert, check_datatype, approx_max_abs_eig
from . import lmi_cqlf

from .norms import iv_P_norm, iv_P_norm_expm, approx_P_norm_expm, iv_spectral_norm, approx_P_norm
from datetime import datetime

# TODO document datatype
def analyze(s, datatype=None):
    datatype = check_datatype(datatype or iv)
    print("")
    print("")
    print("Analyzing system: {}".format(s))
    print("Analysis is exact (interval arithmetic): {}".format(datatype == iv))
    print("Please note that some of the above parameters, eg. spaceex_... are not used in this LMI-based analysis, they are only present for reachability analysis.")
    # TODO: move the parameters mentioned above to some sub-structure, e.g. sys.param.reach.XXX
    time_start = datetime.now()
    l = LISControlLoop(s, datatype)
    result = {'time_approx': None, 'time': None, 'n': l.n}
    A = l.Ak_nominal
    rho_ideal = approx_max_abs_eig(A)
    print('spectral radius(A) = ', rho_ideal)
    print('interval spectral_norm(A) = ', iv_spectral_norm(A))
    
    #%% 
    
    # "Pressure factor" to reduce P_norm(A) at the cost of numerical robustness and the other goals (see other factors)
    tighteningRho = 0.8 # 0 ... 1, typically 0.8
    
    print('tightening rho=', tighteningRho)
    
    
    precondition=True
    print('precondition=', precondition)
    
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
        Delta_list.append(l.Ak_delta_to_nominal_approx(dtu, dty))
    
    
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
        pnorm_approx = approx_P_norm(M=A, P_sqrt_T=P_sqrt_T)
        print('approx P_norm(A) = ', pnorm_approx)
        rho_approx = pnorm_approx
        
        for [M1, A_cont, M2, tau, info] in l.m1_a_m2_tau():
            print(info + ":")
            pnorm_exp_approx = approx_P_norm_expm(P_sqrt_T, M1=M1, A=A_cont, M2=M2, tau=tau)
            print('sampled P_norm(...expm(...)...) = ', pnorm_exp_approx)
            rho_approx += pnorm_exp_approx
        
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
        # TODO: proper formula and definition for alpha, this is only a first guess (however the result is not yet used)
        p_sqrt_eigv = np.diag(convert(P_sqrt_T, np))
        print('Eccentricity alpha (approx):', max(p_sqrt_eigv)/min(p_sqrt_eigv))
    # Choose best result
    (beta, rho_approx, gamma, P_sqrt_T) = min(P_list, key=lambda x: x[1])
    print("Best rho (approx): ", rho_approx)
    result['rho_approx'] = rho_approx
    result['time_approx'] = (datetime.now() - time_start).total_seconds()
    if rho_approx > 1:
        print("Cannot show stability. Returning approximate result for instability, skipping verification.")
        return result
    if datatype == np:
        print("Only inexact computation was requested. Not performing exact analysis")
        return result
    #%%
    print("Exact results:")
    pnorm = iv_P_norm(M=A, P_sqrt_T=P_sqrt_T)
    print('interval P_norm(A) = ', pnorm)
    rho_total = pnorm
    for [M1, A_cont, M2, tau, info] in l.m1_a_m2_tau():
        print(info + ":")
        pnorm_exp = iv_P_norm_expm(P_sqrt_T, M1=M1, A=A_cont, M2=M2, tau=tau)
        print('interval P_norm(...expm(...)) = ', pnorm_exp)
        pnorm_exp_approx = approx_P_norm_expm(P_sqrt_T, M1=M1, A=A_cont, M2=M2, tau=tau)
        rho_total += pnorm_exp
        print('sampled P_norm(...expm(...)) = ', pnorm_exp_approx)
    print('Overall rho: ', rho_total)
    # The following is a simple but slightly pessimistic variant for rounding up from mpmath.mpi to float
    result['rho'] = float(mp.mpf(rho_total.b)) * 1.0000001
    result['time'] = (datetime.now() - time_start).total_seconds()

    
    return result

def analyze_examples():
    """
    Analyze some example systems to generate the table shown in
    Gaukler et al. (2019).
    """
    problems = {}
    s = examples.example_C_quadrotor_attitude_one_axis()
    problems['C2'] = s

    s = examples.example_D_quadrotor_attitude_three_axis()
    problems['D2'] = s

    # Example d2, timing*2    
    s = examples.example_D_quadrotor_attitude_three_axis()
    s.increase_timing(2)
    problems[r'D2\textsubscript{b}: $2\Delta t$'] = s
    
    # Example D2, dimension*2
    s = examples.example_D_quadrotor_attitude_three_axis()
    s.increase_dimension(2)
    problems[r'D2\textsubscript{c}: $2n$'] = s
    
    # Example D2, dimension*2, dt_y_max=0.1*dt_y_max
    s = examples.example_D_quadrotor_attitude_three_axis()
    s.increase_dimension(2)
    s.delta_t_y_max=0.1*s.delta_t_y_max
    problems[r'D2\textsubscript{d}: $2n$, $\frac{\overline{\Delta t}_{\subsMeasure}}{10}$'] = s
    
    results = {}
    for (key, s) in problems.items():
        results[key] = analyze(s)
        results[key]['name'] = key

    print(results)
    print('')
    
    from ..util.latex_table import generate_table, format_float_ceil, format_float_sci_ceil
    rho_digits = 3
    columns = columns = [ ('name', 'l|', lambda i: i['name']),
                         ('$n$', 'r', lambda i: i['n']),
                          (r'$\tilde \rho_{\mathrm{approx}}$', 'r', lambda i: format_float_ceil(i.get('rho_approx', float('inf')), rho_digits)),
                          (r'$|\tilde \rho - \tilde \rho_{\mathrm{approx}}|$', 'r', lambda i: '---' if not 'rho' in i else format_float_sci_ceil(i['rho'] - i['rho_approx'], 1)),
                          ('$t_{\mathrm{approx}}$', 'r', lambda i: format_float_ceil(i.get('time_approx', float('inf')), 1)),
                          ('$t$', 'r', lambda i: '---' if not 'time' in i else format_float_ceil(i['time'], 1)),
                         ]
    print(generate_table(columns, results.values()))


if __name__ == "__main__":
    analyze_examples()
