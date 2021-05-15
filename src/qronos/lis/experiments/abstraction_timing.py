#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments on Convergence Rate Abstractions for uncertain timing or skipped events
"""
import os
import re
import sys

import qronos.lis.lis as l
from qronos.lis import analyze
import qronos.examples as ex
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import tikzplotlib
import scipy.linalg
from qronos.lis.generic_matrix import NumpyMatrix
from copy import deepcopy

# workaround https://github.com/matplotlib/matplotlib/issues/8423
matplotlib.rcParams['axes.unicode_minus']=False

# %%
def main(argv):
    np.random.seed(0)
    plt.figure()

    system = ex.example_D_quadrotor_attitude_three_axis()
    lis = l.LISControlLoop(system, datatype=np)
    n = lis.n
    n_p = lis.sys.n_p
    C = scipy.linalg.block_diag(1 * np.eye(n_p), 0 * np.eye(n - n_p))
    C0 = C
    x0 = C0 @ np.ones((lis.n,))
    # TODO make sure that C != C0 is properly supported; the code currently assumes C=C0 in some places

    P_sqrt_T = np.array([[2.98283705e-06, -2.09845108e-16, -3.05644038e-15,
                        2.59327293e-05, -1.88328101e-03, 6.85284376e-16,
                        -6.24321807e-13, -7.86129022e-15, -2.15863659e-11,
                        1.88175399e-03, 6.24337621e-13, 2.15876471e-11,
                        -3.04494508e-11, -2.03770623e-04, -3.04424360e-11,
                        2.03770562e-04],
                        [0.00000000e+00, 3.08897441e-06, 1.35900905e-14,
                        5.06620829e-16, -3.81575844e-13, 2.70420358e-05,
                        -6.12584959e-04, 1.18696922e-13, -9.06674714e-13,
                        3.81493905e-13, 6.11015350e-04, 8.99582927e-13,
                        -2.11321169e-04, -1.78163924e-12, 2.11321166e-04,
                        -1.75643132e-12],
                        [0.00000000e+00, 0.00000000e+00, 2.59587622e-05,
                        -4.66667971e-15, -1.09213879e-11, -2.83260972e-15,
                        5.50464572e-12, 2.37312359e-04, 1.36973481e-03,
                        1.09213353e-11, -5.50473334e-12, -1.38215773e-03,
                        8.28751096e-05, -8.29492687e-05, 8.28751096e-05,
                        -8.29492687e-05],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        3.34335552e-05, 1.55411611e-03, 9.36636518e-16,
                        4.32713924e-13, -2.99930113e-14, 4.11945598e-11,
                        -1.55295485e-03, -4.32748698e-13, -4.11957200e-11,
                        6.15893333e-11, -2.81959221e-04, 6.16057151e-11,
                        2.81959345e-04],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 4.44904868e-01, 9.23122441e-18,
                        7.31043299e-14, -4.46200413e-14, -6.08666439e-10,
                        -4.44904865e-01, -7.31060000e-14, 6.08663925e-10,
                        -1.88159793e-10, -5.72556594e-07, -1.88160551e-10,
                        5.72180451e-07],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 3.45808573e-05,
                        -3.31054348e-04, 1.56413897e-13, 9.35963643e-13,
                        -3.14497538e-17, 3.32249252e-04, -9.30192832e-13,
                        -2.86772424e-04, -6.87632323e-12, 2.86772411e-04,
                        -6.89189630e-12],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        4.45023028e-01, -2.37382503e-15, -4.86244612e-14,
                        4.00940538e-19, -4.45023029e-01, 4.84973448e-14,
                        7.08416761e-07, 2.26124343e-12, -7.08412246e-07,
                        2.26153390e-12],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 2.86305641e-04, 5.34123555e-03,
                        8.79710102e-17, 6.50741734e-17, -5.33219700e-03,
                        9.38121974e-05, -9.40818028e-05, 9.38121974e-05,
                        -9.40818028e-05],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 4.45148070e-01,
                        8.89579369e-17, 2.77426957e-18, -4.45147989e-01,
                        2.25612180e-06, -2.25225617e-06, 2.25612180e-06,
                        -2.25225611e-06],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        8.90797779e-07, 3.30208354e-17, -5.50422933e-16,
                        6.69069674e-11, -2.39824923e-04, 6.69703451e-11,
                        2.39825057e-04],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 9.20966596e-07, 4.28717908e-15,
                        -2.45651230e-04, -6.04297035e-12, 2.45651218e-04,
                        -5.97337663e-12],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 7.70802831e-06,
                        9.36014677e-05, -9.38397080e-05, 9.36014677e-05,
                        -9.38397080e-05],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        3.14688911e-01, 3.14688578e-01, 3.14688407e-01,
                        3.14688578e-01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 4.53531182e-04, 3.49713599e-04,
                        1.21659720e-04],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 4.41488986e-04,
                        2.62883032e-04],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                        3.48972717e-04]])
    if "--fast" not in argv:
        (rho_list, scale_list, _, _) = analyze.analyze_cqlf_timing_range(system, P_sqrt_T=P_sqrt_T, num_points=3, max_scale=1, verbose=False)
        assert scale_list[0] == 0
        assert scale_list[-1] == 1
        print(f"rho for perfect timing: {rho_list[0]},  for nominal timing: {rho_list[-1]}")

    def max_rho(cond):
        return analyze.analyze_cqlf_skip_condition(lis.sys, P_sqrt_T, cond)


    condition_0 = lambda skip_ctrl, skip_u, skip_y: not any(skip_u) and not any(skip_y) and not skip_ctrl
    rho_0 = max_rho(condition_0)
    assert rho_0 < 0.83
    condition_S = lambda skip_ctrl, skip_u, skip_y: not any(skip_u) and any(skip_y) and not skip_ctrl
    rho_S = max_rho(condition_S)
    assert rho_S < 1.53
    condition_E = lambda skip_ctrl, skip_u, skip_y: all(skip_u) and all(skip_y) and skip_ctrl
    rho_E = max_rho(condition_E)
    assert rho_E < 1.77
    condition_A4 = lambda skip_ctrl, skip_u, skip_y: all(skip_u) and not skip_ctrl
    rho_A4 = max_rho(condition_A4)
    assert rho_A4 < 1.64

    rho_A3 = max_rho(lambda skip_ctrl, skip_u, skip_y: sum(skip_u) in [1, 2, 3] and not skip_ctrl)
    assert rho_A3 < 917
    # Note: for simplicity, the category "X" is broader than its actual definition.
    # Here: "everything", originally: "everything not contained in the other categories".
    # This doesn't hurt; it is pessimistic and doesn't change the result since this category is the worst case anyway.
    rho_X = max_rho(lambda unused1, unused2, unused3: True)
    assert rho_X < 277000

    ### Noise

    # discretize (A_p,B_p) to get B_p_discretized
    # expm(T*[A_p B_p; 0 0]) = [A_p_discretized B_p_discretized; 0 I]
    tmp = np.zeros((lis.sys.n_p + lis.sys.m, lis.sys.n_p + lis.sys.m))
    tmp[0:lis.sys.n_p, 0:lis.sys.n_p] = lis.sys.A_p;
    tmp[0:lis.sys.n_p, lis.sys.n_p:] = lis.sys.B_p;
    d = lis.abstract_matrix_type
    tmp = d.expm(tmp * lis.sys.T)
    B_discrete = tmp[0:lis.sys.n_p, lis.sys.n_p:]
    B_extended = d.zeros(lis.n, lis.n)
    B_extended[0:lis.sys.n_p, 0:lis.sys.m] = B_discrete

    disturbance_amplitude = 1 / 1000  # arbitrarily chosen such that stationarily |Cx| ca. = 1. (choice doesn't matter because system is linear.)
    # factor in disturbance amplitude as scaling of B
    B_extended = B_extended * disturbance_amplitude
    # now, disturbance is normalized to amplitude 1

    # "input gain" of disturbance into abstraction
    beta = NumpyMatrix.qlf_lower_bound(P_sqrt_T, C) / NumpyMatrix.qlf_upper_bound(P_sqrt_T, B_extended)
    print(f"beta after normalization to disturbance amplitude 1 = {beta}")
    print(f"beta with original disturbance amplitude = {beta / disturbance_amplitude}")
    del disturbance_amplitude

    # stationary gain of disturbance into abstraction: v_k for |d|=1
    # v_infinity = rho v_infinity + beta => v_infinity = beta / (1-rho)
    rho = NumpyMatrix.P_norm(lis.Ak_nominal, P_sqrt_T)
    w_impact_max = beta / (1 - rho)

    w_impact_max_actual = sum(
        [scipy.linalg.norm(C @ np.linalg.matrix_power(lis.Ak_nominal, i) @ B_extended, 2) for i in range(999)])

    default_scenario = {}
    default_scenario['rho_normal'] = 0.83
    default_scenario['beta'] = beta
    del beta
    assert rho_0 < default_scenario['rho_normal']
    # default_scenario['rho_skip'] = dict ( condition: rho ), where condition(skip_ctrl, skip_u, skip_y)==True means that the corresponding rho is valid.
    # see analyze.analyze_cqlf_skip_condition().
    default_scenario['rho_skip'] = {condition_0: 0.83, condition_S: 1.53, condition_A4: 1.64, condition_E: 1.77}
    for (condition, rho) in default_scenario['rho_skip'].items():
        assert max_rho(condition) <= rho
    default_scenario['timing_rho_offset'] = 0.860  # TODO compute
    default_scenario['timing_rho_scale'] = 0.0996  # TODO compute
    default_scenario['runs'] = 1 # random runs
    default_scenario['skip_probability'] = np.inf  # probability that an allowed skip will actually happen (np.inf: always)
    default_scenario['random_timing_scale'] = np.inf  # 4 # np.inf  # scaling factor for random timing (if the policy does not restrict the maximum timing). Normalized to 1 = nominal timing deviation.
    default_scenario['y_max_permitted'] = 10
    default_scenario['delta_t_max_permitted'] = 10
    default_scenario['title'] = 'nominal case, perfect timing, no skips'
    default_scenario['soundness'] = True # check if abstraction guarantee holds in simulation -- disable only for heuristically chosen pseudo-abstraction parameters
    default_scenario['forced_skip'] = None # force skip sequence regardless of abstraction: [ (skip_ctrl_1, [skip_y1_1, skip_y2_1, ...], [skip_u1_1, skip_u2_1, ...]), (skip_ctrl_2, ...), ...]
    default_scenario['forced_timing'] = None # force timing sequence regardless of abstraction: [ timing_scale_1, timing_scale_2, ... ],  given relative to delta_t_max_permitted (0...1)

    default_scenario['permit_skip'] = False
    default_scenario['permit_delta_t'] = False
    default_scenario['seed'] = 0

    # %%
    L = 102

    scenario = deepcopy(default_scenario)
    scenario['runs'] = 1
    scenarios = [scenario]
    
    scenario = deepcopy(default_scenario)
    scenario['title'] = 'skips, fixed, short'
    # skip 5 execute 2
    scenario['forced_skip'] = [(False, [False]*3, [False]*4)]*20 + ([(True, [True, True, True], [True,True,True,True])]*5 + [(False, [False, False, False], [False]*4)]*2)*50 + [(True, [True]*3, [True]*4)]*10 +  [(False, [False]*3, [False]*4)]*15 +  [(True, [True]*3, [True]*4)]*10 +  [(False, [False]*3, [False]*4)]*99
    scenario['permit_skip'] = True
    scenario['soundness'] = False
    scenarios.append(scenario)

    scenario = deepcopy(default_scenario)
    # skip 10 execute 15
    scenario['forced_skip'] = [(False, [False]*3, [False]*4)]*20 + ([(True, [True, True, True], [True,True,True,True])]*10 + [(False, [False, False, False], [False]*4)]*15)*50 + [(True, [True]*3, [True]*4)]*10 +  [(False, [False]*3, [False]*4)]*15 +  [(True, [True]*3, [True]*4)]*10 +  [(False, [False]*3, [False]*4)]*99
    scenario['permit_skip'] = True
    scenario['title'] = 'skips fixed long'
    scenarios.append(scenario)
    
    
    scenario = deepcopy(default_scenario)
    scenario['title'] = 'timing forced max'
    scenario['permit_delta_t'] = True
    scenario['forced_timing'] = [0]*50 + [1]*100;
    scenario['delta_t_max_permitted'] = 49
    scenarios.append(scenario)    
    
    
    if "--fast" not in argv:
        scenario = deepcopy(default_scenario)
        scenario['permit_skip'] = True
        scenario['title'] = 'skips, deterministic'
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['skip_probability'] = 0.02
        scenario['title'] = 'skips, low probability'
        scenarios.append(scenario)
    
        scenario = deepcopy(scenario)
        scenario['skip_probability'] = 0.1
        scenario['title'] = 'skips, medium probability'
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['skip_probability'] = 1 - (1 - 0.02) ** 5
        scenario['title'] = 'skips, low probability, only "all or nothing"'
        scenario['rho_skip'][condition_S] = np.inf
        scenario['rho_skip'][condition_A4] = np.inf
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['skip_probability'] = 1 - (1 - 0.1) ** 5
        scenario['title'] = 'skips, medium probability, only "all or nothing"'
        scenarios.append(scenario)

        scenario = deepcopy(default_scenario)
        scenario['permit_skip'] = True
        scenario['runs'] = 1
        scenario['skip_probability'] = 0.6
        scenario['title'] = 'skips, high probability'
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['rho_normal'] = 0.7
        for key in scenario['rho_skip']:
            scenario['rho_skip'][key] = np.power(scenario['rho_skip'][key], 1/5)
        scenario['beta'] = scenario['beta'] * 0.15
        scenario['title'] = 'skips, high probability, ASSUMING BETTER ABSTRACTION (UNPROVEN!)'
        scenario['soundness'] = False
        scenarios.append(scenario)

        scenario = deepcopy(default_scenario)
        scenario['permit_delta_t'] = True
        scenario['title'] = 'timing deviation, infinite variance'
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['random_timing_scale'] = 1
        scenario['title'] = 'timing deviation, medium variance'
        scenarios.append(scenario)

        scenario = deepcopy(scenario)
        scenario['random_timing_scale'] = .25
        scenario['title'] = 'timing deviation, low variance'
        scenarios.append(scenario)

    # uncomment to only show one specific scenario:
    if len(argv) >= 2:
        print(f"Filtering scenarios: only considering '{ argv[1] }'")
        scenarios = [scenario for scenario in scenarios if (argv[1] in scenario['title'])]
    # scenarios=[scenarios[4]]

    for scenario in scenarios:
        print(f'Scenario: {scenario}')
        max_t_static = ((scenario['y_max_permitted'] - scenario['beta']) / (scenario['y_max_permitted']) - scenario['timing_rho_offset']) / scenario['timing_rho_scale']
        print(f'max. static timing would be: sigma={ max_t_static }')
        assert not (scenario['permit_skip'] and scenario['permit_delta_t'])
        plt.figure(figsize=(12, 6))
        first = True
        for run in range(scenario['runs']):
            np.random.seed(scenario['seed'] + run)
            print(run)
            w = np.random.uniform(low=-1, high=1, size=(lis.n, L))
            #    w[(1),:]=w[1,:]
            #    w[(0,2,3),:]=-w[1,:]
            w[lis.sys.m:, :] = 0
            # normalize to |w_k|<1
            w = w / np.broadcast_to(np.sqrt(np.sum(w ** 2, axis=0)), (lis.n, L))
            #if run == 1:
                #w[(1),:]=1
                #w[(0,2,3),:]=0
            # w[:,120:]=0.1*w[:,120:]
            may_skip_y = np.full(L, False)
            may_skip_all_u = may_skip_y.copy()
            may_skip_everything = may_skip_y.copy()
            may_scale_delta_t = np.full(L, 1.)
            skip_y = np.full((lis.sys.p, L), False)
            skip_u = np.full((lis.sys.m, L), False)


            def random_normalized_timings(size):
                # gamma distribution: mean = shape*scale, variance=shape*scale^2.
                # Therefore, scale=variance/mean, shape=mean/scale
                MEAN = 1 / 3
                VARIANCE = 1
                scale = VARIANCE / MEAN
                return np.random.gamma(shape=MEAN / scale, scale=scale, size=size) * np.sign(
                    np.random.uniform(low=-1, high=1, size=size))


            # Note: the timing variance is given normalized to the nominal maximum, regardless of the minimum
            delta_t_u = (random_normalized_timings(size=(lis.sys.m, L)).T * lis.sys.delta_t_u_max).T * scenario['random_timing_scale']
            delta_t_y = (random_normalized_timings(size=(lis.sys.p, L)).T * lis.sys.delta_t_y_max).T * scenario['random_timing_scale']
            skip_ctrl = np.full(L, False)
            scale_delta_t = np.full(L, 0.)
            abstraction = np.full(L, np.nan)
            abstraction[0] = 0  # TODO support x0 != 0
            beta = scenario['beta']
            for i in range(0, L - 1):
                if not scenario['permit_delta_t']:
                    # support skips
                    delta_t_u[:, :] = 0
                    delta_t_y[:, :] = 0
                    if scenario['permit_skip']:
                        def is_allowed(condition):
                            return abstraction[i] * scenario['rho_skip'][condition] + beta <= scenario['y_max_permitted']
                        # check that the "weaker" rule implies the "stronger" one, as claimed in the paper
                        assert is_allowed(condition_S) >= is_allowed(condition_A4)
                        if not np.isinf(scenario['rho_skip'][condition_A4]): # A4 was not disabled
                            assert is_allowed(condition_A4) >= is_allowed(condition_E)
                        if is_allowed(condition_S):
                            may_skip_y[i] = True
                            # Skip any number of sensors
                            skip_y[:, i] = np.random.uniform(size=lis.sys.p) > 1 - scenario['skip_probability']
                        if is_allowed(condition_A4):
                            may_skip_all_u[i] = True
                            if np.random.uniform(size=1) > 1 - scenario['skip_probability']:
                                # Skip *all* actuators or none at all
                                skip_u[:, i] = True
                        if is_allowed(condition_E):
                            may_skip_everything[i] = True
                            # Skip everything
                            if np.random.uniform(size=1) > 1 - scenario['skip_probability']:
                                skip_ctrl[i ] = True
                                skip_y[:, i] = True
                                skip_u[:, i] = True
                        if scenario['forced_skip'] is not None:
                            [skip_ctrl[i], skip_y[:, i], skip_u[:, i]] = scenario['forced_skip'][i]
                    rho_i = None
                    if any(skip_u[:, i]) or any(skip_y[:, i]) or skip_ctrl[i]:
                        rho_i = min([rho_conditional for (condition, rho_conditional) in scenario['rho_skip'].items() if
                                    condition(skip_u=skip_u[:, i], skip_y=skip_y[:, i], skip_ctrl=skip_ctrl[i])])
                    else:
                        rho_i = scenario['rho_normal']
                    if rho_i is None:
                        raise Exception("invalid skip combination; no rho known")
                    abstraction[i + 1] = abstraction[i] * rho_i + beta
                elif scenario['permit_delta_t']:
                    # max. delta_t such that abstraction[i+1] <= scenario['y_max_permitted']:
                    # abstraction[i+1] = beta + abstraction[i] * (offset + scale * delta_t_normalized)
                    # from qronos.lis.analyze import analyze_cqlf_timing_range
                    # from qronos import examples
                    # (_, _, offset, scale) = analyze_cqlf_timing_range(examples.example_D_quadrotor_attitude_three_axis(), P_sqrt_T=P_sqrt_T)
                    # print(offset, scale)
                    offset = scenario['timing_rho_offset']
                    scale = scenario['timing_rho_scale']
                    # Compute permissible scaling of delta-t range
                    may_scale_delta_t[i] = ((scenario['y_max_permitted'] - beta - 1e-9) / (abstraction[i] + 1e-99)
                                            - offset) / scale
                    may_scale_delta_t[i] = np.clip(may_scale_delta_t[i], 0, scenario['delta_t_max_permitted'])
                    if scenario['forced_timing'] is not None:
                        may_scale_delta_t[i] = scenario['forced_timing'][i] * scenario['delta_t_max_permitted']
                    # clip precomputed random delta-t to permissible (scaled) range
                    delta_t_u[:, i] = np.clip(delta_t_u[:, i],
                                                lis.sys.delta_t_u_min * may_scale_delta_t[i],
                                                lis.sys.delta_t_u_max * may_scale_delta_t[i])
                    delta_t_y[:, i] = np.clip(delta_t_y[:, i],
                                                lis.sys.delta_t_y_min * may_scale_delta_t[i],
                                                lis.sys.delta_t_y_max * may_scale_delta_t[i])

                    # simplifying assumptions; modify the computation of scale_delta_t to remove them:
                    assert all(lis.sys.delta_t_u_min < 0)
                    assert all(lis.sys.delta_t_u_max > 0)
                    assert all(lis.sys.delta_t_y_min < 0)
                    assert all(lis.sys.delta_t_y_max > 0)
                    # get smallest scale_delta_t >= 0 such that
                    # delta_t_min <= delta_t <= delta_t_max  (componentwise, for y and u).
                    # under the above assumptions, this is the maximum of:
                    #    delta_t / delta_t_max   if  delta_t > 0
                    #    delta_t / delta_t_min   if  delta_t < 0
                    scale_delta_t[i] = np.max(np.hstack(
                        (delta_t_u[:, i] / lis.sys.delta_t_u_min,
                        delta_t_u[:, i] / lis.sys.delta_t_u_max,
                        delta_t_y[:, i] / lis.sys.delta_t_y_min,
                        delta_t_y[:, i] / lis.sys.delta_t_y_max)
                    ))
                    assert scale_delta_t[i] <= may_scale_delta_t[i] * (1 + 1e-12)
                    abstraction_worst_predicted = beta + abstraction[i] * (offset + scale * may_scale_delta_t[i])
                    abstraction[i + 1] = beta + abstraction[i] * (offset + scale * scale_delta_t[i])
                    assert abstraction[i + 1] <= abstraction_worst_predicted * (1 + 1e-12)
                    if scenario['forced_timing'] is None:
                        assert abstraction_worst_predicted <= scenario['y_max_permitted']
            plt.subplot(2, 1, 1)
            plt.grid()
            plt.plot([0, L], [scenario['y_max_permitted'], scenario['y_max_permitted']], 'k--',
                    label='allowed maximum' if first else '_nolabel_')
            abstraction[abstraction > 1e3] = np.inf; # work around pgfplots floating-point error
            plt.plot(abstraction, color=plt.cm.Set1.colors[2], label='abstraction' if first else '_nolabel_')
            # plt.ylim(0,11); -> disabled to avoid NotImplementedError
            plt.subplot(2, 1, 2)
            plt.xlabel('$k$')
            plt.grid()
            if scenario['permit_delta_t']:
                plt.ylabel('$|\Delta t / \Delta t_{max,nominal}|$')
                plt.plot(may_scale_delta_t, 'r', label='allowed delta-t scaling')
                plt.plot(scale_delta_t, 'b', label='actual delta-t scaling')
                # TODO is the following computation correct (shapes for division)?
                plt.plot(delta_t_u.T / lis.sys.delta_t_u_max, label='$\Delta t_u$ normalized to nominal max.')
                plt.plot(delta_t_y.T / lis.sys.delta_t_y_max, label='$\Delta t_y$ normalized to nominal max.')
                plt.plot(np.full(L, max_t_static), label='max. static $|\Delta t|$ normalized')
            if scenario['permit_skip']:
                plt.yticks([])
                # This checks that skip_u only if skip_y, and skip_everything only if skip_u,
                # *except* if some scenarios have been disabled by setting the corresponding rho to infinity
                assert all(may_skip_y >= may_skip_all_u) or np.isinf(scenario['rho_skip'][condition_S])
                assert all(may_skip_all_u >= may_skip_everything) or np.isinf(scenario['rho_skip'][condition_A4])

                def plot_activation_bar(skip: np.ndarray, yshift: float, color='k', **kwargs):
                    height = 0.1
                    yshift = -yshift * 1.25 * height
                    # the following is roughly equivalent to:
                    # plt.fill_between(np.arange(0, len(skip)), skip * height + yshift, yshift, step='pre', color=color, **kwargs)
                    # however, this is not well supported by tikzplotlib.
                    skip = np.hstack(([False], skip, [False]))
                    plt.step(np.arange(-1, len(skip) - 1), skip * height + yshift, color=color, **kwargs)
                plot_activation_bar(may_skip_y, 0, plt.cm.Paired.colors[0], label='may skip sensors')
                plot_activation_bar(may_skip_all_u, 1, plt.cm.Paired.colors[2], label='may skip actuators')
                plot_activation_bar(may_skip_everything, 2, (.5, .5, .5), label='may skip everything')
                j = 3
                first_plot = True
                for skip_y_i in skip_y:
                    j = j + 1
                    plot_activation_bar(skip_y_i, j, plt.cm.Paired.colors[1], label=f'skip sensor' if first_plot else '_nolabel_')
                    first_plot = False
                first_plot = True
                j = j + 1
                plot_activation_bar(skip_u[0,:], j, plt.cm.Paired.colors[3], label=f'skip actuators')
                j = j + 1
                plot_activation_bar(skip_ctrl, j, color=plt.cm.Paired.colors[5], label='skip controller')

            # SKIP_PERIOD = 40
            # SKIP_AMOUNT = 3
            # skip_y[:,np.mod(np.arange(L), SKIP_PERIOD) > SKIP_PERIOD - 1 - SKIP_AMOUNT] = True
            x, Ak = lis.simulate_random(L, x0=np.zeros(lis.n, ), scale_timing=scale_delta_t, w=B_extended @ w,
                                        skip_y=skip_y)
            w_impact_max_actual_k_upper = np.zeros(L);
            w_impact_max_actual_k_lower = np.zeros(L);
            w_impact_max_actual_k_upper_nominal = np.zeros(L);
            Ak_product = np.zeros(Ak.shape)
            for i in range(L):
                Ak_product[i, :, :] = np.eye(lis.n)
                # Ak[i, :, :] = np.eye(lis.n)
            for i in range(1, L):
                # compute Ak_product (matrices Phi_i such that x_k =  sum(Phi_i G w_i)):
                # k=0 -> Ak_product = []             -> x_0 = 0
                # k=1 -> Ak_product = [I]            -> x_1 = G w_0
                # k=2 -> Ak_product = [A0, I]        -> x_2 = A_0 G w_0 + G w_1
                # k=3 -> Ak_product = [A1@A0, A1, I] -> x_3 = A_1 A_0 G w_0 + ...
                # k=4 -> Ak_product = [A2@A1@A0, A2@A1, A2, I]
                for j in range(i - 1):  # j = 0 ... i-2
                    Ak_product[j, :, :] = Ak[i - 2, :, :] @ Ak_product[j, :, :]
                # y_3 = C A_1 A_0 G w_0 + C A_1 G w_1 + C G w_2
                # y_3 nominal <=  ||C A_1 A_0 G|| + ||C A_1 G|| + ||C G||, where || || is the spectral norm
                w_impact_max_actual_k_upper_nominal[i] = sum([scipy.linalg.norm(C @ np.linalg.matrix_power(lis.Ak_nominal, j) @ B_extended, 2) for j in range(i)])
                w_impact_max_actual_k_upper[i] = sum([scipy.linalg.norm(C @ Ak_product[j, :, :] @ B_extended, 2) for j in range(i)])
                w_impact_max_actual_k_lower[i] = np.sqrt(sum(np.array([scipy.linalg.norm(C @ Ak_product[j, :, :] @ B_extended, 2) for j in range(i)]) ** 2))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.subplot(2, 1, 1)
            # plt.plot(w_impact_max_actual_k_upper_nominal, 'c',
            #          label='computed upper bound w.o. timing, w.o. skips' if first else '_nolabel_')
            plt.plot(w_impact_max_actual_k_upper, color=plt.cm.Set1.colors[0],
                    label='computed upper bound' if first else '_nolabel_')
            plt.plot(w_impact_max_actual_k_lower, color=plt.cm.Set1.colors[1],
                    label='computed lower worst-case bound' if first else '_nolabel_')
            mag_x = np.sqrt(np.sum((C @ x) ** 2, axis=0))
            plt.plot(mag_x, 'k', label='simulation with random noise' if first else '_nolabel_')
            w_impact_max_simulated = np.max(mag_x)

            first = False
            w_impact_max_abstracted = np.max(abstraction)
            if scenario['soundness']:
                # TODO this test is simplified, it should check every time k, not just the global maxima
                assert w_impact_max_simulated <= w_impact_max_abstracted
            print(
                f'|y|_max under disturbance w=Gd, |d|<1: abstraction: {w_impact_max_abstracted}, simulation: {w_impact_max_simulated}  => pessimism {w_impact_max_abstracted / w_impact_max_simulated:.2f}')
            plt.ylabel('$|s_k|$')
        # plt.subplots_adjust(right=.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(scenario['title'])
        dirname = "../../../../output/abstraction-timing-" + re.sub("[^a-zA-Z]", "_", scenario['title']) + "/"
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
        fname = dirname + "plot"
        try:
            plt.savefig(fname=dirname[:-1] + ".pdf")
            plt.title('')
            tikzplotlib.clean_figure()
            tikzplotlib.save(fname + ".tex", externalize_tables=True, override_externals=True)
        except OSError as e:
            print("Cannot save file, does the output dir exist? is the script run from its directory? " + str(e))

if __name__ == "__main__":
    main(sys.argv)
