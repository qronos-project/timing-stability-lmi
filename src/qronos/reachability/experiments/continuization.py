#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiments on Continuization

Gaukler, Maximilian (2020): Analysis of Real-Time Control Systems using First-Order Continuization.
In: 7th International Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH20).
https://doi.org/10.29007/8nq6

See also: Doctoral Thesis by Maximilian Gaukler 2022, Chapter 5.2 Continuization of Hybrid Automata.
"""
# Possible improvement: Merge this file with the other reachability experiments by making everything configurable

import os
import shutil
import logging
import pickle
import sys
import numpy as np
import math
from qronos.reachability.hybrid_sys import HybridSysControlLoop
from qronos.util.latex_table import generate_table, format_float_ceil
from copy import deepcopy
import qronos.examples

def output_dir():
    return os.path.dirname(os.path.realpath(__file__)) + "/../../../../output/continuization_arch20/"

from qronos.controlloop import DigitalControlLoop

def example_C3_discrete_PI():
    '''
    example_C_quadrotor_attitude_one_axis(), but without the one-step delay from y to u
    '''
    system = qronos.examples.example_C_quadrotor_attitude_one_axis()
    system.enable_immediate_ctrl()
    return system

def example_C4_with_continuous_P_and_discrete_I():
    '''
    academic example based on example_C_quadrotor_attitude_one_axis:

    P-controller is (unsoundly) merged into the continuous system.
    Only the I-controller is discrete.
    '''

    system=DigitalControlLoop()
    Jx=9.0359e-06
    K_control_integral = 3.6144e-3 # K_I,p
    K_control_proportional = 2.5557e-4 # K_f,p
    system.A_p = np.array([[-K_control_proportional/Jx]])
    system.B_p = np.array([[1/Jx]])

    system.T=0.003
    system.spaceex_time_horizon_periods = math.ceil(0.5/system.T)

    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([1]) * -1
    system.x_p_0_max = np.array([1]) * 1

    system.A_d = np.array([[1]])
    system.B_d = np.array([[system.T]])

    system.C_d = np.array([[-K_control_integral]])

    max_timing = 0.0
    system.delta_t_u_min=np.array([1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=system.delta_t_u_min
    system.delta_t_y_max=-system.delta_t_y_min

    system.spaceex_iterations = 100
    system.spaceex_iterations_for_global_time = system.spaceex_time_horizon_periods
    system.spaceex_timeout = 0.0001* 3600 * 10
    system.enable_immediate_ctrl()
    system.plot_ylim_xp = [[-1.5, 1.5]]
    system.plot_ylim_xd = [[-0.035, 0.035]]
    return system

def example_C5_with_lowpass_PI():
    '''
    modified version of example_C_quadrotor_attitude_one_axis

    discrete PI controller, P channel is lowpass filtered
    '''
    system=DigitalControlLoop()
    Jx=9.0359e-06
    K_control_integral = 3.6144e-3 # K_I,p
    K_control_proportional = 2.5557e-4 # K_f,p
    system.A_p = np.array([[0]])
    system.B_p = np.array([[1/Jx]])

    # the original model is continuous. We consider a sampled version of the controller.
    # All following parameters are not from the original example.
    system.T=0.001
    system.spaceex_time_horizon_periods = math.ceil(0.5 / system.T)

    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([1]) * -1
    system.x_p_0_max = np.array([1]) * 1

    # We use the following controller discretization::
    # x_d_1: forward-euler approximation of integrator
    # x_d_2: lowpass for P controller
    T_lowpass = 0.01 # continuous-time equivalent time constant of lowpass
    lowpass_alpha = np.exp(-system.T / T_lowpass) # discretization
    system.A_d = np.asarray(np.diag([1, lowpass_alpha]))
    system.B_d = np.array([[system.T], [1 - lowpass_alpha]])

    system.C_d = np.array([[-K_control_integral, -K_control_proportional]])

    max_timing = 0.0
    system.delta_t_u_min=np.array([1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=system.delta_t_u_min
    system.delta_t_y_max=-system.delta_t_y_min

    # hits timeout even for 10h :-(
    system.spaceex_iterations_for_global_time = system.spaceex_time_horizon_periods
    system.spaceex_timeout = 3600 * 10
    system.enable_immediate_ctrl()
    system.plot_ylim_xp = [[-1.5, 1.5]]
    system.plot_ylim_xd = [[-0.035, 0.035], [-1, 1]]
    return system

def main(argv):
    if "--help" in argv:
        # print("--load: load saved previous results (only runs the code for formatting the result table, skips the time-extensive actual analysis)")
        print("--fast: only run a few experiments and not all, with very short timeout, for a quick test of the toolchain")
        sys.exit()
    if os.path.exists(output_dir()) and not "--load" in argv:
        shutil.rmtree(output_dir())
    os.makedirs(output_dir())

    # Files are denoted with a unique prefix (e.g. A1) to simplify referencing them in publications
    systems={}
    systems['C4']=example_C4_with_continuous_P_and_discrete_I()
    if not "--fast" in sys.argv:
        systems['C5']=example_C5_with_lowpass_PI()
        systems['C3']=example_C3_discrete_PI()
        systems['C3'].spaceex_timeout = 360 # we just want to check if continuization would work, so 10min per iteration step must be enough. The analysis fails at a point where the reachable set of x_p has already grown to 3x the actual reachable set, so aborting then is fine.
        systems['C3_T_0.001']=example_C3_discrete_PI()
        systems['C3_T_0.001'].T = 0.001
        systems['C3_T_0.001'].spaceex_timeout = systems['C3'].spaceex_timeout # see above
        systems['C3_T_0.001'].spaceex_timeout_extra_plots = 3600 # something is wrong with SpaceEx for this system, computing the extra plots takes forever --> abort early


    for key in systems:
        systems[key] = HybridSysControlLoop(systems[key])

    # If system names (the keys of systems[]) are given on the command line, process only these.
    # NOTE: invalid names will be ignored.
    requested_system_names = set(systems.keys()).intersection(set(argv))
    if requested_system_names:
        print("Example names were given on the command line. Only processing these: {}".format(", ".join(requested_system_names)))
        systems = {name: system for (name, system) in systems.items() if name in requested_system_names}

    if "--load" in argv:
        raise NotImplementedError("--load not yet supported. TODO: implement pickle functionality for mpmath intervals and matrices.")
        # Load results from file
        # TODO re-enable pickling below
        with open(output_dir() + "systems.pickle", "rb") as f:
            systems=pickle.load(f)
    else:
        # Save systems to files, run analysis and simulation
        tmp = systems
        systems = {}
        for (name, system) in tmp.items():
            for suffix in ["_continuized", "_orig"]:
                systems[name + suffix] = deepcopy(system)
                systems[name + suffix].name = name + suffix

        for (name, system) in sorted(systems.items()):
            try:
                if name.endswith("_continuized"):
                    system.run_analysis_continuized(name, output_dir())
                else:
                    system.run_analysis(name, output_dir())
            except Exception:
                logging.error("Failed to process system {}".format(name))
                raise
        with open(output_dir() + "systems.pickle", "wb") as f:
            # pickling disabled for now
            # pickle.dump(systems, f)
            pass



    # Generate LaTeX table
    print("producing LaTeX table")
    def format_spaceex_columns(system):
        def format_spaceex_result(stability, time):
            if stability == "stable":
                return r"\checkmark"
            elif stability=="N/A":
                return "---"
            else:
                return r"$\times$ " + stability
        def format_spaceex_runtime(stability, time):
            if stability.startswith("crash") or stability=="N/A" or stability.startswith("diverging") or stability.startswith("error"):
                return "---"
            if stability.startswith("timeout") and time >= 7200:
                return "---"
            return "{:.0f}\,s".format(time)
        stability = system.results.get('stability_spaceex', "NOT RUN")
        time = system.results.get('spaceex_hypy', {}).get('time', -1)
        return {'result':  format_spaceex_result(stability, time),
                'runtime':  format_spaceex_runtime(stability, time)}

    for (name, system) in sorted(systems.items()):
        system.name = name
    # [ ('column name', 'alignment', lambda system: generate_column_from_system(system)), ... ]
    columns = [ ('name', 'l|', lambda s: s.name),
                ('Continuization', 'l|', lambda s: s.results.get('continuization', '')),
                (r'$n\idxPlant$', 'c',  lambda s: s.s.n_p),
                (r'$n\idxDiscrete$', 'c', lambda s: s.s.n_d),
                ('$m$', 'c', lambda s: s.s.m),
                ('$p$', 'c', lambda s: s.s.p),
                ('timing', 'l|', lambda s: 'constant' if s.s.is_fixed_timing() else 'varying'),
                ('$T$', 'l|',  lambda s: str(s.s.T)),
                ('SpaceEx', 'l', lambda s: format_spaceex_columns(s)['result']),
                (r'$t_{\mathrm{SE}}$', 'r', lambda s: format_spaceex_columns(s)['runtime']),
                (r'$K_{\mathrm{SE}}$', 'r|', lambda s: format_float_ceil(s.results['k'], digits=3) if 'k' in s.results else '---'),
                # TODO implement LTI stability check for this case
                # ('LTI-stability', 'l', lambda s: s.results['stability_eigenvalues'].replace("N/A","---"))
            ]



    table = generate_table(columns, sorted(iter(systems.values()), key = lambda s: s.name.split("/")[-1]))
    print(table)
    with open(output_dir() + "results.tex", "w") as f:
        f.write(table)

    if "--fast" in argv:
        print("CAUTION: The script was run with --fast, which means that the results are imprecise and/or useless. Use this ONLY for testing the code, NEVER for publication-ready results.")

if __name__ == "__main__":
    main(sys.argv[1:])
