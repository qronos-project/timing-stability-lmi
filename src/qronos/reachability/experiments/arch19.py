#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiments for the publication:

Gaukler, M. and Ulbrich, P. (2019):
Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing
ARCH19. International Workshop on Applied veRification for Continuous and Hybrid Systems

(Note: The original code corresponding to that publication was published on https://github.com/qronos-project/arch19-benchmark-iotiming . It has been merged into this codebase.)
"""

import os
import shutil
import logging
import pickle
from qronos import examples

import sys
import numpy as np
from qronos.reachability.hybrid_sys import HybridSysControlLoop
from qronos.util.latex_table import generate_table, format_float_ceil
from hybridpy import hypy

def output_dir():
    return os.path.dirname(os.path.realpath(__file__)) + "/output/"

def main(argv):
    if "--help" in argv:
        print("--load: load saved previous results (only runs the code for formatting the result table, skips the time-extensive actual analysis)")
        print("--fast: only run a few experiments and not all, for a quick test of the toolchain")
        print("--ignore-hypy: deprecated, don't use anymore. was historically used to run a part of the code if hyst/hypy is not available.")
        sys.exit()
    if os.path.exists(output_dir()) and not "--load" in argv:
        shutil.rmtree(output_dir())
    for directory in ["unsolved/unknown", "unsolved/unstable", "unsolved/stable", "solved_with_spaceex/stable"]:
        directory = output_dir() + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Files are denoted with a unique prefix (e.g. A1) to simplify referencing them in publications
    systems={}
    systems['solved_with_spaceex/stable/A1_1']=examples.example_A1_stable_1()
    if not "--fast" in argv:
        systems['solved_with_spaceex/stable/A2_1']=examples.example_A2_stable_1()
        systems['solved_with_spaceex/stable/A3_1']=examples.example_A3_stable_1()
        systems['unsolved/unknown/A4_1']=examples.example_A4_unknown_1()
        systems['unsolved/stable/A5_diagonal_2']=examples.example_A5_stable_diagonal(2)
        systems['solved_with_spaceex/stable/B1_stable_3']=examples.example_B1_stable_3()
        systems['unsolved/stable/C1_quadrotor_one_axis_no_jitter_1']=examples.example_C_quadrotor_attitude_one_axis(perfect_timing=True)
        systems['unsolved/unknown/C2_quadrotor_one_axis_with_jitter_1']=examples.example_C_quadrotor_attitude_one_axis(perfect_timing=False)
        systems['unsolved/stable/D1_quadrotor_attitude_three_axis_no_jitter_3']=examples.example_D_quadrotor_attitude_three_axis(perfect_timing=True)
        systems['unsolved/unknown/D2_quadrotor_attitude_three_axis_with_jitter_3']=examples.example_D_quadrotor_attitude_three_axis(perfect_timing=False)
        systems['unsolved/unstable/E_timer']=examples.example_E_timer()
        assert systems['unsolved/unstable/E_timer'].nominal_case_stability() == 'borderline unstable'
    
    for key in systems:
        systems[key] = HybridSysControlLoop(systems[key])
    
    # Test that systems marked as 'stable' are actually stable
    for (key, system) in systems.items():
        if '/stable' in key:
            if all(system.s.delta_t_u_min <= 0) and all(system.s.delta_t_y_min <= 0) \
            and all(system.s.delta_t_u_max >= 0) and all(system.s.delta_t_y_max >= 0):
                # first sanity check (not sufficient, only necessary):
                # if the nominal case delta_t=0 is included in the possible timings,
                # then, a stable system's nominal case (delta_t=0) must be stable too:
                assert system.s.nominal_case_stability() == 'stable'
    
            # sufficient stability tests:
            if key.startswith('solved_'):
                # the system was verified by manually calling some verification program.
                pass
            elif key == 'unsolved/stable/A5_diagonal_2':
                # The system A5_diagonal_x is the diagonal repetition of example A1, which is stable (shown by SpaceEx),
                # so A5_diagonal_x must be stable as well.
                assert 'solved_with_spaceex/stable/A1_1' in systems
                # stability of A1_1 is then tested implicitly.
            else:
                # if the system hasn't been verified with SpaceEx, but is marked as stable,
                # we need to show that it's stable using some stability test.
                # Currently, only one possibility is implemented:
                # - The timing is only the nominal case, and the nominal case is stable.
                # (Future work could include less restrictive stability tests.)
                assert system.s.is_nominal_timing() and system.s.nominal_case_stability() == 'stable', \
                    'system {} is marked stable and unsolved, but stability could not be tested: It is marked as unsolved, i.e. not verified by SpaceEx or some other tool. A simplified stability test was not applicable because the timing is not strictly zero.'.format(key)
    
    
    # If system names (the keys of systems[]) are given on the command line, process only these.
    # NOTE: invalid names will be ignored.
    requested_system_names = set(systems.keys()).intersection(set(argv))
    if requested_system_names:
        print("Example names were given on the command line. Only processing these: {}".format(", ".join(requested_system_names)))
        systems = {name: system for (name, system) in systems.items() if name in requested_system_names}
    
    if "--load" in argv:
        # Load results from file
        with open(output_dir() + "systems.pickle", "rb") as f:
            systems=pickle.load(f)
    else:
        # Save systems to files, run analysis and simulation
        for (name, system) in sorted(systems.items()):
            try:
                system.run_analysis(name, output_dir())
                if not "--ignore-hypy" in argv:
                    if name.startswith("solved"):
                        assert system.results['spaceex_hypy']['code'] == hypy.Engine.SUCCESS, "SpaceEx failed, but the system is marked as solved"
                    else:
                        # "unsolved" system: either SpaceEx crash or SpaceEx fails to compute fixout
                        assert system.results['spaceex_hypy']['code'] != hypy.Engine.SUCCESS or not system.results['spaceex_hypy']['output'].get('fixpoint', False), "System marked as unsolved, but SpaceEx succeeded!"
            except Exception:
                logging.error("Failed to process system {}".format(name))
                raise
        with open(output_dir() + "systems.pickle", "wb") as f:
            pickle.dump(systems, f)
    
    
    # Manual modifications to LaTeX table
    # System E is unstable (plant is double integrator without input)
    if 'unsolved/unstable/E_timer' in systems:
        assert np.all(systems['unsolved/unstable/E_timer'].s.A_p == np.array([[0, 1], [0, 0]]))
        assert np.all(systems['unsolved/unstable/E_timer'].s.B_p == 0)
        systems['unsolved/unstable/E_timer'].results['stability_eigenvalues'] = 'unstable'
        systems['unsolved/unstable/E_timer'].results['stability_spaceex'] = 'N/A'
    
    
    
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
    columns = [ ('name', 'l|', lambda s: s.name.split("/")[-1].split("_")[0]),
                (r'$n\idxPlant$', 'c',  lambda s: s.s.n_p),
                (r'$n\idxDiscrete$', 'c', lambda s: s.s.n_d),
                ('$m$', 'c', lambda s: s.s.m),
                ('$p$', 'c', lambda s: s.s.p),
                ('timing', 'l|', lambda s: 'constant' if s.s.is_fixed_timing() else 'varying'),
                ('SpaceEx', 'l', lambda s: format_spaceex_columns(s)['result']),
                (r'$t_{\mathrm{SE}}$', 'r', lambda s: format_spaceex_columns(s)['runtime']),
                (r'$K_{\mathrm{SE}}$', 'r|', lambda s: format_float_ceil(s.results['k'], digits=3) if 'k' in s.results else '---'),
                ('LTI-stability', 'l', lambda s: s.results['stability_eigenvalues'].replace("N/A","---"))
            ]
    
    
    
    table = generate_table(columns, sorted(iter(systems.values()), key = lambda s: s.name.split("/")[-1]))
    print(table)
    with open(output_dir() + "results.tex", "w") as f:
        f.write(table)
    
    if "--fast" in argv:
        print("CAUTION: The script was run with --fast, which means that the results are imprecise and/or useless. Use this ONLY for testing the code, NEVER for publication-ready results.")

if __name__ == "__main__":
    main(sys.argv[1:])