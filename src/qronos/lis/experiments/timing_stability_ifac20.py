#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments for:
"Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing" (Gaukler et al., 2019/2020; IFAC World Congress 2020).
An extended preprint including details and proofs is available at https://arxiv.org/abs/1911.02537
"""

from qronos import examples
from qronos.lis import analyze
from qronos.util.latex_table import generate_table, format_float_ceil, format_float_sci_ceil
import sys

def analyze_examples(argv):
    """
    Analyze some example systems to generate the table shown in
    Gaukler et al. (IFAC 2020).
    """
    problems = {}
    
    s = examples.example_A1_stable_1()
    problems['A1'] = s
    s = examples.example_A3_stable_1()
    problems['A3'] = s
    s = examples.example_C_quadrotor_attitude_one_axis()
    problems['C2'] = s
    
    if "--fast" not in argv:
        s = examples.example_D_quadrotor_attitude_three_axis()
        problems['D2'] = s
    
        # Example d2, timing*2
        problems[r'D2\textsubscript{b}: $2\Delta t$'] = examples.example_D2b();
    
        problems[r'D2\textsubscript{c}: $2n$'] = examples.example_D2c();
    
        # Example D2, dimension*2, dt_y_max=0.1*dt_y_max
        problems[r'D2\textsubscript{d}: $2n$, $\frac{\overline{\Delta t}_{\subsMeasure}}{10}$\ifpaper{\!\!}'] = examples.example_D2d();

        # Example D2, dt_y_max=0.1*dt_y_max
        problems[r'D2\textsubscript{e}: $\frac{\overline{\Delta t}_{\subsMeasure}}{10}$\ifpaper{\!\!}'] = examples.example_D2e();


    results = {}
    for (key, s) in problems.items():
        results[key] = analyze.analyze(s)
        results[key]['name'] = key

    print(results)
    print('')

    rho_digits = 3
    columns = columns = [ ('name', 'l|', lambda i: i['name']),
                         ('$n$', 'r', lambda i: i['n']),
                          (r'$\tilde \rho_{\mathrm{approx}}$', 'r', lambda i: format_float_ceil(i.get('rho_approx', float('inf')), rho_digits)),
                          (r'$|\tilde \rho - \tilde \rho_{\mathrm{approx}}|$', 'r', lambda i: '---' if not 'rho' in i else format_float_sci_ceil(i['rho'] - i['rho_approx'], 1)),
                          ('$t_{\mathrm{approx}}$', 'r', lambda i: format_float_ceil(i.get('time_approx', float('inf')), 1)),
                          ('$t$', 'r', lambda i: '---' if not 'time' in i else format_float_ceil(i['time'], 1)),
                         ]
    print(generate_table(columns, results.values()))
    if "--fast" in argv:
        print("CAUTION: The script was run with --fast. Use this ONLY for testing the code, NEVER for publication-ready results.")


def main(argv):
    if argv not in [[], ["--fast"]]:
        print("Supported commandline options are:")
        print("--fast      only run one experiment to test the codepath, skip everything else ")
        sys.exit()
    analyze_examples(argv)

    
if __name__ == "__main__":
    main(sys.argv[1:])
