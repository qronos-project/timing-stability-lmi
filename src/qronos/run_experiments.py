#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all experiments
"""
import time
import sys
import qronos.reachability.experiments.arch19
import qronos.reachability.experiments.continuization
import qronos.lis.experiments.timing_stability_ifac20
import qronos.lis.experiments.abstraction_timing
# Interface for the experiment modules:
# - main(argv) runs the experiments, usually main(sys.argv[1:]).
# - A description of each experiment is read from the module docstring.
EXPERIMENTS = {
        'ifac20': qronos.lis.experiments.timing_stability_ifac20,
        'arch19': qronos.reachability.experiments.arch19,
        'cont': qronos.reachability.experiments.continuization,
        'abs': qronos.lis.experiments.abstraction_timing,
        }

def print_help():
    print("NOTE: To run a specific experiments, please specifiy ONE of the following experiments as first commandline parameter: " + " ".join(EXPERIMENTS.keys()))
    print("NOTE: If no arguments are given, all experiments will be run.")
    print("NOTE: Some experiments support further commandline options -- specify the name of the experiment and then --help")
    print("Details:")
    def indent(lines):
        return "\n".join([ " " * 8 + line  for line in lines.splitlines()])
    for key in EXPERIMENTS:
        print(key + ":    \n" + indent(EXPERIMENTS[key].__doc__))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("NOTE: No experiment has been specified. ALL experiments will be run, which may take quite long.")
        print_help()
        time.sleep(5)
        for exp in EXPERIMENTS.values():
            exp.main([])
    else:
        if sys.argv[1] not in EXPERIMENTS.keys():
            print("Unknown experiment name " + repr(sys.argv[1]))
            print_help()
            sys.exit(1)
        exp = EXPERIMENTS[sys.argv[1]]
        exp.main(sys.argv[2:])
