#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for analyze.py
"""


from .. import examples
from .analyze import analyze, analyze_cqlf_skip, analyze_cqlf_skip_condition, analyze_cqlf_timing_range
import unittest
import numpy as np
from mpmath import iv

def example_C():
    """
    DigitalControlLoop and P_sqrt_T value for example C.
    """
    s = examples.example_C_quadrotor_attitude_one_axis()
    # P_sqrt_T = analyze(s, datatype=np)['P_sqrt_T']
    P_sqrt_T = iv.matrix(
        [['0.0019244615174829', '0.010762303718393', '-0.000842841208611818', '-0.000546543030195919', '0.337150563056583'],
         ['0.0', '0.0145387640958695', '0.000181643285522379', '0.000217863576839351', '0.414793996245936'],
         ['0.0', '0.0', '0.443015131922611', '-0.443014979405948', '0.000266278043438505'],
         ['0.0', '0.0', '0.0', '0.000392820699869412', '0.432006259021524'],
         ['0.0', '0.0', '0.0', '0.0', '0.662735066069117']])
    return (s, P_sqrt_T)

class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions
    """

    def test_analyze(self):
        self.check_analyze(np)
        self.check_analyze(iv)

    def check_analyze(self, datatype):
        result = analyze(examples.example_C_quadrotor_attitude_one_axis(), datatype=datatype)
        rho = result['rho_approx']
        self.assertLess(rho, 0.98)
        self.assertGreater(rho, 0.78)
        # compare with reference value which was computed 2019-12-11, to check against *future* implementation defects
        # (Note that we do not have a 'ground truth' reference result.)
        self.assertAlmostEqual(result['rho_approx'], 0.9138044519681982, places=8, msg="Result does not match previously saved numerical result. If the implementation was actually changed, then please adjust the reference value. Otherwise, this is an error.")
        if datatype == iv: # exact computation
            self.assertAlmostEqual(result['rho'], result['rho_approx'], places=4)
            self.assertAlmostEqual(result['rho'], 0.9138045433500117, places=8, msg="Result does not match previously saved numerical result. If the implementation was actually changed, then please adjust the reference value. Otherwise, this is an error.")

    def test_analyze_cqlf_skip(self):
        # we don't have any 'ground truth', therefore compare with results from 2020-03-09
        (s, P_sqrt_T) = example_C()
        result = analyze_cqlf_skip(s, P_sqrt_T)
        expected_result = {(True, (True,), (True,)): 3.499407408771806, (True, (True,), (False,)): 552.2615245016201, (True, (False,), (True,)): 2.236058237586597, (True, (False,), (False,)): 552.257897692413, (False, (True,), (True,)): 3.352792287285962, (False, (True,), (False,)): 2.3209198644774522, (False, (False,), (True,)): 2.135796217354262, (False, (False,), (False,)): 0.8261609094165097}
        for key in expected_result:
            self.assertAlmostEqual(expected_result[key], result[key])
        condition = lambda skip_ctrl, skip_u, skip_y: skip_ctrl and not any(skip_u) and all(skip_y)
        self.assertAlmostEqual(expected_result[(True, (False,), (True,))], analyze_cqlf_skip_condition(s, P_sqrt_T, condition))

    # analyze_cqlf_skip_condition is tested above in test_analyze_cqlf_skip.

    def test_analyze_cqlf_timing_range(self):
        # we don't have any 'ground truth', therefore compare with results from 2020-03-09
        (s, P_sqrt_T) = example_C()
        result = analyze_cqlf_timing_range(s, P_sqrt_T=P_sqrt_T, num_points=10)
        self.assertAlmostEqual(result[2], 1.3130694791287163)
        self.assertAlmostEqual(result[3], 0.08764354254819733)






if __name__ == "__main__":
    unittest.main()