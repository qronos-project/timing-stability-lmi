#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for analyze.py
"""


from .. import examples
from .analyze import analyze
import unittest
import numpy as np
from mpmath import iv

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
        # compare with reference value which was computed 2019-12-10, to check against *future* implementation defects
        # (Note that we do not have a 'ground truth' reference result.)
        self.assertAlmostEqual(result['rho_approx'], 0.9138042779113741, places=8, msg="Result does not match previously saved numerical result. If the implementation was actually changed, then please adjust the reference value. Otherwise, this is an error.")
        if datatype == iv: # exact computation
            self.assertAlmostEqual(result['rho'], result['rho_approx'], places=4)
            self.assertAlmostEqual(result['rho'], 0.913804369292595, places=8, msg="Result does not match previously saved numerical result. If the implementation was actually changed, then please adjust the reference value. Otherwise, this is an error.")

if __name__ == "__main__":
    unittest.main()