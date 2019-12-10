#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import unittest
import mpmath as mp
from mpmath import iv
import numpy as np
from .lis import LISControlLoop
from qronos import examples

"""
tests for Linear Impulsive System (LIS) model of digital control loop
"""

class Tests(unittest.TestCase):
    """
    unit tests (starting with test_) and helper functions
    """
    def test_dtype_iv(self):
        self.check(dtype=iv)
    def test_numpy(self):
        self.check(dtype=np)
    def check_type_matrix(self, M, dtype):
        if dtype == iv:
            self.assertIsInstance(M, iv.matrix)
        else:
            self.assertIsInstance(M, np.ndarray)
    def check_type_scalar(self, variable, dtype):
        if dtype == iv:
            self.assertIsInstance(variable, (float, mp.ctx_iv.ivmpf)) # FIXME: inconsistent return type
        else:
            self.assertIsInstance(variable, float)
    def check(self, dtype):
        c = LISControlLoop(examples.example_A1_stable_1(), datatype=dtype)
        for (M1, A, M2, tau, info_string) in c.m1_a_m2_tau():
            self.check_type_matrix(M1, dtype)
            self.check_type_matrix(A, dtype)
            self.check_type_matrix(M2, dtype)
            self.check_type_scalar(tau, dtype)
            self.assertIsInstance(info_string, str)


if __name__=="__main__":
    unittest.main()
