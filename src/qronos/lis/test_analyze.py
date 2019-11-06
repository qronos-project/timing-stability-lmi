#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for analyze.py
"""


from .. import examples
from .analyze import analyze
import unittest

class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions
    """
    def test_analyze(self):
        result = analyze(examples.example_C_quadrotor_attitude_one_axis())
        rho = result['rho']
        self.assertLess(rho, 0.98)
        self.assertGreater(rho, 0.78)
        self.assertAlmostEqual(result['rho'], result['rho_approx'], places=4)
