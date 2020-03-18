#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for memoize_simple.py
"""

import mpmath as mp
import numpy as np
from .memoize_simple import matrix_memoize_simple, matrix_equals
import unittest

class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions
    """
    
    def test_matrix_equals(self):
        self.assertTrue(matrix_equals(np.ones(14), np.ones(14)))
        self.assertTrue(matrix_equals(np.zeros((12, 7)), np.zeros((12, 7))))
        self.assertFalse(matrix_equals(np.zeros((12, 7)), np.zeros((12, 6))))
        self.assertFalse(matrix_equals(np.zeros((12, 7)), np.zeros((11, 7))))
        self.assertTrue(matrix_equals(mp.iv.eye(14), mp.iv.eye(14)))
        self.assertTrue(matrix_equals(mp.iv.zeros(12, 7), mp.iv.zeros(12, 7)))
        self.assertFalse(matrix_equals(mp.iv.zeros(12, 7), mp.iv.zeros(12, 6)))
        self.assertFalse(matrix_equals(mp.iv.zeros(12, 7), mp.iv.zeros(11, 7)))
        
    
    def test_mp_memoize_simple(self):
        """
        test mp_memoize_simple for the evil case of mutable arguments
        """
        @matrix_memoize_simple
        def myfunction(M):
            """
            sum of matrix elements
            """
            if isinstance(M, np.ndarray):
                return sum(sum(M))
            else:
                assert isinstance(M, (mp.iv.matrix)), "type {}".format(type(M))
                return mp.fdot(M, mp.iv.ones(len(M)))
        for module in mp.iv, np:
            A = module.diag([1,2,3,4])
            Acopy = module.diag([1,2,3,4])
            B = module.diag([1,2,3,5])
            assert myfunction(B) == 11
            assert myfunction(A) == 10
            assert myfunction(A) == 10
            A[1,3] = 42
            assert myfunction(A) == 52 # would fail if the cache stored A by-reference instead of a by-value copy.
            assert myfunction(Acopy) == 10
            assert myfunction(A) == 52
