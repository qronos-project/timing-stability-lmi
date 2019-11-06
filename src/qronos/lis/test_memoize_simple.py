#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for memoize_simple.py
"""

import mpmath as mp
import numpy as np
from .memoize_simple import matrix_memoize_simple
import unittest

class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions
    """
    def test_mp_memoize_simple(self):
        """
        test mp_memoize_simple for the evil case of mutable arguments
        """
        for module in mp, np:
            A = module.diag([1,2,3,4])
            Acopy = module.diag([1,2,3,4])
            B = module.diag([1,2,3,5])
            @matrix_memoize_simple
            def myfunction(M):
                """
                sum of matrix elements
                """
                if isinstance(M, np.ndarray):
                    return sum(sum(M))
                else:
                    assert isinstance(M, (mp.matrix)), "type {}".format(type(M))
                    return mp.fdot(M, mp.ones(len(M)))
            assert myfunction(B) == 11
            assert myfunction(A) == 10
            assert myfunction(A) == 10
            A[1,3] = 42
            assert myfunction(A) == 52 # would fail if the cache stored A by-reference instead of a by-value copy.
            assert myfunction(Acopy) == 10
            assert myfunction(A) == 52
