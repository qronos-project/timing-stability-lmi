#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for generic_matrix
"""
import unittest
import numpy as np
import mpmath as mp
from .generic_matrix import AbstractMatrix, NumpyMatrix, IntervalMatrix
import itertools
class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions for generic_matrix.py
    """

    # TODO further tests

    def test_qlf_bounds(self):
        """
        test qlf_upper_bound, qlf_lower_bound with (partly) random matrices
        """
        # TODO this test does not strictly check if the interval-arithmetic implementation is sound. It would also succeed if it was only correct within the numerical tolerance.
        np.random.seed(0)
        n = 10
        P_sqrt_T=[np.eye(n), np.diag([1]+[0.000001]*(n-1))]
        M=[None, np.eye(n), np.diag([1]+[0]*(n-1)), np.zeros((n,n))]

        for _ in range(5):
            P_sqrt_T.append(np.random.uniform(low=-3, high=3, size=(n, n)))
            M.append(np.random.uniform(low=-3, high=3, size=(n, n)))
        for (P_sqrt_T, M) in itertools.product(P_sqrt_T, M):
            for abstractmatrix in [NumpyMatrix, IntervalMatrix]:
                self.check_qlf_bounds(P_sqrt_T, M, abstractmatrix)

    def check_qlf_bounds(self, P_sqrt_T, M, abstractmatrix: AbstractMatrix):
        """
        test qlf_upper_bound, qlf_lower_bound for given matrices P_sqrt_T, M
        with random vectors x.
        """
        c1 = abstractmatrix.qlf_upper_bound(P_sqrt_T, M) # c1 sqrt(V(Mx)) <= |x|
        c2 = abstractmatrix.qlf_lower_bound(P_sqrt_T, M) # |Mx| <= c2 sqrt(V(x))

        def sqrt_V(x):
            """
            V(x) = x.T P_sqrt_T.T P_sqrt_T x.
            """
            return np.sqrt(x.T @ P_sqrt_T.T @ P_sqrt_T @ x)
        def mag(x):
            """
            |x|
            """
            return np.sqrt(np.sum(x**2))

        M_actual = M
        if M_actual is None:
            M_actual = np.eye(len(P_sqrt_T))

        RELTOL = 1e-10
        for i in range(100):
            x = np.random.uniform(low=-42, high=+42, size=len(P_sqrt_T))
            # c1 sqrt(V(Mx)) <= |x|
            if not mp.iv.isnan(c1) and c1 != mp.mpi('-inf', '+inf') and not mp.iv.isinf(c1):
                self.assertLessEqual(c1 * sqrt_V(M_actual @ x),  mag(x) * (1 + RELTOL))
            # |Mx| <= c2 sqrt(V(x))
            self.assertLessEqual(mag(M_actual @ x), c2 * sqrt_V(x)  * (1 + RELTOL))


if __name__ == "__main__":
    unittest.main()