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
        self.check_return_types(dtype=iv)
        
    def test_numpy(self):
        self.check_return_types(dtype=np)
        
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
            
    def check_return_types(self, dtype):
        c = LISControlLoop(examples.example_A1_stable_1(), datatype=dtype)
        for (M1, A, M2, tau, info_string) in c.m1_a_m2_tau():
            self.check_type_matrix(M1, dtype)
            self.check_type_matrix(A, dtype)
            self.check_type_matrix(M2, dtype)
            self.check_type_scalar(tau, dtype)
            self.assertIsInstance(info_string, str)
            
    def test_Ak_delta_to_nominal(self):
        """
        Test that Ak_delta_to_nominal_approx returns identical results for the two available methods of computation.
        
        This validates the "Decomposition" theorem in [arXiv:1911.02537], which is used only for one of the methods.
        """
        self.check_Ak_delta_to_nominal(np)
        self.check_Ak_delta_to_nominal(iv)
        
    def check_Ak_delta_to_nominal(self, dtype):
        """
        test Ak_delta_to_nominal for the given data type (np: numpy or iv: mpmath interval arithmetic)
        """
        for (example, iterations) in [ 
                (examples.example_A1_stable_1(), 100),
                (examples.example_B1_stable_3(), 50),
                (examples.example_C_quadrotor_attitude_one_axis(), 50),
                (examples.example_D_quadrotor_attitude_three_axis(), 10)
                ]:
            c = LISControlLoop(example, datatype=dtype)
            np.random.seed(0)
            if dtype == iv:
                iterations = iterations // 5 # keep the test runtime at a reasonable level, as interval arithmetic is way slower
            for i in range(iterations):
                dtu = (1 - 2*np.random.uniform(size=(c.sys.m, ))) * c.sys.T/2 * 0.9999
                dty = (1 - 2*np.random.uniform(size=(c.sys.p, ))) * c.sys.T/2 * 0.9999
                Ak_delta_decomposition = c.Ak_delta_to_nominal(dtu, dty, 'sum', datatype=dtype)
                Ak_delta_reference = c.Ak_delta_to_nominal(dtu, dty, 'impulsive', datatype=dtype)
                self.assertMatrixAlmostEqual(Ak_delta_decomposition, Ak_delta_reference)

    def assertIntervalWidthIsSmall(self, x, abs_tolerance=1e-7, rel_tolerance=1e-6):
        self.assertIsInstance(x, mp.ctx_iv.ivmpf)
        self.assertLess(float(x.delta), float(abs(x).b) * rel_tolerance + abs_tolerance)
    
    def test_assertMatrixAlmostEqual(self):
        def test_rel_tolerance(tol):
            """
            test with given relative deviation
            """
            self.assertMatrixAlmostEqual(1e3 * np.eye(4), 1e3 * np.eye(4) * (1 + tol))
        def test_abs_tolerance(tol):
            """
            test with given absolute deviation
            """
            self.assertMatrixAlmostEqual(1e-3 * np.eye(4), 1e-3 * np.eye(4) + tol * np.ones((4,4)))
        with self.assertRaises(AssertionError):
            test_rel_tolerance(1.1 * 1e-6)
            test_abs_tolerance(1.1 * 1e-7)
        test_rel_tolerance(0.99 * 1e-6)
        test_abs_tolerance(0.99 * 1e-7)

    def assertMatrixAlmostEqual(self, A, B, abs_tolerance=1e-7, rel_tolerance=1e-6):
        """
        Check that two matrices are equal up to numerical precision
        
        For interval arithmetic, this tests that the intervals have nonzero intersection and their width is small.
        """
        assert type(A) == type(B)            
        
        if type(A) == np.ndarray:
                np.testing.assert_allclose(A, B, rtol=rel_tolerance, atol=abs_tolerance)
        elif type(A) == iv.matrix:
            for i in range(A.rows):
                for j in range(A.cols):
                    self.assertTrue(A[i,j].intersects(B[i,j]))
                    self.assertIntervalWidthIsSmall(A[i,j], abs_tolerance, rel_tolerance)
                    self.assertIntervalWidthIsSmall(B[i,j], abs_tolerance, rel_tolerance)
        else:
            assert False, "invalid type"

if __name__=="__main__":
    unittest.main()
