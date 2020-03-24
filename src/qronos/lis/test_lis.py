#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import unittest
import mpmath as mp
from mpmath import iv
import numpy as np
from .lis import LISControlLoop
from . import generic_matrix
from qronos import examples
import scipy
import scipy.linalg
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

    def assertIntervalIntersects(self, x: mp.ctx_iv.ivmpf, y: mp.ctx_iv.ivmpf):
        self.assertTrue(x.a in y or x.b in y or y.a in x or y.b in x)

    def test_assertMatrixAlmostEqual(self):
        def test_rel_tolerance(tol, matrixClass):
            """
            test with given relative deviation
            """
            lower = 1e3 * matrixClass.eye(4)

            if matrixClass == generic_matrix.IntervalMatrix:
                upper = lower * mp.mpi(1 - tol / 2, 1 + tol / 2)
            else:
                upper = lower * (1 + tol)
            self.assertMatrixAlmostEqual(lower, upper)
        def test_abs_tolerance(tol, matrixClass):
            """
            test with given absolute deviation
            """
            lower = 1e-3 * matrixClass.eye(4)
            if matrixClass == generic_matrix.IntervalMatrix:
                upper = lower + matrixClass.ones(4, 4) * mp.mpi(-tol / 2, tol / 2)
            else:
                upper = lower + tol * matrixClass.ones(4, 4)
            self.assertMatrixAlmostEqual(lower, upper)
        for matrixClass in [generic_matrix.NumpyMatrix, generic_matrix.IntervalMatrix]:
            with self.assertRaises(AssertionError):
                test_rel_tolerance(1.1 * 1e-6, matrixClass)
            with self.assertRaises(AssertionError):
                test_abs_tolerance(1.1 * 1e-7, matrixClass)
            test_rel_tolerance(0.99 * 1e-6, matrixClass)
            test_abs_tolerance(0.99 * 1e-7, matrixClass)
        with self.assertRaises(TypeError):
            self.assertMatrixAlmostEqual(True, True)

    def assertMatrixAlmostEqual(self, A, B, abs_tolerance=1e-7, rel_tolerance=1e-6):
        """
        Check that two matrices are equal up to numerical precision

        For interval arithmetic, this tests that the intervals have nonzero intersection and their width is small.
        """
        self.assertEqual(type(A), type(B))

        if type(A) == np.ndarray:
                np.testing.assert_allclose(A, B, rtol=rel_tolerance, atol=abs_tolerance)
        elif type(A) == iv.matrix:
            for i in range(A.rows):
                for j in range(A.cols):
                    self.assertIntervalIntersects(A[i,j], B[i,j])
                    self.assertIntervalWidthIsSmall(A[i,j], abs_tolerance, rel_tolerance)
                    self.assertIntervalWidthIsSmall(B[i,j], abs_tolerance, rel_tolerance)
        else:
            raise TypeError("unknown type")

    def test_simulate(self):
        """
        test LISControlLoop.simulate_random() with a specific example
        for which the result can be computed by hand
        """
        c=LISControlLoop(examples.example_E_timer(), np)
        x0 = np.zeros(c.n)
        x0[0] = -0.5 # initial time (defined such that the nominal sampling is at t=kT, here T=1)
        x0[1] = 1
        L=6
        delta_t_u=np.array([[.1, .2, .3, -.2, -.1, 0]])
        delta_t_y=np.array([[0, 0.3, -.1, .2, .1, float('NaN')]]) # last value doesn't matter
        skip_u=np.array([[False, False, False, False, False, True]])
        skip_y=np.array([[False, True, False, False, False, False]])
        skip_ctrl=np.array([False, False, True, False, False, False])
        (xk, Ak) = c.simulate_random(L, x0, delta_t_u=delta_t_u, delta_t_y=delta_t_y, skip_u=skip_u, skip_y=skip_y, skip_ctrl=skip_ctrl)
        print(repr(xk), repr(Ak))

        recent_y = 0
        recent_ctrl = 0
        recent_u = 0
        for k in range(L):
            # x_p[0]: counter state
            self.assertEqual(xk[0, k], k + x0[0])
            # x_p[0]: constant 1
            self.assertEqual(xk[1, k], 1)
            # x_d: sampling time of the value y at the end of the last period in which the controller wasn't skipped
            self.assertEqual(xk[2, k], recent_ctrl, xk)
            # y_d: sampling time of y
            self.assertEqual(xk[3, k], recent_y, xk)
            # u_d: sampling time of the value y that was used for computing u
            self.assertEqual(xk[4, k], recent_u, xk)
            if not skip_u[0, k]:
                recent_u = recent_ctrl
            if not skip_y[0, k]:
                recent_y = k + delta_t_y[0, k]
            if not skip_ctrl[k]:
                recent_ctrl = recent_y
        print(Ak.shape)
        for k in range(L - 1):
            self.assertLess(scipy.linalg.norm(Ak[k, :, :] @ xk[:, k] - xk[:, k + 1], 2), 1e-9)

    def test_state_names(self):
        c = LISControlLoop(examples.example_A1_stable_1(), np)
        self.assertEqual(list(c.state_names()), ["x_p[0]", "x_d[0]", "y_d[0]", "u[0]"])

if __name__ == "__main__":
    unittest.main()
