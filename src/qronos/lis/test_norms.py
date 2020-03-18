#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for norms.py
"""

import mpmath as mp
from mpmath import iv
import unittest
from .norms import iv_spectral_norm, iv_spectral_norm_rough, iv_matrix_to_numpy_ndarray, iv_matrix_mid_as_mp, approx_P_norm, approx_P_norm_expm, iv_P_norm, iv_P_norm_expm, approx_P_sqrt_T
from .iv_matrix_utils import matrix_abs_max
import scipy
import random

class Tests(unittest.TestCase):
    """
    unittests (starting with test_) and helper functions for norms.py
    """
    def assertInInterval(self, value, interval, relativeTolerance=0):
        """
        assert that value is in the given interval
        
        @param relativeTolerance: widen the interval by this factor
        """
        self.assertIn(value, interval * (1 + iv.mpf([-1, +1]) * mp.mpf(relativeTolerance)))
        
    def example_matrices(self, include_singular=True, random=100):
        examples = [iv.matrix([[1,2],[4,5]]) + iv.mpf([-1,+1])*mp.mpf(1e-10),
                    iv.eye(2) * 0.5,
                    iv.diag([1, 1e-10]),
                    iv.eye(2) * 1e-302
                    ]
        for n in [1,4,17]:
            for i in range(random // n):
                examples.append(iv.matrix(mp.randmatrix(n)))
        if include_singular:
            examples.append(iv.matrix(mp.zeros(4)))
            examples.append(iv.diag([1, 1e-200]))
        return examples
    
    def test_matrix_abs_max(self):
        A = iv.matrix([[1,2],[4,5]]) + iv.mpf([-1,+1])*mp.mpf(1e-10)
        assert mp.matrix([[1,2],[4,5]]) + mp.mpf(1e-10) == matrix_abs_max(A)
        assert mp.matrix([[1,2],[4,5]]) + mp.mpf(1e-10) == matrix_abs_max(-A)
        
    
    def test_P_norm_and_spectral_norm(self, A=None, P_sqrt_T=None):
        if A is None:
            # call this test for some examples
            for A in self.example_matrices(include_singular=True):
                self.check_spectral_norm(A)
                for P_sqrt_T in self.example_matrices(include_singular=False, random=3):    
                    if len(P_sqrt_T) != len(A):
                        continue
                    self.test_P_norm_and_spectral_norm(A, P_sqrt_T)
                    self.check_P_norm(A, P_sqrt_T)
        else:
            assert P_sqrt_T is not None
            # actual test for given A, P_sqrt_T
            M1 = mp.randmatrix(len(A))
            M2 = mp.randmatrix(len(A))
            tau = 0.01
            self.check_P_norm_expm(P_sqrt_T, M1, A, M2, tau)

    def check_spectral_norm(self, M, tolerance = 1e-10):
        """
        check iv_spectral_norm() and iv_spectral_norm_rough() against a numerical result from scipy.
        
        @param tolerance: assumed guaranteed relative tolerance on scipy's result
        """
        approx_spectral_norm = scipy.linalg.norm(iv_matrix_to_numpy_ndarray(iv_matrix_mid_as_mp(M)), 2)
        self.assertIn(approx_spectral_norm, iv_spectral_norm(M) * (1 + tolerance * mp.mpi(-1, 1)))
        self.assertIn(approx_spectral_norm, iv_spectral_norm_rough(M) * (1 + tolerance * mp.mpi(-1, 1)))
    
    def check_P_norm(self, M, P_sqrt_T, tolerance = 1e-6):
       """
       check iv_P_norm() against approx_P_norm() for given matrices
       
       @param tolerance: assumed guaranteed relative tolerance of approx_P_norm()
       """
       self.assertInInterval(approx_P_norm(M, P_sqrt_T), interval=iv_P_norm(M, P_sqrt_T), relativeTolerance=tolerance)

    def check_P_norm_expm(self, P_sqrt_T, M1, A, M2, tau, tolerance=1e-10):
        """
        check iv_P_norm_expm() against approx_P_norm_expm() for given matrices
        
        @param tolerance: assumed guaranteed relative tolerance of approx_P_norm_expm()
        """
        pn_approx = approx_P_norm_expm(P_sqrt_T, M1, A, M2, tau)
        pn_interval = iv_P_norm_expm(P_sqrt_T, M1, A, M2, tau)
        self.assertInInterval(pn_approx, pn_interval, relativeTolerance=tolerance)
        
    def test_P_synthesis(self):
        """
        Combined test:
        For A with eigenvalues inside the unit disk,
        generate P_sqrt_T such that P_norm(A, P_sqrt_T) < 1.
        """
        for c in [0.001234, 1, -2, 42.123]:
            eigenvalue = 0.99
            # rotation matrix with eigenvalue magnitude 0.99,
            # transformed with factor c (c=1: invariant set is a circle, otherwise: invariant set is elliptical)
            A = iv.matrix([[0., -c*eigenvalue], [eigenvalue/c, 0]])
            # Note that A must be well-conditioned, otherwise this test will fail
            # compute eigenvalues of A
            eigv_A, _= mp.eig(iv_matrix_mid_as_mp(A))
            self.assertAlmostEqual(eigenvalue, max([abs(i) for i in eigv_A]), 3)
            P_sqrt_T = approx_P_sqrt_T(A)
            P_norm_iv = iv_P_norm(A, P_sqrt_T)
            self.check_P_norm(A, P_sqrt_T)
            # P_norm_iv must be at least 0.99, but should not be much larger than that
            self.assertLess(P_norm_iv, 1)
            self.assertGreater(P_norm_iv.b, 0.99)
            self.check_P_norm_expm(P_sqrt_T, M1=mp.randmatrix(2), A=A, M2=mp.randmatrix(2), tau=0.01)


if __name__ == "__main__":
    random.seed(42)
    unittest.main() 
