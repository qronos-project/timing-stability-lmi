#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Tests for lmi_cqlf.py
"""

from .lmi_cqlf import cqlf
from .norms import iv_P_norm
from .generic_matrix import convert, approx_max_abs_eig
from mpmath import iv
import numpy as np
import unittest



class Tests(unittest.TestCase):
    """
    unit tests (starting with test_) and helper functions
    """
    Ad=np.array([[  1.00000000e+00,  -2.00002213e+00,  -1.41419228e-01,
          0.00000000e+00,   5.53348311e+02],
       [  1.00000000e-02,   1.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   5.53348311e+00],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   5.53348311e+02],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   5.53348311e+02],
       [  0.00000000e+00,  -3.61440000e-03,  -2.55570000e-04,
          0.00000000e+00,   0.00000000e+00]])
    Ad2 = np.array([[  1.00000000e+00,  -3.00003320e+00,  -2.12128842e-01,
              0.00000000e+00,   2.76674155e+02],
           [  1.00000000e-02,   1.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   8.30022466e+00],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   8.30022466e+02],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   8.30022466e+02],
           [  0.00000000e+00,  -3.61440000e-03,  -2.55570000e-04,
              0.00000000e+00,   0.00000000e+00]])
    def test_cqlf(self):
        self.check_cqlf(R_inv=None)
        self.check_cqlf(R_inv=np.eye(len(self.Ad2)))
        self.check_cqlf(R_inv=np.diag([1,2,3,4,5]))
        self.check_cqlf(R_inv=np.diag([500,2,3,4,5]))        
        R_inv, _ = cqlf(A=self.Ad, Delta_list=[self.Ad2], rho=0.9, beta=0.9)
        self.check_cqlf(R_inv=R_inv)
        R_inv = self.check_cqlf(R_inv='auto')
        self.check_cqlf(R_inv=R_inv)


    def check_cqlf(self, R_inv, beta_scale=1):    
        Ad_list = [self.Ad, self.Ad2]
        eig = max([approx_max_abs_eig(A) for A in Ad_list])
#        print(eig)
        rho = eig * 1.1
        beta = eig * 0.25 * beta_scale # unusually high, the test problem is rather evil
        P_sqrt_T, _ = cqlf(A=Ad_list[0], Delta_list=[Ad_list[1]-Ad_list[0]], rho=rho, beta=beta, R_inv=R_inv)
        pnorm = iv_P_norm(convert(Ad_list[0], iv), convert(P_sqrt_T, iv))
        print(pnorm)
        assert pnorm < rho + 1e-3
        for Ad in Ad_list:
            pnorm = iv_P_norm(convert(Ad, iv), convert(P_sqrt_T, iv))
            print(pnorm)
            assert(pnorm < rho + beta + 1e-3)
        pnorm_delta = iv_P_norm(Ad_list[1]-Ad_list[0], P_sqrt_T)
        print(pnorm_delta)
        assert(pnorm_delta < beta + 1e-3)
        return P_sqrt_T
        

if __name__=="__main__":
    unittest.main()
