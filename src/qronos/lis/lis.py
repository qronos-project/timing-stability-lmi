#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Linear Impulsive System (LIS) model of digital control loop
"""
import numpy as np
import scipy.linalg
import mpmath as mp
from mpmath import iv
from .generic_matrix import blockmatrix, eye, zeros, check_datatype, convert
from repoze.lru import lru_cache


class LISControlLoop(object):
    """
    Linear Impulsive System representation of digital control loop
    """
    def __init__(self, s, datatype):
        """
        Construct a LIS from a digital control loop
        
        NOTE: All parameters must not be modified later, as results are cached.
        
        NOTE: Unlike in the publication (Gaukler et al. (2019/2020): Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing. Submitted for publication),
        the event matrices are here called "A" instead of "E",
        and the selector matrix is called "E" instead of "S".
        """
        datatype = check_datatype(datatype)
        self.datatype = datatype
        self.blocks = ['x_p', 'x_d', 'y_d', 'u']
        s._check_and_update_dimensions()
        self.blocklengths = [s.n_p, s.n_d, s.p, s.m]
        self.n = n = sum(self.blocklengths)
        def E(i, dim):
            """
            construct matrix M (dim x dim) which is zero except for M[i,i]=1,
            Note: For consistency with numpy/mpmath, indices count from 0!
            """
            M = zeros(dim, dim, datatype)
            M[i,i] = 1
            return M
        def bm(M):
            """
            construct a block matrix
            """
            return blockmatrix(M, self.blocklengths, datatype)
        self.A_cont = bm([[s.A_p, 0, 0, s.B_p], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.A_u = [ eye(n, datatype) + bm([[0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, E(i - 1, s.m) * convert(s.C_d, datatype), 0, -E(i - 1, s.m)]]) 
                     for i in range(1, s.m + 1) ]
        self.A_y = [ eye(n, datatype) + bm([[0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [E(i - 1, s.p) * convert(s.C_p, datatype), 0, -E(i - 1, s.p), 0],
                                            [0, 0, 0, 0]])
                     for i in range(1, s.p + 1) ]
        self.A_ctrl = bm([[1, 0, 0, 0],
                          [0, s.A_d, s.B_d, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.sys = s
        # The following formula is the LIS discretization for nominal timing, A(delta_t=0)
        self.Ak_nominal = self.A_ctrl * iv.expm(self.A_cont * s.T/2) * (sum(self.A_u + self.A_y) + eye(n, datatype) * (1 - len(self.A_u) - len(self.A_y)))  * iv.expm(self.A_cont * s.T/2)                                                                

    @lru_cache(1)
    def expm_a_t_half(self):
        """
        expm(A*T/2)
        """
        return iv.expm(self.A_cont * self.sys.T/2)
    
    @lru_cache(1)
    def m1_a_m2_tau(self):
        """
        list of all m1_a_m2_*(), which represent all Delta_A_...:
        Delta_A_... = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        
        (M1, A, M2, tau, info_string) = m1_a_m2_tau()[i]
        
        info_string is a textual identifier
        """
        return ([self.m1_a_m2_tau_u(i) for i in range(self.sys.m)] + 
               [self.m1_a_m2_tau_y(i) for i in range(self.sys.p)] +
               sum([[self.m1_a_m2_tau_uy(i, j) for i in range(self.sys.m)] for j in range(self.sys.p)], [])) # note that 'sum' here is for joining lists
    
    @lru_cache(99)
    def m1_a_m2_tau_u(self, u_index=None):
        """
        constituent parts of Delta_A_u_i = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_u(i)
        info_string is a textual identifier
        """
        return (self.A_ctrl * self.expm_a_t_half(),
                self.A_cont,
                self.A_u[u_index] - eye(self.n, datatype=iv),
                max(abs(self.sys.delta_t_u_min[u_index]), abs(self.sys.delta_t_u_max[u_index])),
                "u {}".format(u_index))
    
    @lru_cache(99)
    def m1_a_m2_tau_y(self, y_index):
        """
        constituent parts of Delta_A_y_i = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_y(i)
        info_string is a textual identifier
        """
        return (self.A_ctrl * (self.A_y[y_index] - eye(self.n, datatype=iv)) * self.expm_a_t_half(),
                self.A_cont,
                eye(self.n, datatype=iv),
                max(abs(self.sys.delta_t_y_min[y_index]), abs(self.sys.delta_t_y_max[y_index])),
                "y {}".format(y_index))
    
    @lru_cache(999)
    def m1_a_m2_tau_uy(self, u_index, y_index):
        """
        constituent parts of Delta_A_uy_i_j = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_uy(i,j)
        info_string is a textual identifier
        """
        # tau = max(delta t_u - delta t_y)
        dtu = mp.mpi(self.sys.delta_t_u_min[u_index], self.sys.delta_t_u_max[u_index])
        dty = mp.mpi(self.sys.delta_t_y_min[y_index], self.sys.delta_t_y_max[y_index])
        tau = (dty - dtu).b # max(dty_j - dtu_i) for the given intervals of dty_j and dtu_i
        if tau < 0:
            tau = 0
        return (self.A_ctrl * (self.A_y[y_index] - eye(self.n, datatype=iv)),
                self.A_cont,
                self.A_u[u_index] - eye(self.n, datatype=iv),
                tau,
                "u {} combined with y {}".format(u_index, y_index))
    
    def Ak_delta_to_nominal_approx(self, dtu, dty):
        assert len(dtu) == self.sys.m
        assert len(dty) == self.sys.p
        D = np.zeros((self.n, self.n))
        def delta_A(m1_a_m2, tau):
            M1, A, M2, _, _ = m1_a_m2
            M1 = convert(M1, np)
            M2 = convert(M2, np)
            A = convert(A, np)
            tau = float(tau)
            return M1.dot(scipy.linalg.expm(A*tau) - np.eye(len(A))).dot(M2)
        for i in range(self.sys.m):
            D = D + delta_A(self.m1_a_m2_tau_u(i), tau=dtu[i])
        for i in range(self.sys.p):
            D = D + delta_A(self.m1_a_m2_tau_y(i), tau=dty[i])
        for i in range(self.sys.m):
            for j in range(self.sys.p):
                delta = dty[j] - dtu[i]
                if (delta > 0):
                    D = D + delta_A(self.m1_a_m2_tau_uy(i, j), tau=delta)
        return D
