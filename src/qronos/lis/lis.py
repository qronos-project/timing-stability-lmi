#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Linear Impulsive System (LIS) model of digital control loop
"""
import numpy as np
import mpmath as mp
from mpmath import iv
from .generic_matrix import check_datatype, AbstractMatrix
from repoze.lru import lru_cache
from deprecation import deprecated
from numbers import Number

# FIXME monkey-patching...
mp.ctx_iv.ivmpf.__float__ = lambda self: float(mp.mpf(self))
def interval_intersects(x: Number, y: Number) -> bool:
    x = iv.convert(x)
    y = iv.convert(y)
    return (x.a in y or x.b in y or y.a in x or y.b in x)
mp.ctx_iv.ivmpf.intersects = interval_intersects

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
        self.abstract_matrix_type = AbstractMatrix.from_type(datatype)
        d = self.abstract_matrix_type
        self.blocks = ['x_p', 'x_d', 'y_d', 'u']
        s._check_and_update_dimensions()
        self.blocklengths = [s.n_p, s.n_d, s.p, s.m]
        self.n = n = sum(self.blocklengths)
        def E(i, dim):
            """
            construct matrix M (dim x dim) which is zero except for M[i,i]=1,
            Note: For consistency with numpy/mpmath, indices count from 0!
            """
            M = d.zeros(dim, dim)
            M[i,i] = 1
            return M
        def bm(M):
            """
            construct a block matrix
            """
            return d.blockmatrix(M, self.blocklengths)
        self.A_cont = bm([[s.A_p, 0, 0, s.B_p], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.A_u = [ d.eye(n) + bm([[0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, E(i - 1, s.m) @ d.convert(s.C_d), 0, -E(i - 1, s.m)]])
                     for i in range(1, s.m + 1) ]
        self.A_y = [ d.eye(n) + bm([[0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [E(i - 1, s.p) @ d.convert(s.C_p), 0, -E(i - 1, s.p), 0],
                                            [0, 0, 0, 0]])
                     for i in range(1, s.p + 1) ]
        self.A_ctrl = bm([[1, 0, 0, 0],
                          [0, s.A_d, s.B_d, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        self.sys = s
        # The following formula is the LIS discretization for nominal timing, A(delta_t=0)
        self.Ak_nominal = self.A_ctrl @ d.expm(self.A_cont * s.T/2) @ (sum(self.A_u + self.A_y) + d.eye(n) * (1 - len(self.A_u) - len(self.A_y))) @ d.expm(self.A_cont * s.T/2)

    @lru_cache(1)
    def expm_a_t_half(self):
        """
        expm(A*T/2)
        """
        d = self.abstract_matrix_type
        return d.expm(self.A_cont * self.sys.T/2)

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
    def m1_a_m2_tau_u(self, u_index):
        """
        constituent parts of Delta_A_u_i = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_u(i)
        info_string is a textual identifier
        """
        d = self.abstract_matrix_type
        return (self.A_ctrl @ self.expm_a_t_half(),
                -self.A_cont,
                self.A_u[u_index] - d.eye(self.n),
                max(abs(self.sys.delta_t_u_min[u_index]), abs(self.sys.delta_t_u_max[u_index])),
                "u {}".format(u_index))

    @lru_cache(99)
    def m1_a_m2_tau_y(self, y_index):
        """
        constituent parts of Delta_A_y_i = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_y(i)
        info_string is a textual identifier
        """
        d = self.abstract_matrix_type
        return (self.A_ctrl @ (self.A_y[y_index] - d.eye(self.n)) @ self.expm_a_t_half(),
                self.A_cont,
                d.eye(self.n),
                max(abs(self.sys.delta_t_y_min[y_index]), abs(self.sys.delta_t_y_max[y_index])),
                "y {}".format(y_index))

    @lru_cache(999)
    def m1_a_m2_tau_uy(self, u_index, y_index):
        """
        constituent parts of Delta_A_uy_i_j = M1 * (exp(A*t)-I) * M2,  |t|<tau:
        (M1, A, M2, tau, info_string) = m1_a_m2_tau_uy(i,j)
        info_string is a textual identifier
        """
        d = self.abstract_matrix_type
        # tau = max(delta t_u - delta t_y) maximized over the range delta_t_u_min ... _max and delta_t_y_min ... _max
        dtu = mp.mpi(self.sys.delta_t_u_min[u_index], self.sys.delta_t_u_max[u_index])
        dty = mp.mpi(self.sys.delta_t_y_min[y_index], self.sys.delta_t_y_max[y_index])
        tau = (dty - dtu).b # max(dty_j - dtu_i) for the given intervals of dty_j and dtu_i
        tau = d.convert_scalar(tau) # convert to float if required
        if tau < 0:
            tau = 0
        return (self.A_ctrl @ (self.A_y[y_index] - d.eye(self.n)),
                self.A_cont,
                self.A_u[u_index] - d.eye(self.n),
                tau,
                "u {} combined with y {}".format(u_index, y_index))

    def rho_total(self, P_sqrt_T, scale_delta_t=1, datatype=None, verbose=False):
        '''
        Return (rho_total, rho_nominal)

        rho_total: upper P-norm-bound for A_k(delta_t) for all possible delta_t
        rho_nominal: same as rho_total, but for delta_t == 0

        @param P_sqrt_T: matrix for CQLF, see AbstractMatrix.
        @param float scale_delta_t: increase timing by given amount
        '''
        d = AbstractMatrix.from_type(datatype or self.datatype)
        rho_nominal = d.P_norm(M=self.Ak_nominal, P_sqrt_T=P_sqrt_T)
        if verbose:
            print('P_norm(A) = ', rho_nominal)
        rho_total = rho_nominal
        for [M1, A_cont, M2, tau, info] in self.m1_a_m2_tau():
            pnorm_exp = d.P_norm_expm(P_sqrt_T, M1=M1, A=A_cont, M2=M2, tau=tau * scale_delta_t)
            rho_total += pnorm_exp
            if verbose:
                print(info + ":")
                print('P_norm(...expm(...)) = ', pnorm_exp)
        if verbose:
            if rho_nominal >= 1:
                margin = float('inf')
            else:
                margin = (rho_total - rho_nominal) / (1 - rho_nominal)
            print(f'total:\nsum(P_norm...) = {rho_total}, ({margin*100:.1f}% of stability reserve used by timing)')
        return (rho_total, rho_nominal)

    @deprecated()
    def Ak_delta_to_nominal_approx(self, dtu, dty, method=None, datatype=None):
        return self.Ak_delta_to_nominal(dtu=dtu, dty=dty, method=method, datatype=np)

    def Ak_delta_to_nominal(self, dtu, dty, method=None, datatype=None):
        '''
        Return A_k(delta_t) - A_k(delta_t=0) for given delta_t.

        dtu: delta_t for u
        dty: delta_t for y
        method: 'sum' (default): Use the Decomposition theorem from arXiv:1911.02537.
                'impulsive' (only as a reference for tests): Discretize (simulate) the Linear Impulsive System
        '''
        assert len(dtu) == self.sys.m
        assert len(dty) == self.sys.p
        datatype = datatype or self.datatype
        datatype = check_datatype(datatype)
        d = AbstractMatrix.from_type(datatype)

        if method is None:
            method = 'sum'
        assert method in ['sum', 'impulsive']
        if method == 'sum':
            D = d.zeros(self.n, self.n)
            def delta_A(m1_a_m2, tau):
                if tau == 0:
                    return d.zeros(self.n, self.n)
                M1, A, M2, _, _ = m1_a_m2
                M1 = d.convert(M1)
                M2 = d.convert(M2)
                A = d.convert(A)
                tau = d.convert_scalar(tau)
                return M1 @ (d.expm(A*tau) - d.eye(len(A))) @ M2
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
        elif method == 'impulsive':
            time = -self.sys.T/2
            A = d.eye(self.n)
            events = [(dtu[i], self.A_u[i]) for i in range(self.sys.m)]
            events += [(dty[i], self.A_y[i]) for i in range(self.sys.p)]
            events += [(self.sys.T/2, self.A_ctrl)]
            events.sort(key=lambda ev: ev[0]) # sort events by time
            for next_time, A_event in events:
                next_time = float(next_time)
                assert next_time >= time
                A = d.convert(A_event) @ d.expm(d.convert(self.A_cont) * (d.convert_scalar(next_time) - d.convert_scalar(time))) @ d.convert(A)
                time = next_time
            assert time == self.sys.T/2
            D = A - d.convert(self.Ak_nominal)
            return D