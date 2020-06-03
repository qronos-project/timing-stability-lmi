#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Linear Impulsive System (LIS) model of digital control loop
"""
import numpy as np
import mpmath as mp
from .generic_matrix import check_datatype, AbstractMatrix
from repoze.lru import lru_cache

class LISControlLoop(object):
    """
    Linear Impulsive System representation of digital control loop
    """
    def __init__(self, s, datatype=None):
        """
        Construct a LIS from a digital control loop

        @param DigitalControlLoop s
        @param datatype: datatype for computations -- mpmath.iv (interval arithmetic) or numpy (standard float, default)

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

    def state_names(self):
        """
        list of human-readable state names
        """
        for (block, length) in zip(self.blocks, self.blocklengths):
            for i in range(length):
                yield block + f'[{i}]'

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

        rho_total: upper bound for P-ellipsoid norm of A_k(delta_t) for all possible delta_t
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

    def Ak_delta_to_nominal(self, dtu=None, dty=None, method=None, datatype=None, skip_u=None, skip_y=None, skip_ctrl=False):
        '''
        Return A_k(delta_t) - A_k(delta_t=0) for given delta_t.

        dtu: delta_t for u
        dty: delta_t for y
        skip_...: skip actuation/sampling/controller event. The corresponding delta_t is ignored
        method:
            'sum' (default, except if skip is used): Use the Decomposition theorem from arXiv:1911.02537.
            'impulsive' (used as a reference for tests): Discretize (simulate) the Linear Impulsive System
            The result of both methods must be the same (up to numerical error).
        datatype:
            Datatype for computations:
            numpy (standard double-precision float) or mpmath.iv (interval arithmetic).
            Defaults to the datatype given at initialization (which defaults to numpy).
        '''
        datatype = datatype or self.datatype
        datatype = check_datatype(datatype)
        d = AbstractMatrix.from_type(datatype)
        def check_and_fill_array_param(array, length, default):
            """
            Check an array parameter for the specified length.
            If the array is None, create an array filled with the given default value.
            """
            if array is None:
                array = np.full(length, default)
            assert len(array) == length, "Parameter has invalid dimension"
            return array
        skip_u = check_and_fill_array_param(skip_u, self.sys.m, False)
        dtu = check_and_fill_array_param(dtu, self.sys.m, 0)
        skip_y = check_and_fill_array_param(skip_y, self.sys.p, False)
        dty = check_and_fill_array_param(dty, self.sys.p, 0)
        if method is None:
            method = 'sum'
            if any(skip_u) or any(skip_y) or skip_ctrl:
                # Skipping is not yet supported for the 'sum' method because the underlying theorem has not yet been proven for this case
                # fall back to other method
                method = 'impulsive'
        assert method in ['sum', 'impulsive']
        if method == 'sum':
            assert not(any(skip_u) or any(skip_y) or skip_ctrl), "Skipping is not yet supported for the 'sum' method because the underlying theorem has not yet been proven for this case"
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
        else:
            assert method == 'impulsive'
            time = -self.sys.T/2
            A = d.eye(self.n)
            events = [(dtu[i], self.A_u[i]) for i in range(self.sys.m) if not skip_u[i]]
            events += [(dty[i], self.A_y[i]) for i in range(self.sys.p) if not skip_y[i]]
            events += [(self.sys.T/2, d.eye(self.n) if skip_ctrl else self.A_ctrl)]
            events.sort(key=lambda ev: ev[0]) # sort events by time
            for next_time, A_event in events:
                next_time = float(next_time)
                assert next_time >= time
                A = d.convert(A_event) @ d.expm(d.convert(self.A_cont) * (d.convert_scalar(next_time) - d.convert_scalar(time))) @ d.convert(A)
                time = next_time
            assert time == self.sys.T/2
            D = A - d.convert(self.Ak_nominal)
            return D

    def simulate_random(self, L, x0, scale_timing=None, w=None, skip_u=None, skip_y=None, skip_ctrl=None, delta_t_u=None, delta_t_y=None):
        """
        simulate the system

        @return (x, A), where
            x = hstack(x_0, x_1, x_2, ...)
                is np.array of size (self.n, L)
                where x[i, k] is the i-th component of x at time k.
            A = np.array([A_0, A_1, ...])
                is np.array of size (L, self.n, self.n)
                such that ``x_(k+1) = A_k @ x_k + (...) @ w_k``
                where ``A_k = A[k, :, :]``

        @param L: number of timesteps

        @param x0: initial state

        @param scale_timing: increase timing uncertainty by given factor.
            Must not be used if delta_t_* is given.

            None: no scaling.
            scalar: global scale
            np.array of length L: time-dependent scaling by scale_timing[k] at time k

        @param delta_t_u: timing sequence for u
            None: randomly generate it as uniform distribution.
                  If scale_timing != None, then it is scaled by scale_timing
            np.array of size (self.sys.m, L):
                use given time delay series (absolute times, not normalized).
                delta_t_u[i,k] is the absolute time delay of the i-th (starting at 0) component of u at time k.

        @param delta_t_y: timing sequence for y
            analogous to delta_t_u, but of array size (self.sys.p, L)

        @param skip_u: skip sequence for u
            None:
                never skip.
            boolean np.array of size (self.sys.m, L):
                skip update of u[i] at time k if skip_u[i,k] is True.

        @param skip_y: skip sequence for y
            None:
                never skip.
            boolean np.array of size (self.sys.m, L):
                see skip_u

        @param skip_ctrl: skip sequence for controller
            None:
                never skip.
            boolean np.array of size L:
                skip update of controller at the end of cycle k if skip_ctrl[k] is True.
        """
        A_k = np.zeros((L, self.n, self.n))
        d = AbstractMatrix.from_type(self.datatype)
        x = d.zeros(self.n, L)
        if w is None:
            w = d.zeros(self.n, L)
        else:
            w = d.convert(w)
        x[:,0] = d.convert(x0)
        if scale_timing is None:
            scale_timing = 1
        else:
            assert delta_t_u is None and delta_t_y is None
        def random_in_range(t_min, t_max):
            """
            return a uniformly random vector of the same size as t_min
            with values between t_min and t_max.
            """
            return t_min + np.random.uniform(size=(len(t_min), )) * (t_max - t_min)
        for i in range(1, L):
            if isinstance(scale_timing, (int, float)) or len(scale_timing) == 1:
                scale_timing_i = scale_timing
            else:
                scale_timing_i = scale_timing[i]
            def delta_t(delta_t_array, random_delta_t_min, random_delta_t_max):
                """
                determine delta_t at the current time from
                - the array (time series), if given
                - else, random_scale * uniform random number from the given range
                """
                if delta_t_array is None:
                    return scale_timing_i * random_in_range(random_delta_t_min, random_delta_t_max)
                else:
                    dt = delta_t_array[:, i-1]
                    assert len(dt) == len(random_delta_t_min), "given delta_t_{u,y} has wrong shape"
                    assert all(dt < self.sys.T/2), "values of given delta_t_{u,y} are out of bounds"
                    assert all(dt > -self.sys.T/2), "values of given delta_t_{u,y} are out of bounds"
                    return dt
            dtu = delta_t(delta_t_array=delta_t_u, random_delta_t_min=self.sys.delta_t_u_min, random_delta_t_max=self.sys.delta_t_u_max)
            dty = delta_t(delta_t_array=delta_t_y, random_delta_t_min=self.sys.delta_t_y_min, random_delta_t_max=self.sys.delta_t_y_max)
            skip_u_k = skip_u[:, i-1] if skip_u is not None else None
            skip_y_k = skip_y[:, i-1] if skip_y is not None else None
            skip_ctrl_k = skip_ctrl[i-1] if skip_ctrl is not None else None
            A_k[i-1, :, :] = self.Ak_nominal + self.Ak_delta_to_nominal(dtu=dtu,dty=dty,skip_u=skip_u_k, skip_y=skip_y_k, skip_ctrl=skip_ctrl_k)
            x[:, i] = A_k[i-1, :, :] @ x[:, i-1] + w[:, i-1]
        return (x, A_k)