#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameters for a digital control loop, as modeled in the linear disturbance-free case of the following publication:
Gaukler, M., & Ulbrich, P. (2019). Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing. 6th International Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH '19).
https://dx.doi.org/10.29007/c4zl
"""

import numpy as np
from scipy.linalg import expm
from collections import OrderedDict
from .lis.np_matrix_utils import blkrepeat
from .lis.generic_matrix import approx_max_abs_eig, NumpyMatrix
blockmatrix = NumpyMatrix.blockmatrix

class DigitalControlLoop(object):
    """
    linear digital control loop with uncertain timing
    """
    def __init__(self):
        self.A_p = None
        self.B_p = None
        self.C_p = None

        self.x_p_0_min = None
        self.x_p_0_max = None

        # NOTE: Disturbance is not supported yet. If support is added, please make sure to also add it in the to_latex() function.

        self.A_d = None
        self.B_d = None
        self.C_d = None


        # all times are given in seconds (delta_t is absolute, not relative to T)
        self.T = None
        self.delta_t_u_max = None
        self.delta_t_u_min = None
        self.delta_t_y_max = None
        self.delta_t_y_min = None

        # disable the "roughly one-step delay" from x_p to y to u
        # see DigitalControlLoop.enable_immediate_ctrl()
        self.immediate_ctrl = False

        ## TODO: factor out the following reachability-related options. They are not used in the LMI-based analysis.
        # enable to add the global time t as a state for debugging
        # If enabled, the reachability computation cannot terminate, but its intermediate result can be useful to check the behavior over time and the liveness of the automaton (t must keep growing, not get stuck).
        self.global_time = False

        # SpaceEx: simulation only? (deterministic x0)
        self.spaceex_only_simulate = False
        
        # Skip PySim simulation (experimental, only as a workaround if PySim crashes)
        self.skip_simulation = False

        # apply continuization? (only for internal use by HybridSysControlLoop)
        self.continuize = False

        # Normally, the automaton uses nondeterministic transitions in the semantics of Henzinger (same as SpaceEx). We need that because the sampling transition *does not necessarily have to* happen at the first possible time, it may also happen at some other time in the permitted interval [delta_t_min, delta_t_max].
        # This means that a simulator would find a "random tree" of possible solutions, however pysim does not support that and always takes the first possible transition ("urgent semantics").
        # As a workaround, this option can be enabled, so that the automaton is written in urgent semantics, and the randomness is modeled by including a (very weak, not truly random!) "pseudo-random" number generator (PRNG) in the automaton.
        # If enabled, this means that for zero disturbance, the simulation result only depends on the initial state and the simulator does not have to understand the semantics of nondeterministic, non-urgent transitions
        # This option does not make much sense for analysis, even though -- assuming the analysis tool uses urgent semantics -- it should in theory yield correct results with significantly less efficiency. (Except if the analysis tool is clever enough to use the weakness (non-randomness) of the pseudorandom number generator to exclude some traces that the RNG will never reach.)
        self.use_urgent_semantics_and_pseudorandom_sequence = False

        # spaceEx configuration parameters from .cfg file
        # time horizon (in multiples of the controller period)
        self.spaceex_time_horizon_periods = 25
        # maximum iterations (make small enough so that the timeout isn't hit)
        self.spaceex_iterations = 2000
        # timeout for reachability analysis (computation time in seconds) - problematic because it may be hit on slower computers
        self.spaceex_timeout = 7200
        # timeout for generating extra plots - (computation time in seconds) problematic because it may be hit on slower computers
        self.spaceex_timeout_extra_plots = 5*60*60
        # maximum iterations for "reachability over global time" plot (None: same as spaceex_iterations)
        self.spaceex_iterations_for_global_time = None
        # Scenario (overapproximation) - ignored for simulation
        self.spaceex_scenario="stc"
        # approximation directions
        self.spaceex_directions = "oct"
        # sampling time for time elapse
        self.spaceex_sampling_time=1e-3
        # aggregation
        self.spaceex_set_aggregation="none"
        # clustering percentage
        self.spaceex_clustering_percent=100

        # Time-axis limit for plotting (None: auto)
        self.plot_t_max = None
        
        # Y-axis limits for plotting:
        # [[min_xp_0, max_xp_0], [min_xp_1, max_xp_1], ...]
        self.plot_ylim_xp = None
        self.plot_ylim_xd = None

        # results of analysis and simulation
        self.results = {}


    def _check_and_update_dimensions(self):
        def isarray(a):
            '''
            determine whether object is np.array and not np.matrix

            (Note that array has slightly different semantics than matrix!)
            We ensure that objects are not numpy matrices, because they behave slightly differently - see https://www.numpy.org/devdocs/user/numpy-for-matlab-users.html#array-or-matrix-which-should-i-use
            This is out of an abundance of caution.
            '''
            return isinstance(a, np.ndarray) and not isinstance(a, np.matrixlib.defmatrix.matrix)

        # This is one of the few parts where I hate Python and love Java: Checking types.
        for m in [self.A_p, self.B_p, self.C_p, self.A_d, self.B_d, self.C_d]:
            assert isarray(m), "A,B,C must be of type numpy.array, but one is of type {}. System: {}".format(type(m), self)
        for a in [self.delta_t_u_min, self.delta_t_u_max, self.delta_t_y_min, self.delta_t_y_max]:
            assert isarray(m), "delta_t_... must be of type numpy.array"
        for a in [self.x_p_0_min, self.x_p_0_max]:
            assert isarray(m), "x_p_0_... must be of type numpy.array"

        self.n_p = self.A_p.shape[0]
        assert self.n_p >= 1, "system must have at least one continuous state"
        self.n_d = self.A_d.shape[0]
        self.m = self.B_p.shape[1]
        self.p = self.C_p.shape[0]
        assert self.A_p.shape == (self.n_p, self.n_p)
        assert self.B_p.shape == (self.n_p, self.m)
        assert self.C_p.shape == (self.p, self.n_p)
        assert self.A_d.shape == (self.n_d, self.n_d)
        assert self.B_d.shape == (self.n_d, self.p)
        assert self.C_d.shape == (self.m, self.n_d)
        assert self.x_p_0_min.shape == self.x_p_0_max.shape
        assert self.x_p_0_max.shape == (self.n_p, )

        assert self.delta_t_u_min.shape == self.delta_t_u_max.shape
        assert self.delta_t_u_max.shape == (self.m, )
        assert self.delta_t_y_min.shape == self.delta_t_y_max.shape
        assert self.delta_t_y_max.shape == (self.p, )
        assert all(self.delta_t_u_max >= self.delta_t_u_min), "delta_t_u_max={} is not >= delta_t_u_min={}".format(self.delta_t_u_max, self.delta_t_u_min)
        assert all(self.delta_t_y_max >= self.delta_t_y_min), "delta_t_y_max={} is not >= delta_t_y_min={}".format(self.delta_t_y_max, self.delta_t_y_min)
        assert all(self.delta_t_y_min > -self.T/2)
        assert all(self.delta_t_u_min > -self.T/2)
        assert all(self.delta_t_y_max < self.T/2)
        assert all(self.delta_t_u_max < self.T/2)
        assert all(self.x_p_0_max >= self.x_p_0_min)

        for (lim, length) in [(self.plot_ylim_xp, self.n_p), (self.plot_ylim_xd, self.n_d)]:
            if lim is not None:
                assert isinstance(lim, list), "self.plot_ylim_xd and self.plot_ylim_xp must be lists or 'None'"
                assert len(lim) == length, "self.plot_ylim_xd and self.plot_ylim_xp must be lists of length self.n_d and self.n_p"

    def __repr__(self):
        ret = "System("
        for (key,val) in sorted(self.__dict__.items()):
            multiline_variable_indentation = "\n" + " " * (len(key)+6)
            ret += "\n   {} = {}".format(key, multiline_variable_indentation.join(repr(val).split("\n")))
        ret += ")"
        # TODO: unfortunately, the output of this is not directly a valid python expression for getting the system object.
        return ret

    def to_latex(self):
        def number_to_latex(number):
            s=str(number)
            if s.endswith(".0"):
                s=s[:-2]
            return r"\num{" + s + "}"
        def numpy_to_latex(matrix):
            if isinstance(matrix, str):
                return matrix
            if isinstance(matrix, (float, int)):
                return number_to_latex(matrix)
            if matrix.shape == (1,):
                # 1d scalar
                return number_to_latex(matrix[0])
            elif matrix.shape ==  (1,1):
                # 2d scalar
                return number_to_latex(matrix[0,0])
            elif len(matrix.shape) == 1:
                return r"\mat{" + r"\\".join([number_to_latex(row) for row in matrix]) + r"} "
            else:
                assert len(matrix.shape) == 2
                return r"\mat{" + r"\\".join([" & ".join([number_to_latex(col) for col in row]) for row in matrix]) + r"} "

        def numpy_interval_to_latex(minv, maxv):
            """
            format multidimensional interval [minv, maxv] as LaTeX
            @param minv: np.ndarray
            @param maxv: np.ndarray
            """
            dimension = minv.shape[0]
            FORMAT = r"[{low}; {high}]"
            if np.all(minv == minv[0]) and np.all(maxv == maxv[0]):
                s = FORMAT.format(low=minv[0], high=maxv[0])
                if dimension > 1:
                    s += "^{}".format(dimension)
                return s
            return r" \times ".join([FORMAT.format(low=minv[i], high=maxv[i]) for i in range(dimension)])
        variables = OrderedDict()
        variables["A_p"] = self.A_p
        variables["B_p"] = self.B_p
        variables["C_p"] = self.C_p
        variables["A_d"] = self.A_d
        variables["B_d"] = self.B_d
        variables["C_d"] = self.C_d
        variables["T"] = self.T
        variables["X_{\subsPlant,0}"] = numpy_interval_to_latex(self.x_p_0_min, self.x_p_0_max)
        variables[r"\underline {\Delta t}_u"] = self.delta_t_u_min
        variables[r"\overline {\Delta t}_u"] = self.delta_t_u_max
        variables[r"\underline {\Delta t}_y"] = self.delta_t_y_min
        variables[r"\overline {\Delta t}_y"] = self.delta_t_y_max
        variables[r"n_{dist}"] = 0 # not supported yet
        return r"\begin{align}" + "\\\\ \n".join(["{name} &= {value}".format(name=name, value=numpy_to_latex(value)) for (name, value) in variables.items()]) + "\n" + r"\end{align}"

    def increase_dimension(self, factor):
        '''
        increase the state dimension by repeating the system n times.
        The resulting system has a block-diagonal structure.
        '''
        self._check_and_update_dimensions()
        assert factor >= 1
        assert isinstance(factor, int)

        self.A_p = blkrepeat(self.A_p, factor)
        self.B_p = blkrepeat(self.B_p, factor)
        self.C_p = blkrepeat(self.C_p, factor)
        self.A_d = blkrepeat(self.A_d, factor)
        self.B_d = blkrepeat(self.B_d, factor)
        self.C_d = blkrepeat(self.C_d, factor)
        self.x_p_0_max = np.tile(self.x_p_0_max, factor)
        self.x_p_0_min = np.tile(self.x_p_0_min, factor)
        self.delta_t_u_max = np.tile(self.delta_t_u_max, factor)
        self.delta_t_u_min = np.tile(self.delta_t_u_min, factor)
        self.delta_t_y_max = np.tile(self.delta_t_y_max, factor)
        self.delta_t_y_min = np.tile(self.delta_t_y_min, factor)
        # Y-axis limits for plotting:
        # [[min_xp_0, max_xp_0], [min_xp_1, max_xp_1], ...]
        
        if self.plot_ylim_xd:
            assert isinstance(self.plot_ylim_xp, list)
            self.plot_ylim_xp = self.plot_ylim_xd * factor
        if self.plot_ylim_xd:
            assert isinstance(self.plot_ylim_xp, list)
            self.plot_ylim_xd = self.plot_ylim_xd * factor
        self._check_and_update_dimensions()

    def increase_timing(self, factor):
        '''
        Scale up timing by a given factor
        '''
        self.delta_t_u_min = self.delta_t_u_min * factor
        self.delta_t_u_max = self.delta_t_u_max * factor
        self.delta_t_y_min = self.delta_t_y_min * factor
        self.delta_t_y_max = self.delta_t_y_max * factor
        self._check_and_update_dimensions()

    def transform_states_to_outputs(self):
        """
        Model transformation to C_d=I, C_p=I.
        Old values of C_p, C_d are merged into B_p, B_d.
        """
        assert self.is_nominal_timing(), "This transformation is only supported for perfect timing"
        self.B_d = self.B_d @ self.C_p
        self.B_p = self.B_p @ self.C_d
        self.C_d = np.eye(self.n_d)
        self.C_p = np.eye(self.n_p)
        self.delta_t_u_max = np.zeros((self.n_d, ))
        self.delta_t_u_min = np.zeros((self.n_d, ))
        self.delta_t_y_max = np.zeros((self.n_p, ))
        self.delta_t_y_min = np.zeros((self.n_p, ))
        self._check_and_update_dimensions()

    def enable_immediate_ctrl(self):
        '''
        switch to "immediate controller mode", where the dynamics are
        \dot x_p = A_p x_p + B_p x_d
        x_d' = B_d x_p + A_d x_d  at  t=kT
        C_p = I
        C_d = I
        delta_t_u = 0
        delta_t_y = 0.

        Old values of C_p, C_d are merged into B_p, B_d.
        Old delta_t is discarded.
        The new dynamics however differ because they no longer have the one-step delay from x_p[k] to u[k+1].
        Old: u[k] only depends on x_p[0] ... x_p[k-1].
        New: u[k] also depends on x_p[k].
        The new dynamics match what you would get for delta_t_u = -T/2 + epsilon  and delta_t_y = +T/2 - epsilon, except for a time shift by T/2.
        This only makes sense if the real-world controller has negligible execution time.

        WARNING: This mode is only supported by a small number of functions.
        '''
        self.increase_timing(0)
        self.transform_states_to_outputs()
        self.immediate_ctrl = True



    def nominal_case_stability(self):
        '''
        Check stability for delta_t_...=0
        @return 'stable' or 'unstable' if stability is clearly known, otherwise another other descriptive value.
        '''
        self._check_and_update_dimensions()
        if self.immediate_ctrl:
            return 'not implemented for immediate_ctrl'
        # rewrite system as linear impulsive system, similar to [Gaukler et al. HSCC 2017].
        # Note that it does not matter for stability when the controller is computed,
        # as long as it is between the previous updates of u,y and the next update of u,y.
        # This is because the controller does not directly affect the physical plant.
        # Therefore, if there are no delays, we can move the controller computation to
        # "just before" t=kT, immediately before the update of u and y.
        # The new system now is continuous-time between kT and (k+1)T, and jumps at kT.
        #
        # new total state: x_total = [x_p; x_d; u; y]
        # dimensions of state components:
        blocklengths = [self.n_p, self.n_d, self.m, self.p]

        # continuous dynamics (except at t=kT):
        # x_total' = A_cont_total * x_total
        A_cont_total = blockmatrix([[self.A_p, 0, self.B_p, 0],
                                    [0,        0, 0,        0],
                                    [0,        0, 0,        0],
                                    [0,        0, 0,        0]],
                                   blocklengths)
        # at t=kT   (we moved the controller computation to t=kT, without loss of generality, see note above)
        # compute controller, then update u (currently computed value) and y (for next computation)
        A_discrete_total = blockmatrix([[np.eye(self.n_p),        0,        0,   0],
                                        [0,        self.A_d,        0,   self.B_d],
                                        [0,        self.C_d, 0,   0],
                                        [self.C_p, 0,        0,   0]],
                                       blocklengths)
        A_total = expm(A_cont_total*self.T) @ A_discrete_total
        worst_eigenvalue = approx_max_abs_eig(A_total)
        accuracy = 1e-5
        if worst_eigenvalue < 1 - accuracy:
            return 'stable'
        elif worst_eigenvalue > 1 + accuracy:
            return 'unstable'
        else:
            # numerically at the edge of "stable" and "unstable",
            # for example: sine oscillator, integrator ( = marginally stable) or double integrator ( = unstable!)
            return 'borderline unstable'

    def is_nominal_timing(self):
        '''
        are all delta_t == 0?
        '''
        for delta_t in [self.delta_t_u_min, self.delta_t_u_max, self.delta_t_y_min, self.delta_t_y_max]:
            if any(delta_t != 0):
                return False
        return True

    def is_fixed_timing(self):
        '''
        are all delta_t_min == delta_t_max?
        '''
        return all(self.delta_t_u_min == self.delta_t_u_max) and all(self.delta_t_y_min == self.delta_t_y_max)

