#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

"""
Examples from the publication
Gaukler, M., & Ulbrich, P. (2019). Worst-Case Analysis of Digital Control Loops with Uncertain Input/Output Timing. 6th International Workshop on Applied Verification of Continuous and Hybrid Systems (ARCH '19).
https://dx.doi.org/10.29007/c4zl

"""

from .controlloop import DigitalControlLoop
import numpy as np
from .controlloop import blkrepeat

def example_A1_stable_1():
    '''
    scalar example, succeeds with SpaceEx (with the settings we use here).
    '''
    system=DigitalControlLoop()
    system.A_p = np.array([[0.05]])
    system.B_p = np.array([[0.5]])
    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([-1]);
    system.x_p_0_max = np.array([+1]);

    system.A_d = np.array([[-0.01]])
    system.B_d = np.array([[-0.4]])
    system.C_d = np.array([[1]])

    system.T=1
    system.delta_t_u_min=np.array([-0.001])
    system.delta_t_u_max=np.array([0.002])
    system.delta_t_y_min=np.array([-0.1])
    system.delta_t_y_max=np.array([0.002])

    system.spaceex_iterations=200
    system.spaceex_iterations_for_global_time=500
    # with this setting, it also works in the SpaceEx "LGG scenario"
    system.spaceex_sampling_time=1e-3
    system.plot_t_max = 7
    system.plot_ylim_xp = [[-1.5, 1.5]]
    system.plot_ylim_xd = [[-1.5, 1.5]]
    return system

def example_A2_stable_1():
    system=example_A1_stable_1()
    system.delta_t_u_min=np.array([-0.0001])
    system.delta_t_u_max=np.array([0.0001])
    system.delta_t_y_min=np.array([-0.0001])
    system.delta_t_y_max=np.array([0.0001])
    return system

def example_A3_stable_1():
    system=example_A1_stable_1()
    system.delta_t_u_min=np.array([-0.4])
    system.delta_t_u_max=np.array([0.1])
    system.delta_t_y_min=np.array([-0.1])
    system.delta_t_y_max=np.array([0.4])
    return system

def example_A4_unknown_1():
    '''
    scalar example, made more difficult.
    No success with SpaceEx (with the settings we tried).
    '''
    system=example_A3_stable_1()
    system.delta_t_u_max[0]=0.4
    # number of iterations:
    # 3600 -> K=1.3, no fixpoint
    # 3670 ... 20000 -> crash
    # unfortunately, there is no usable number of iterations without crashing.
    system.spaceex_iterations=3670
    system.spaceex_iterations_for_global_time=2000 # avoid crash when plotting over t
    return system

def example_A5_stable_diagonal(repetitions):
    '''
    scalar stable example (example_A3_stable_1()), repeated multiple times.
    Repeating (duplicating without interconnection) does not change stability, so this must be stable.
    For repetitions=2: No success with SpaceEx (with the settings we tried), although this system has only two inputs, outputs, and physical states.
    '''
    assert repetitions >= 2
    system=example_A3_stable_1();
    system.increase_dimension(repetitions)
    system.spaceex_iterations=2000
    system.spaceex_iterations_for_global_time=2000 # works for 2000 (but time axis only up to 3 seconds), crash for >= 4000
    return system


def example_B1_stable_3():
    '''
    Extremely stable example: The plant is stable, and the controller has negligible influence.
    You can almost tell from looking at the matrices that this must be stable.
    '''
    system=DigitalControlLoop()
    system.A_p = np.array([[-1,0.002,0.003],[0.004,-5,0.006],[0.007,0.008,-9]])
    system.B_p = np.array([[10,11],[12,13],[14,15]])/10000.
    system.C_p = np.array([[16, 17, 18]])

    system.x_p_0_min = np.array([-1,-1,-1]);
    system.x_p_0_max = np.array([+1,+1,+1]);

    system.A_d = np.array([[0.019,0.020], [0.021,0.022]])
    system.B_d = np.array([[0.023], [0.024]])
    system.C_d = np.array([[0.025,0.026], [0.027,0.028]])

    system.T=2
    system.delta_t_u_min=np.array([-0.1,-0.2])*system.T
    system.delta_t_u_max=np.array([0.1,0.2])*system.T
    system.delta_t_y_min=np.array([-0.3])*system.T
    system.delta_t_y_max=np.array([0.3])*system.T

    # fails with spaceex_directions="oct". Unclear why.
    system.spaceex_directions="box"
    
    system.plot_t_max = 10
    return system

def example_C_quadrotor_attitude_one_axis(perfect_timing=False):
    '''
    Angular rate control of quadrotor around one axis, linear and highly simplified.

    see also: example_C_simulation_of_nominal_case.slx

    Based on: "Benchmark: Quadrotor Attitude Control" - A. E. C. Da Cunha, ARCH15
    citable short version: https://doi.org/10.29007/dc68
    extended version: https://cps-vo.org/node/20290
    '''

    system=DigitalControlLoop()
    Jx=9.0359e-06
    K_control_integral = 3.6144e-3 # K_I,p
    K_control_proportional = 2.5557e-4 # K_f,p
    system.A_p = np.array([[0]])
    system.B_p = np.array([[1/Jx]])

    # the original model is continuous. We consider a sampled version of the controller.
    # All following parameters are not from the original example.
    system.T=0.01

    system.C_p = np.array([[1]])

    system.x_p_0_min = np.array([1]) * -1
    system.x_p_0_max = np.array([1]) * 1

    # We use a very primitive controller discretization:
    # x_d_1: forward-euler approximation of integrator
    # x_d_2: delay-state for feedthrough (this is not optimal, but the current controller model does not support feedthrough)
    system.A_d = np.asarray(np.diag([1, 0]))
    system.B_d = np.array([[system.T], [1]])

    system.C_d = np.array([[-K_control_integral, -K_control_proportional]])

    max_timing = 0.0 if perfect_timing else 0.01
    system.delta_t_u_min=np.array([1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=system.delta_t_u_min
    system.delta_t_y_max=-system.delta_t_y_min
    
    system.plot_t_max = 0.25
    
    system.plot_ylim_xp = [[-4, 4]]
    system.spaceex_iterations = 300 if perfect_timing else 2000 # reduced to prevent extremely long (infinite?) plot computation time for perfect timing
    # for perfect timing, unfortunately, even at 1000 iterations the computation of the interval bounds runs into a timeout, although the iterations itself are very fast. So it seems impossible to find a number of iterations for which the computation finishes within two hours, but shows that K becomes very large
    # perfect_timing=False: 5000 -> K=21, 6000 -> K=25, 11000 -> crash
    return system


def example_D_quadrotor_attitude_three_axis(perfect_timing=False):
    '''
    Angular rate control of quadrotor around all three axes axis, linear and highly simplified.

    This is more difficult than a repetition of example_C, because one input influences multiple axes. The input matrix can be "inverted", but input timing uncertainties prevent perfect decoupling of the three subsystems.

    see also: example_C_simulation_of_nominal_case.slx

    Based on: "Benchmark: Quadrotor Attitude Control" - A. E. C. Da Cunha, ARCH15
    citable short version: https://doi.org/10.29007/dc68
    extended version: https://cps-vo.org/node/20290
    '''
    # see example_C for explanations (of the one-axis case)
    system=DigitalControlLoop()
    J=np.array([9.0359e-06, 9.1268e-06, 1.9368e-05]) # J_{x,y,z}
    system.A_p = np.diag([0,0,0])
    B_torque = np.diag(1/J)

    # Original, continuous controller is PI.
    # Gains from paper (Da Cunha, ARCH15, extended version), Table 1.
    K_control_proportional = np.array([2.557, 2.5814, 5.4781]) * 1e-4 # originally named K_f,{p,q,r}
    K_control_integral = np.array([3.6144, 3.6507, 7.7472]) * 1e-3 # originally named K_I,{p,q,r}

    # the original model is continuous. We consider a sampled version of the controller.
    # All following parameters are not from the original example.
    system.T=0.01

    system.C_p = np.eye(3)

    system.x_p_0_min = np.array([1]*3) * -1
    system.x_p_0_max = np.array([1]*3) * 1


    # We use a very primitive discretization:
    # x_d_1: forward-euler approximation of integrator
    # x_d_2: delay-state for feedthrough (this is not optimal, but the current controller model does not support feedthrough)
    system.A_d = blkrepeat(np.diag([1, 0]), 3)
    system.B_d = blkrepeat([[system.T], [1]], 3)

    # first step: controller output for torque around x,y,z axis
    system.C_d = np.array([
                            [-K_control_integral[0], -K_control_proportional[0], 0, 0, 0, 0],
                            [0, 0, -K_control_integral[1], -K_control_proportional[1], 0, 0],
                            [0, 0, 0, 0, -K_control_integral[2], -K_control_proportional[2]],
                          ])
    # second step: controller output is rotor commands 1-4.

    # plant input is rotor command (speed) delta_{1,2,3,4}
    # This produces rotor force F_{1,2,3,4} and rotor torque tau_{1,2,3,4}.
    # The original model does not seem to provide information on the relationship from delta to F_i and tau_i.
    # For simplicity, we assume that delta_i = F_i = tau_i/gamma.
    # For more simplicity:
    gamma=1./100 # force to torque ratio
    length=0.1 # "length" of quadrocopter frame (centerpoint to rotor)

    # relation of rotor force to torque: (Da Cunha, ARCH15, extended version), page 2 bottom / page 3 top
    # tau_phi (around x-axis) = l * (F4 - F2)
    # tau_theta (around y-axis) = l * (F3 - F1)
    # tau_psi (around z-axis) = tau_1 - tau_2 + tau_3 - tau_4 = gamma * (F1 - F2 + F3 - F4)
    # Fz (force in z-direction, not used here) = F1 + F2 + F3 + F4

    # written in matrix form:
    # [tau_phi; tau_theta; tau_psi; Fz] = rotor_to_torque * [delta_1; delta_2; delta_3; delta_4] = rotor_to_torque * u
    rotor_to_torque = np.array([[0, -length, 0, length], [-length, 0, length, 0], [gamma, -gamma, gamma, -gamma], [1, 1, 1, 1]])
    # x_p' = Ax + B_torque*tau = Ax + B_torque * rotor_to_torque[without last row] * tau
    system.B_p = B_torque.dot(rotor_to_torque[:-1,:])
    # u = rotor_to_torque^-1 * [tau_{...};  0] = rotor_to_torque^-1 * [C_d * x_d; 0]
    system.C_d = np.linalg.inv(rotor_to_torque).dot(np.vstack((np.eye(3), np.zeros((1,3))))).dot(system.C_d)

    # maximum timing deviation, relative to T
    max_timing = 0.0 if perfect_timing else 0.01
    system.delta_t_u_min=np.array([1, 1, 1, 1]) * -max_timing * system.T
    system.delta_t_u_max=-system.delta_t_u_min
    system.delta_t_y_min=np.array([1, 1, 1]) * -max_timing * system.T
    system.delta_t_y_max=-system.delta_t_y_min
    system.spaceex_iterations = 6000 if perfect_timing else 2500
    system.spaceex_iterations_for_global_time = 3000 # avoid crash with "Support function evaluation requested for an empty set" at iteration 3763 when plotting over t for perfect_timing=True
    # 2000 ... 2400 -> K=1.0001 and no fixpoint for perfect_timing=False
    # 2475 ... 4000 -> crash for perfect_timing=False
    # 5000 -> timeout for perfect_timing=False, K>800 for perfect_timing=True
    return system

def example_D2b():
    '''
    Example D2b from Gaukler et al IFAC2020 https://doi.org/10.1016/j.ifacol.2020.12.1019
    
    = Example D2 with doubled timing uncertainty
    '''
    s = example_D_quadrotor_attitude_three_axis()
    s.increase_timing(2)
    return s

def example_D2c():
    '''
    Example D2c from Gaukler et al IFAC2020 https://doi.org/10.1016/j.ifacol.2020.12.1019
    
    = Example D2 with doubled dimension
    '''
    s = example_D_quadrotor_attitude_three_axis()
    s.increase_dimension(2)
    s.skip_simulation = True # Workaround: PySim crashes (out of memory), so skip it
    return s


def example_D2d():
    '''
    Example D2d from Gaukler et al IFAC2020 https://doi.org/10.1016/j.ifacol.2020.12.1019
    
    = Example D2  with dimension*2, dt_y_max=0.1*dt_y_max
    '''
    s = example_D_quadrotor_attitude_three_axis()
    s.increase_dimension(2)
    s.delta_t_y_max=0.1*s.delta_t_y_max
    s.skip_simulation = True # Workaround: PySim crashes (out of memory), so skip it
    return s


def example_D2e():
    '''
    Example D2e from Gaukler et al IFAC2020 https://doi.org/10.1016/j.ifacol.2020.12.1019
    
    = Example D2  with dt_y_max=0.1*dt_y_max
    '''
    s = example_D_quadrotor_attitude_three_axis()
    s.delta_t_y_max=0.1*s.delta_t_y_max
    return s

def example_E_timer():
    '''
    not really an example, just a simple test to visualize the timing.

    The plant just a timer with x2(t)=1 (constant) and output x1(t)=y(t)=t, regardless of the input. Therefore, this is obviously unstable.

    The controller is u[k+1] = y[k] = kT + delta_t_y[k]. However, u has no effect.
    This means that x1 is the time axis and the value of u and y[k] is the time t at which the measurement was sampled.
    '''
    system=DigitalControlLoop()
    system.A_p = np.array([[0, 1], [0, 0]])
    system.B_p = np.array([[0], [0]])
    system.C_p = np.array([[1, 0]])

    system.x_p_0_min = np.array([0, +1]);
    system.x_p_0_max = np.array([0, +1]);

    system.A_d = np.array([[0]])
    system.B_d = np.array([[1]])
    system.C_d = np.array([[1]])

    system.T=1
    system.delta_t_u_min=np.array([-0.4])
    system.delta_t_u_max=np.array([0.1])
    system.delta_t_y_min=np.array([-0.2])
    system.delta_t_y_max=np.array([0.3])
    
    system.spaceex_iterations = 200 # SpaceEx will never find a fixpoint because this system is unstable.
    return system

