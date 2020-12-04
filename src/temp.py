#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:19:34 2019

@author: ak22emur
"""

import qronos.lis.lis as l
import qronos.examples as ex
import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg
from qronos.lis.generic_matrix import NumpyMatrix

#%%
lis = l.LISControlLoop(ex.example_D_quadrotor_attitude_three_axis(), datatype=np)
n = lis.n
n_p = lis.sys.n_p
Select_xp = scipy.linalg.block_diag(1*np.eye(n_p), 0*np.eye(n - n_p))
# Sel = np.eye(n)
C0 = Select_xp
C = C0
x0 = C0 @ np.ones((lis.n,))

def normalize(x):
    return x / np.sqrt(x.T @ x)
L=25
NUM_RUNS = 42
P_sqrt_T = np.array([[ 2.98283705e-06, -2.09845108e-16, -3.05644038e-15,
         2.59327293e-05, -1.88328101e-03,  6.85284376e-16,
        -6.24321807e-13, -7.86129022e-15, -2.15863659e-11,
         1.88175399e-03,  6.24337621e-13,  2.15876471e-11,
        -3.04494508e-11, -2.03770623e-04, -3.04424360e-11,
         2.03770562e-04],
       [ 0.00000000e+00,  3.08897441e-06,  1.35900905e-14,
         5.06620829e-16, -3.81575844e-13,  2.70420358e-05,
        -6.12584959e-04,  1.18696922e-13, -9.06674714e-13,
         3.81493905e-13,  6.11015350e-04,  8.99582927e-13,
        -2.11321169e-04, -1.78163924e-12,  2.11321166e-04,
        -1.75643132e-12],
       [ 0.00000000e+00,  0.00000000e+00,  2.59587622e-05,
        -4.66667971e-15, -1.09213879e-11, -2.83260972e-15,
         5.50464572e-12,  2.37312359e-04,  1.36973481e-03,
         1.09213353e-11, -5.50473334e-12, -1.38215773e-03,
         8.28751096e-05, -8.29492687e-05,  8.28751096e-05,
        -8.29492687e-05],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         3.34335552e-05,  1.55411611e-03,  9.36636518e-16,
         4.32713924e-13, -2.99930113e-14,  4.11945598e-11,
        -1.55295485e-03, -4.32748698e-13, -4.11957200e-11,
         6.15893333e-11, -2.81959221e-04,  6.16057151e-11,
         2.81959345e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  4.44904868e-01,  9.23122441e-18,
         7.31043299e-14, -4.46200413e-14, -6.08666439e-10,
        -4.44904865e-01, -7.31060000e-14,  6.08663925e-10,
        -1.88159793e-10, -5.72556594e-07, -1.88160551e-10,
         5.72180451e-07],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  3.45808573e-05,
        -3.31054348e-04,  1.56413897e-13,  9.35963643e-13,
        -3.14497538e-17,  3.32249252e-04, -9.30192832e-13,
        -2.86772424e-04, -6.87632323e-12,  2.86772411e-04,
        -6.89189630e-12],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         4.45023028e-01, -2.37382503e-15, -4.86244612e-14,
         4.00940538e-19, -4.45023029e-01,  4.84973448e-14,
         7.08416761e-07,  2.26124343e-12, -7.08412246e-07,
         2.26153390e-12],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  2.86305641e-04,  5.34123555e-03,
         8.79710102e-17,  6.50741734e-17, -5.33219700e-03,
         9.38121974e-05, -9.40818028e-05,  9.38121974e-05,
        -9.40818028e-05],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  4.45148070e-01,
         8.89579369e-17,  2.77426957e-18, -4.45147989e-01,
         2.25612180e-06, -2.25225617e-06,  2.25612180e-06,
        -2.25225611e-06],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         8.90797779e-07,  3.30208354e-17, -5.50422933e-16,
         6.69069674e-11, -2.39824923e-04,  6.69703451e-11,
         2.39825057e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  9.20966596e-07,  4.28717908e-15,
        -2.45651230e-04, -6.04297035e-12,  2.45651218e-04,
        -5.97337663e-12],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  7.70802831e-06,
         9.36014677e-05, -9.38397080e-05,  9.36014677e-05,
        -9.38397080e-05],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         3.14688911e-01,  3.14688578e-01,  3.14688407e-01,
         3.14688578e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  4.53531182e-04,  3.49713599e-04,
         1.21659720e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  4.41488986e-04,
         2.62883032e-04],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         3.48972717e-04]])
#T = np.array([[ 1.27436392e+06,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  1.27436392e+06,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  0.00000000e+00,  1.27436392e+06,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         1.41021986e+05, -4.92609895e+02, -3.81950609e-06,
#        -1.39881816e-07,  1.46965101e-05, -1.39002541e-05,
#        -1.83836102e+05,  1.68706469e-05, -9.15069799e-06,
#         1.11994248e-05, -9.53917896e+03,  7.55620346e+03,
#         1.00307305e+04],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  1.05974708e+01, -2.82895333e-12,
#        -1.74296046e-12,  1.65159019e-09,  1.44704734e-08,
#         5.29285820e+06, -1.89772983e-04,  3.77959313e-04,
#        -1.12558492e-03,  2.79883660e+06, -2.21702296e+06,
#        -2.94305996e+06],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  1.36343247e+05,
#         1.01426268e+02, -7.44859158e-05,  6.07077621e-07,
#         4.81357384e-06, -1.76897756e+05,  8.36517673e-05,
#        -1.38414150e+01,  9.60404766e+03,  1.21244392e+04,
#         2.78494647e-05],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         1.05946571e+01,  8.78427055e-11,  1.03271051e-13,
#        -4.76856544e-12,  5.11947599e+06, -2.84743503e-03,
#         3.99634535e+03, -2.77291674e+06, -3.50061365e+06,
#        -4.23970437e-03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  1.64679478e+04, -1.97595350e+02,
#        -1.60656477e-06, -1.16300625e-06, -1.93085039e+04,
#         8.35287536e-01, -1.15950269e+03,  9.18469753e+02,
#        -1.79456999e+03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  1.05916810e+01,
#        -1.05771940e-09, -3.19058025e-11,  6.11682433e+05,
#        -1.81939672e+02,  2.52803922e+05, -2.00252019e+05,
#         3.91266304e+05],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         5.29285824e+06, -1.89773007e-04,  3.77957946e-04,
#        -1.12559126e-03,  2.79883661e+06, -2.21702297e+06,
#        -2.94305997e+06],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  5.11947598e+06, -2.84743509e-03,
#         3.99634536e+03, -2.77291675e+06, -3.50061366e+06,
#        -4.23970283e-03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  6.11682543e+05,
#        -1.81939629e+02,  2.52803862e+05, -2.00251972e+05,
#         3.91266212e+05],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         1.49826263e+01, -1.03958924e+04, -2.44462241e+03,
#        -8.04489900e+03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  1.03959034e+04, -8.23483464e+03,
#         2.57910019e+03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  1.06794654e+04,
#        -8.04489899e+03],
#       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         1.35107019e+04]])
#T = T / T[0,0] # keep xp equal
#T_inv = scipy.linalg.inv(T)
P = P_sqrt_T.T @ P_sqrt_T
runs=list([lis.simulate_random(L=L, x0=normalize(x0 * np.random.uniform(low=-1, high=1, size=x0.shape)), scale_timing=1) for _ in range(NUM_RUNS)])
# -> x(k=time_index)[state_index] == runs[run_number][state_index, time_index]
#%%



def c1_c2(P_sqrt_T, M=None):
    """
    return (c1, c2) such that
    c1 sqrt(V(Mx)) <= |x|   and   |Mx| <= c2 sqrt(V(x))
    for all x, where
    V(x) = x.T P_sqrt_T.T P_sqrt_T x.

    If M is not given, M=I is used, resulting in
    c1 sqrt(V(x)) <= |x| <= c2 sqrt(V(x)).
    """
    return (NumpyMatrix.qlf_upper_bound(P_sqrt_T, M), NumpyMatrix.qlf_lower_bound(P_sqrt_T, M))
c1_T, c2_T = c1_c2(P_sqrt_T, Select_xp)

max_mag = np.zeros((L,))
first = True
for data in runs:
    data=data[0]
    plt.yscale('log')
    mag = np.sqrt(np.sum((C @ data) ** 2, 0))
    plt.plot(mag, 'k', label="|C*x|" if first else "_nolabel_")
    max_mag = np.maximum(max_mag, mag)
    plt.plot(np.sqrt(np.sum((data) ** 2, 0)), 'r', label="|x|" if first else "_nolabel_")
    sqrt_lyap = np.apply_along_axis(lambda x: np.sqrt(x.T @ P @ x), 0, data).squeeze()
    plt.plot(sqrt_lyap * c2_T, 'm', label="c2' $\sqrt{V(x)}$" if first else "_nolabel_")
    #plt.plot(np.log(sqrt_lyap/sqrt_lyap[0]))
    first = False
plt.ylabel('|C * x|')
plt.xlabel('k')
#plt.plot(np.log10(max_mag), 'c', linewidth=3)
plt.plot(max_mag[0] * (c2_T / c1_T) * (0.915 ** np.arange(L)), 'b', linewidth=3, label='exp. bound with timing')
plt.plot(max_mag[0] * (c2_T / c1_T) * (0.86 ** np.arange(L)), 'b:', linewidth=3, label='exp. bound w.o. timing')
plt.legend()


#%%




#%% Other experiments

if False:
    plt.figure()
    for data in runs:
        plt.gca().set_prop_cycle(None) # reset colors
        # plt.plot(data.T) # plot states
        plt.plot(np.sqrt(np.sum(data ** 2, 0))) # plot magnitude
    plt.title("Initial response for random timing")
    plt.legend(list(lis.state_names()))
if False:
    plt.figure()
    for data in runs:
        for i in range(lis.n):
            for j in range(lis.n):
                plt.subplot(lis.n, lis.n, i*lis.n + j + 1)
                plt.plot(data[i,:], data[j,:], '.', markersize=1)
    plt.title("Correlation plots for extended state vector")


#%% Observability experiment - work in progress
S=np.zeros((lis.n, lis.n))
for i in range(lis.sys.n_p):
    S[i,i]=1
def nonzero_rows(M):
    return M[(M != 0).any(axis=1), :]
# Kalman observability matrix
obs = lambda n: np.vstack([nonzero_rows(S@np.linalg.matrix_power(lis.Ak_nominal, i)) for i in range(n)])
print(np.linalg.matrix_rank(obs(6)))