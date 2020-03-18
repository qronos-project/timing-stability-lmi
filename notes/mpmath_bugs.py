# The following are bugs encountered in the python mpmath library.
# TODO report these bugs


# Not really a bug, but still important to know:
# For python2, mp.matrix has arbitrary ordering, just like object. In Python3, this would be an error.
#print([object()   > object()   for i in range(10)]) 
# -> [True, False, True, False, True, False, True, False, True, False]
#print([mp.matrix([1])   > mp.matrix([1])   for i in range(10)])
# -> [False, True, False, True, False, True, False, True, False, True]
#print([mp.matrix([1])[0]   > mp.matrix([1])[0]   for i in range(10)])
# -> [False, False, False, False, False, False, False, False, False, False]

# mpmath.memoize is broken for the mpmath.iv.matrix datatype.
# It may possibly be fixed by adding __pos__() (see below),
# but even then no speedup could be measured at all.


# add missing support for unary plus for matrices (bugfix for memoize)
# TODO: send that upstream to mpmath
if not hasattr(iv.matrix, '__pos__'):
    iv.matrix.__pos__ = lambda x: 1 * x



mpmath.mpi(-1, 1) * mpmath.iv.ones(2, 3)
# -> TypeError: 'NoneType' object is not iterable
mpmath.iv.ones(2, 3) * mpmath.mpi(-1, 1)
# -> works.

mpmath.iv.randmatrix(4)
# -> AttributeError: 'MPIntervalContext' object has no attribute 'rand'

mpmath.iv.matrix(mpmath.randmatrix(1))[0,0]
# -> mpf (no interval!)

( mpmath.iv.matrix(mpmath.randmatrix(1))**2 )[0,0]
# -> mpi (interval!)

max([( mpmath.iv.matrix(mpmath.randmatrix(1))**2 )[0,0].delta.b for i in range(4000)])
# -> always exact???

# mpmath's memoize is broken for interval matrices, see ../src/qronos/lis/memoize_simple.py and ../src/qronos/lis/iv_matrix_utils.py

# TODO: some more bugs are noted (and worked around) in src/qronos/lis/norms.py
