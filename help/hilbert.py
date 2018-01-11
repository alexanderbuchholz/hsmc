# -*- coding: utf-8 -*-

""" Wrapper for Numba+Hilbert 

"""

from numpy import * 

#numba 
from numba import jit
from numba.types import int64

# the numba'ized module TODO find a better name 
import numba_hilbert_jitted as numba_hilbert

#@jit(nopython=True)
def invlogit(x): 
    return 1./(1.+exp(-x))


@jit(nopython=True)
def hilbert_array(xint):
    """ input: a [N,d] array of ints, with d>=2 
        output: a [N,] array of Hilbert indices 
    """
    N, d = xint.shape
    h = zeros(N, int64)
    for n in range(N):
        h[n] = numba_hilbert.Hilbert_to_int(xint[n, :])
    return h

@jit
def brutehilbertsort(x):
    """ input: array x[N] or x[N,d]
        output: argsort of Hilbert sort of x
    """
    d = 1 if x.ndim==1 else x.shape[1] 
    if d==1: 
        return argsort(x,axis=0)
    xs = invlogit((x-mean(x,axis=0))/std(x,axis=0)) 
    maxint = floor(2**(62 / d))
    xint = floor(xs * maxint).astype('int')
    return argsort(hilbert_array(xint))                            
