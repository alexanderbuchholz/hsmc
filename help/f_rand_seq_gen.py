# -*- coding: utf-8 -*-
import cProfile
import numpy.random as nr

import numpy as np
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from scipy.stats import itemfreq, gamma, norm
import random
#import ipdb as pdb
#from numba import jit

randtoolbox = rpackages.importr('randtoolbox')

def test_random(u):
    tol = 10**-16
    if (u+tol>1.).any() or (u-tol<0.).any():
        print("ERROR: outside 0 1 ! ")
        return True
    else: 
        return False


def random_sequence_qmc(size_mv, i, n=1, randomized=True):
    """
    generates QMC sequence
    now randomized
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    #u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=0, seed=random_seed)).reshape((n,size_mv))
    if randomized:
        shift = np.random.rand(1,size_mv)
        u = np.mod(np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0))).reshape((n,size_mv)) + shift, 1)
    else: 
        u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0))).reshape((n,size_mv))
    
    while test_random(u):
        random_seed = random.randrange(10**9)
        #u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=0, seed=random_seed)).reshape((n,size_mv))
        if randomized:
            shift = np.random.rand(1,size_mv)
            u = np.mod(np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0))).reshape((n,size_mv)) + shift, 1)
        else: 
            u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0))).reshape((n,size_mv))
        
    return(u)



def random_sequence_rqmc(size_mv, i, n=1):
    """
    generates RQMC random sequence
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=1, seed=random_seed)).reshape((n,size_mv))
    # randtoolbox for sobol sequence
    #pdb.set_trace()
    while test_random(u):
        random_seed = random.randrange(10**9)
        u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=1, seed=random_seed)).reshape((n,size_mv))

    return(u)

#@jit
#@profile
def random_sequence_mc(size_mv, i=None,n=1):
    """
    generates MC random sequence for the movement of particles
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u = np.asarray(nr.uniform(size=size_mv*n).reshape((n,size_mv)))
    #pdb.set_trace()
    test_random(u)
    return(u)

