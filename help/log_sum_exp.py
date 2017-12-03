'''
help function
'''
import numpy as np  # Thinly-wrapped numpy
#import ipdb as pdb
from numba import jit


@jit(nopython=True)
def logplus_one(x):
    """
    log sum exp trick
    """
    max_x = np.max(x)
    res = max_x + np.log(1+np.sum(np.exp(x-max_x), axis=0))
    return res

@jit(nopython=True)
def logplus(x):
    """
    log sum exp trick
    """
    max_x = np.max(x)
    res = max_x + np.log(np.sum(np.exp(x-max_x), axis=0))
    return res


def logplus_nojit(x):
    return np.log(np.exp(x).sum(axis=0))

@jit
def logplus_autojit(x):
    """
    version with autojit, much faster
    """
    return np.log(np.exp(x).sum(axis=0))


if __name__ == '__main__':
    x = np.random.random(size=(100))
    print np.log(np.exp(x).sum())
    print logplus(x)