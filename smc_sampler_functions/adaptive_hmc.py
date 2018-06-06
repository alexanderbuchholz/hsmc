# adaptive HMC based on the paper of wang, de freitas and mohammed

import numpy as np
from functools import partial
import GPy
import GPyOpt
import sys
import copy
sys.path.append("/home/alex/Dropbox/smc_hmc/python_smchmc/")
import matplotlib.pyplot as plt
from smc_sampler_functions.standard_mh_sampler import parallel_mh_sampler

def target_function(hmc_params, temperedist, parameters, hmcdict):
    """
    the target function for the bayesian optimization
    """
    T = hmc_params.shape[0]
    jumping_dist = np.zeros((T,1))
    for t in range(T):
        print t
        hmcdict['epsilon'] = np.atleast_2d(np.array(hmc_params[t, 0]))
        hmcdict['L_steps'] = np.atleast_2d(np.array(hmc_params[t, 1]))
        res = parallel_mh_sampler(temperedist, parameters, hmcdict)
        jumping_dist[t,:] = res['ESJD']/np.sqrt(hmc_params[t, 1])
    return(-jumping_dist)


def bayes_opt_hmc(bounds, temperedist, parameters, hmcdict):

    #bounds = [(0.0001,1.),(5,100)]
    # we use the same kernel as in wang et al.
    alpha = 0.2
    lengthscale = (np.array((bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0]))*alpha)**2
    ## Define the kernel
    k = GPy.kern.RBF(2, variance=2.0, lengthscale=lengthscale, ARD=True)
    max_iter = 30
    # make functions partial for optimisation

    hmc_info = "sample constrained wishart first training"
    ################################################################################################################################################

    # define HMC objective for sampler
    hmc_obj = partial(target_function, temperedist=temperedist, parameters=parameters, hmcdict=hmcdict)

    myProblem = GPyOpt.methods.BayesianOptimization(hmc_obj, bounds, acquisition='LCB', acquisition_par = 2.,  normalize= False, kernel=k)    # Normalize the acquisition funtction))
    myProblem.run_optimization(max_iter, acqu_optimize_method = 'fast_random',    # method to optimize the acq. function
                    acqu_optimize_restarts = 10,             # number of local optimizers
                    eps=10e-6)

    # show output of first optimzation
    eps, L =  myProblem.x_opt
    L = int(L)
    print myProblem.fx_opt
    print myProblem.x_opt
    return([eps, L], myProblem)

if __name__ == '__main__':
    bounds = [(0.0001,1.),(5,100)]

    bayes_opt_hmc(bounds, temperedist, parameters, hmcdict)