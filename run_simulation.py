# Notebook for smc sampler 
# Notebook for smc sampler 
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma

import sys
import os

from smc_sampler_functions.functions_smc_help import sequence_distributions


# define the parameters
dim_list = [2, 5, 10, 20, 50, 100, 200, 300]
try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 10
N_particles = 2**10
T_time = 50
move_steps = 10
ESStarget = 0.9
M_num_repetions = 20
#rs = np.random.seed(1)
targetmean = np.ones(dim)*4
targetvariance = np.eye(dim)*0.1
targetvariance_inv = np.linalg.inv(targetvariance)
l_targetvariance_inv = np.linalg.cholesky(targetvariance_inv)
parameters = {'dim' : dim, 
              'N_particles' : N_particles, 
              'targetmean': targetmean, 
              'targetvariance':targetvariance,
              'targetvariance_inv':targetvariance_inv,
              'l_targetvariance_inv':l_targetvariance_inv,
              'df' : 5,
              'T_time' : T_time,
              'autotempering' : True,
              'move_steps': move_steps,
              'ESStarget': ESStarget,
              'adaptive_covariance' : True
             }


# define the target distributions
#from smc_sampler_functions.cython.cython_target_distributions import priorlogdens, priorgradlogdens
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens
from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student
#from smc_sampler_functions.target_distributions import targetlogdens_student as targetlogdens_student_py
#from smc_sampler_functions.target_distributions import targetgradlogdens_student as targetgradlogdens_student_py

#import ipdb; ipdb.set_trace()
#particles_test = np.random.randn(N_particles, dim)
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens}
#targetdistribution = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}
targetdistribution = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}

temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)

# prepare the kernels and specify parameters
from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_parallel
from smc_sampler_functions.functions_smc_main import smc_sampler

maladict = {'proposalkernel_tune': proposalmala,
                      'proposalkernel_sample': proposalmala,
                      'proposalname' : 'MALA',
                      'target_probability' : 0.65,
                      'covariance_matrix' : np.eye(dim), 
                      'epsilon' : 1.,
                      'epsilon_max' : 1.,
                      'tune_kernel': True,
                      'sample_eps_L' : True
                      }
rwdict = {'proposalkernel_tune': proposalrw,
                      'proposalkernel_sample': proposalrw,
                      'proposalname' : 'RW',
                      'target_probability' : 0.3,
                      'covariance_matrix' : np.eye(dim), 
                      'epsilon' : 1.,
                      'epsilon_max' : 1.,
                      'tune_kernel': True,
                      'sample_eps_L' : True
                      }

hmcdict1 = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_L_random',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 50,
                      'epsilon' : 1.,
                      'epsilon_max' : 1.,
                      'accept_reject' : True,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False
                      }

hmcdict2 = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc,
                      'proposalname' : 'HMC',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 50,
                      'epsilon' : 1.,
                      'epsilon_max' : 1.,
                      'accept_reject' : True,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False
                      }



#print temperatures
#import yappi
#yappi.start()
# sample and compare the results
#res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict)
#res_dict_mala = smc_sampler(temperedist,  parameters, maladict)
#res_dict_rw = smc_sampler(temperedist,  parameters, rwdict)
#yappi.get_func_stats().print_all()

#from functions_smc_plotting import plot_results_single_simulation
#plot_results_single_simulation([res_dict_hmc, res_dict_mala, res_dict_rw])
#import yappi
#yappi.start()
# sample and compare the results
#res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict1)
#yappi.get_func_stats().print_all()

#yappi.start()
#res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict2)
#yappi.get_func_stats().print_all()


import ipdb; ipdb.set_trace()
if __name__ == '__main__':
    from smc_sampler_functions.functions_smc_main import repeat_sampling
    samplers_list_dict = [hmcdict1, hmcdict2, maladict, rwdict]
    #samplers_list_dict = [hmcdict1, hmcdict2]
    res_repeated_sampling, res_first_iteration = repeat_sampling(samplers_list_dict, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = targetdistribution['target_name'])
    from smc_sampler_functions.functions_smc_plotting import plot_repeated_simulations, plot_results_single_simulation
    plot_repeated_simulations(res_repeated_sampling)
    plot_results_single_simulation(res_first_iteration)
