# simulations server
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
dim_list = [2, 5, 10, 20, 31, 50, 100, 200, 300]
try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 2
N_particles = 2**10
T_time = 1000
move_steps_hmc = 10
move_steps_rw_mala = 50
ESStarget = 0.95
M_num_repetions = 100
epsilon = .1
epsilon_hmc = .1
verbose = False
#rs = np.random.seed(1)
targetmean = np.ones(dim)*8.
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
              'ESStarget': ESStarget,
              'adaptive_covariance' : True
             }



# define the target distributions
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens}

# prepare the kernels and specify parameters
from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_parallel
from smc_sampler_functions.functions_smc_main import smc_sampler


maladict = {'proposalkernel_tune': proposalmala,
                      'proposalkernel_sample': proposalmala,
                      'proposalname' : 'MALA',
                      'target_probability' : 0.65,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 1,
                      'epsilon' : np.array([epsilon]),
                      'epsilon_max' : np.array([epsilon]),
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'verbose' : verbose,
                      'move_steps': move_steps_rw_mala
                      }
rwdict = {'proposalkernel_tune': proposalrw,
                      'proposalkernel_sample': proposalrw,
                      'proposalname' : 'RW',
                      'target_probability' : 0.3,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 1,
                      'epsilon' : np.array([epsilon]),
                      'epsilon_max' : np.array([epsilon]),
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'verbose' : verbose,
                      'move_steps': move_steps_rw_mala
                      }

hmcdict1 = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_L_random',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 50,
                      'epsilon' : np.array([epsilon_hmc]),
                      'epsilon_max' : np.array([epsilon_hmc]),
                      'accept_reject' : True,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False,
                      'verbose' : verbose,
                      'move_steps': move_steps_hmc
                      }

hmcdict2 = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc,
                      'proposalname' : 'HMC',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 50,
                      'epsilon' : np.array([epsilon_hmc]),
                      'epsilon_max' : np.array([epsilon_hmc]),
                      'accept_reject' : True,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False,
                      'verbose' : verbose,
                      'move_steps': move_steps_hmc
                      }



if __name__ == '__main__':

    from smc_sampler_functions.functions_smc_main import repeat_sampling
    samplers_list_dict = [hmcdict2, hmcdict1, rwdict, maladict]

    from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
    from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student
    from smc_sampler_functions.target_distributions import targetlogdens_logistic, targetgradlogdens_logistic, f_dict_logistic_regression
    parameters_logistic = f_dict_logistic_regression(dim)
    parameters.update(parameters_logistic)

    priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens}
    targetdistribution1 = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}
    targetdistribution2 = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}
    targetdistribution3 = {'logdensity' : targetlogdens_logistic, 'gradlogdensity' : targetgradlogdens_logistic, 'target_name': 'logistic'}

    target_dist_list = [targetdistribution1, targetdistribution2, targetdistribution3]
    for target_dist in target_dist_list: 
        temperedist = sequence_distributions(parameters, priordistribution, target_dist)
        res_repeated_sampling, res_first_iteration = repeat_sampling(samplers_list_dict, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = target_dist['target_name'])
        #from smc_sampler_functions.functions_smc_plotting import plot_repeated_simulations, plot_results_single_simulation
        #plot_repeated_simulations(res_repeated_sampling)
        #plot_results_single_simulation(res_first_iteration)

