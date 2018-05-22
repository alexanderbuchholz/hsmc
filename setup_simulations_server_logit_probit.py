# run simulations on server for logit probit distribution

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
import copy

from smc_sampler_functions.functions_smc_help import sequence_distributions


# define the parameters
dim_list = [2, 25, 31, 301]

try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 25

def prepare_samplers(dim):
    N_particles = 2**10
    T_time = 20
    move_steps_hmc = 100
    move_steps_rw_mala = 1000
    ESStarget = 0.5
    M_num_repetions = 1
    epsilon = 1.
    epsilon_hmc = .1
    verbose = False
    factor_variance = 5.
    parameters = {'dim' : dim, 
                'N_particles' : N_particles,
                'factor_variance': factor_variance}




    # prepare the kernels and specify parameters
    from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_parallel
    from smc_sampler_functions.functions_smc_main import smc_sampler


    maladict = {'proposalkernel_tune': proposalmala,
                        'proposalkernel_sample': proposalmala,
                        'proposalname' : 'MALA',
                        'target_probability' : 0.65,
                        'covariance_matrix' : np.eye(dim), 
                        'L_max' : 1,
                        'epsilon' : np.array([epsilon]),
                        'epsilon_max' : np.array([epsilon]),
                        'tune_kernel': 'fearnhead_taylor',
                        'sample_eps_L' : True,
                        'verbose' : verbose,
                        'move_steps': move_steps_rw_mala,
                        'T_time' : T_time,
                        'autotempering' : True,
                        'ESStarget': ESStarget,
                        'adaptive_covariance' : True,
                        'quantile_test': 0.1
                        }

    rwdict = {'proposalkernel_tune': proposalrw,
                        'proposalkernel_sample': proposalrw,
                        'proposalname' : 'RW',
                        'target_probability' : 0.3,
                        'covariance_matrix' : np.eye(dim), 
                        'L_max' : 1,
                        'epsilon' : np.array([epsilon]),
                        'epsilon_max' : np.array([epsilon]),
                        'tune_kernel': 'fearnhead_taylor',
                        'sample_eps_L' : True,
                        'verbose' : verbose,
                        'move_steps': move_steps_rw_mala,
                        'T_time' : T_time,
                        'autotempering' : True,
                        'ESStarget': ESStarget,
                        'adaptive_covariance' : True,
                        'quantile_test': 0.1
                        }

    hmcdict_ft_adaptive = {'proposalkernel_tune': proposalhmc,
                        'proposalkernel_sample': proposalhmc_parallel,
                        'proposalname' : 'HMC_L_random_ft_adaptive',
                        'target_probability' : 0.9,
                        'covariance_matrix' : np.eye(dim), 
                        'L_max' : 25,
                        'epsilon' : np.array([epsilon_hmc]),
                        'epsilon_max' : np.array([epsilon_hmc]),
                        'accept_reject' : True,
                        'tune_kernel': 'fearnhead_taylor',
                        'sample_eps_L' : True,
                        'parallelize' : False,
                        'verbose' : verbose,
                        'move_steps': move_steps_hmc, 
                        'mean_L' : False,
                        'T_time' : T_time,
                        'autotempering' : True,
                        'ESStarget': ESStarget,
                        'adaptive_covariance' : True,
                        'quantile_test': 0.1
                        }

    hmcdict_ours_adaptive = {'proposalkernel_tune': proposalhmc,
                        'proposalkernel_sample': proposalhmc_parallel,
                        'proposalname' : 'HMC_L_random_ours_adaptive',
                        'target_probability' : 0.9,
                        'covariance_matrix' : np.eye(dim), 
                        'L_max' : 25,
                        'epsilon' : np.array([epsilon_hmc]),
                        'epsilon_max' : np.array([epsilon_hmc]),
                        'accept_reject' : True,
                        'tune_kernel': 'ours_simple',
                        'sample_eps_L' : True,
                        'parallelize' : False,
                        'verbose' : verbose,
                        'move_steps': move_steps_hmc, 
                        'mean_L' : False,
                        'T_time' : T_time,
                        'autotempering' : True,
                        'ESStarget': ESStarget,
                        'adaptive_covariance' : True,
                        'quantile_test': 0.1
                        }

    return(parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive)





if __name__ == '__main__':

    from smc_sampler_functions.functions_smc_main import repeat_sampling
    samplers_list_dict_adaptive = [hmcdict_ft_adaptive, hmcdict_ours_adaptive, rwdict, maladict]

    # define the target distributions
    from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
    from smc_sampler_functions.target_distributions import targetlogdens_logistic, targetgradlogdens_logistic, f_dict_logistic_regression
    from smc_sampler_functions.target_distributions import targetlogdens_probit, targetgradlogdens_probit


    priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
    targetdistribution = {'logdensity' : targetlogdens_logistic, 'gradlogdensity' : targetgradlogdens_logistic, 'target_name': 'logistic'}

    parameters_logistic = f_dict_logistic_regression(dim)
    parameters.update(parameters_logistic)

    target_dist_list = [targetdistribution]
    for target_dist in target_dist_list: 
        temperedist = sequence_distributions(parameters, priordistribution, target_dist)
        res_repeated_sampling_adaptive, res_first_iteration_adaptive = repeat_sampling(samplers_list_dict_adaptive, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_res_intermediate=False, save_name = target_dist['target_name'])

        

        #import ipdb; ipdb.set_trace()