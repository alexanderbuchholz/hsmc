# run simulations on server for student distribution

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
dim_list = [10, 20, 50, 100, 200]

try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 10
N_particles = 2**10
T_time = 20
move_steps_hmc = 20
move_steps_rw_mala = 100
ESStarget = 0.9
M_num_repetions = 40
epsilon = 1.
epsilon_hmc = .1
verbose = False
targetmean = np.ones(dim)*2.
targetvariance = (np.diag(np.linspace(start=0.01, stop=100, num=dim)) +0.7*np.ones((dim, dim)))
#targetvariance = ((np.diag(np.linspace(start=1, stop=2, num=dim)) +0.7*np.ones((dim, dim))))
targetvariance_inv = np.linalg.inv(targetvariance)
l_targetvariance_inv = np.linalg.cholesky(targetvariance_inv)
parameters = {'dim' : dim, 
              'N_particles' : N_particles, 
              'targetmean': targetmean, 
              'mean_shift' : np.ones(dim)*1,
              'targetvariance':targetvariance,
              'det_targetvariance' : np.linalg.det(targetvariance),
              'targetvariance_inv':targetvariance_inv,
              'l_targetvariance_inv':l_targetvariance_inv,
              'df' : 5
             }




# prepare the kernels and specify parameters
from help.f_rand_seq_gen import random_sequence_qmc, random_sequence_rqmc, random_sequence_mc
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
                      'tune_kernel': 'fearnhead_taylor',
                      'sample_eps_L' : True,
                      'verbose' : verbose,
                      'move_steps': move_steps_rw_mala,
                      'T_time' : T_time,
                      'autotempering' : True,
                      'ESStarget': ESStarget,
                      'adaptive_covariance' : True,
                      'quantile_test': 0.5
                      }

rwdict = {'proposalkernel_tune': proposalrw,
                      'proposalkernel_sample': proposalrw,
                      'proposalname' : 'RW',
                      'target_probability' : 0.3,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 1,
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
                      'quantile_test': 0.5
                      }

hmcdict_ft_adaptive = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_L_random_ft_adaptive',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 100,
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
                      'quantile_test': 0.5
                      }

hmcdict_ft_non_adaptive = copy.copy(hmcdict_ft_adaptive)
hmcdict_ft_non_adaptive['autotempering'] = False

hmcdict_ours_adaptive = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_L_random_ours_adaptive',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 100,
                      'epsilon' : np.array([epsilon_hmc]),
                      'epsilon_max' : np.array([epsilon_hmc]),
                      'accept_reject' : True,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False,
                      'verbose' : verbose,
                      'move_steps': move_steps_hmc, 
                      'mean_L' : False,
                      'T_time' : T_time,
                      'autotempering' : True,
                      'ESStarget': ESStarget,
                      'adaptive_covariance' : True,
                      'quantile_test': 0.5
                      }
hmcdict_ours_non_adaptive = copy.copy(hmcdict_ours_adaptive)
hmcdict_ours_non_adaptive['autotempering'] = False







if __name__ == '__main__':

    from smc_sampler_functions.functions_smc_main import repeat_sampling
    samplers_list_dict_adaptive = [hmcdict_ft_adaptive, hmcdict_ours_adaptive, rwdict, maladict]
    samplers_list_dict_non_adaptive = []

    # define the target distributions
    from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
    from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student


    priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
    targetdistribution1 = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}

    target_dist_list = [targetdistribution1]
    for target_dist in target_dist_list: 
        temperedist = sequence_distributions(parameters, priordistribution, target_dist)
        res_repeated_sampling_adaptive, res_first_iteration_adaptive = repeat_sampling(samplers_list_dict_adaptive, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = target_dist['target_name'])

        
        T_time_non_adaptive = np.ceil(res_repeated_sampling_adaptive['temp_steps'].mean(axis=1))[0]
        hmcdict_ft_non_adaptive['T_time'] = T_time_non_adaptive
        hmcdict_ft_non_adaptive['proposalname'] = 'HMC_L_random_ft_non_adaptive'
        hmcdict_ours_non_adaptive['T_time'] = T_time_non_adaptive
        hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
        samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
        res_repeated_sampling_non_adaptive, res_first_iteration_non_adaptive = repeat_sampling(samplers_list_dict_non_adaptive, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = target_dist['target_name']+'_non_adaptive')

        #import ipdb; ipdb.set_trace()