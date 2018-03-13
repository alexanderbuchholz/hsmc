# run simulations on server for log cox model

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
dim_list = [10**2, 20**2, 30**2, 64**2]

try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 30**2


def prepare_samplers(dim):
    N_particles = 2**10
    T_time = 20
    move_steps_hmc = 60
    move_steps_rw_mala = 250
    ESStarget = 0.5
    M_num_repetions = 40
    epsilon = 1.
    epsilon_hmc = .1
    verbose = False
    parameters = {'dim' : dim, 
                'N_particles' : N_particles}




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
    hmcdict_ft_non_adaptive['proposalname'] = 'HMC_L_random_ft_non_adaptive'
    hmcdict_ft_non_adaptive['quantile_test'] = 0.0001


    hmcdict_ours_adaptive = {'proposalkernel_tune': proposalhmc,
                        'proposalkernel_sample': proposalhmc_parallel,
                        'proposalname' : 'HMC_L_random_ours_adaptive',
                        'target_probability' : 0.9,
                        'covariance_matrix' : np.eye(dim), 
                        'L_steps' : 100,
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
                        'quantile_test': 0.5
                        }
    hmcdict_ours_non_adaptive = copy.copy(hmcdict_ours_adaptive)
    hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
    hmcdict_ours_non_adaptive['quantile_test'] = 0.0001


    return(parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive, hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive)




if __name__ == '__main__':
    M_num_repetions = 1
    parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive, hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive = prepare_samplers(dim)
    from smc_sampler_functions.functions_smc_main import repeat_sampling
    samplers_list_dict_adaptive = [hmcdict_ft_adaptive]#, hmcdict_ours_adaptive, rwdict, maladict]
    samplers_list_dict_non_adaptive = []

    # define the target distributions
    from smc_sampler_functions.target_distributions_logcox import priorlogdens_log_cox, priorgradlogdens_log_cox, priorsampler_log_cox
    from smc_sampler_functions.target_distributions_logcox import f_dict_log_cox, targetlogdens_log_cox, targetgradlogdens_log_cox


    priordistribution = {'logdensity' : priorlogdens_log_cox, 'gradlogdensity' : priorgradlogdens_log_cox, 'priorsampler': priorsampler_log_cox}
    targetdistribution1 = {'logdensity' : targetlogdens_log_cox, 'gradlogdensity' : targetgradlogdens_log_cox, 'target_name': 'log_cox'}

    parameters_log_cox = f_dict_log_cox(int(dim**0.5))
    parameters.update(parameters_log_cox)

    target_dist_list = [targetdistribution1]
    for target_dist in target_dist_list: 
        temperedist = sequence_distributions(parameters, priordistribution, target_dist)
        res_repeated_sampling_adaptive, res_first_iteration_adaptive = repeat_sampling(samplers_list_dict_adaptive, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_res_intermediate=True, save_name = target_dist['target_name'])
        import ipdb; ipdb.set_trace()
        adjusted_steps = int(np.ceil(res_repeated_sampling_adaptive['temp_steps'].mean(axis=1)/res_repeated_sampling_adaptive['temp_steps_single'].mean(axis=1))[0])
        hmcdict_ours_non_adaptive['move_steps'] = adjusted_steps
        hmcdict_ft_non_adaptive['move_steps'] = adjusted_steps
        samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
        res_repeated_sampling_non_adaptive, res_first_iteration_non_adaptive = repeat_sampling(samplers_list_dict_non_adaptive, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_res_intermediate=True, save_name = target_dist['target_name']+'_non_adaptive')

        