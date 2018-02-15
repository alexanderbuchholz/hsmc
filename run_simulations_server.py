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
#dim_list = [2, 5, 10, 20, 31, 50, 100, 200, 300]
#dim_list = [2, 5, 10, 20, 30, 50, 100, 200, 300]
dim_list = [900, 4086]
#dim_list = [5**2, 10**2, 15**2, 20**2, 30**2, 40**2, 50**2, 64**2]
try:
    dim = dim_list[int(sys.argv[1])-1]
except:
    dim = 31
N_particles = 2**10
T_time = 20
move_steps_hmc = 20
move_steps_rw_mala = 100
ESStarget = 0.9
M_num_repetions = 40
epsilon = 1.
epsilon_hmc = .1
verbose = False
#rs = np.random.seed(1)
targetmean = np.ones(dim)*2.
#targetvariance = np.eye(dim)*0.1
#targetvariance = (0.1*(np.diag(np.linspace(start=0.01, stop=100, num=dim))/float(dim) +0.7*np.ones((dim, dim))))
#targetvariance = (np.diag(np.linspace(start=0.01, stop=100, num=dim)) +0.7*np.ones((dim, dim)))
targetvariance = ((np.diag(np.linspace(start=1, stop=2, num=dim)) +0.7*np.ones((dim, dim))))
targetvariance_inv = np.linalg.inv(targetvariance)
l_targetvariance_inv = np.linalg.cholesky(targetvariance_inv)
parameters = {'dim' : dim, 
              'N_particles' : N_particles, 
              'targetmean': targetmean, 
              'mean_shift' : np.ones(dim)*1,
              'targetvariance':targetvariance,
              'targetvariance_inv':targetvariance_inv,
              'l_targetvariance_inv':l_targetvariance_inv,
              'df' : 5,
              'T_time' : T_time,
              'autotempering' : True,
              'ESStarget': ESStarget,
              'adaptive_covariance' : True,
              'quantile_test': 0.9
             }




# prepare the kernels and specify parameters
from help.f_rand_seq_gen import random_sequence_qmc, random_sequence_rqmc, random_sequence_mc
from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_parallel, proposalhmc_is
from smc_sampler_functions.functions_smc_main import smc_sampler
from smc_sampler_functions.functions_smc_is_main import smc_sampler_is_qmc


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
                      'tune_kernel': 'fearnhead_taylor',
                      'sample_eps_L' : True,
                      'verbose' : verbose,
                      'move_steps': move_steps_rw_mala
                      }

hmcdict1 = {'proposalkernel_tune': proposalhmc_parallel,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_L_random_ft',
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
                      'mean_L' : False
                      }

hmcdict3 = copy.copy(hmcdict1)
hmcdict3['proposalname'] = 'HMC_L_random'
hmcdict3['tune_kernel'] = True

hmcdict2 = {'proposalkernel_tune': proposalhmc_parallel,
                      'proposalkernel_sample': proposalhmc_parallel,
                      'proposalname' : 'HMC_ft',
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
                      'mean_L' : True
                      }

hmcdict4 = copy.copy(hmcdict2)
hmcdict4['proposalname'] = 'HMC'
hmcdict4['tune_kernel'] = True


hmcdict_is_mc = {'proposalkernel_tune': proposalhmc_is,
                      'proposalkernel_sample': proposalhmc_is,
                      'proposalname' : 'HMC IS MC',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 100,
                      'epsilon' : np.array([epsilon_hmc]),
                      'epsilon_max' : np.array([epsilon_hmc]),
                      'accept_reject' : False,
                      'tune_kernel': True,
                      'sample_eps_L' : True,
                      'parallelize' : False,
                      'verbose' : verbose,
                      'move_steps': move_steps_hmc,
                      'mean_L' : True,
                      'unif_sampler' : random_sequence_mc, 
                      'trajectory_selector_energy' : True
                      }
hmcdict_is_qmc = copy.copy(hmcdict_is_mc)
hmcdict_is_qmc["unif_sampler"] = random_sequence_rqmc
hmcdict_is_qmc["proposalname"] = 'HMC IS QMC'



if __name__ == '__main__':

    from smc_sampler_functions.functions_smc_main import repeat_sampling
    from smc_sampler_functions.functions_smc_is_main import repeat_sampling_is
    samplers_list_dict = [rwdict, hmcdict1, hmcdict2, maladict, hmcdict3, hmcdict4]
    #samplers_list_dict = [hmcdict1, hmcdict2, maladict]
    samplers_list_dict_is = [hmcdict_is_mc, hmcdict_is_qmc]

    # define the target distributions
    from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
    from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
    from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student
    from smc_sampler_functions.target_distributions import targetlogdens_logistic, targetgradlogdens_logistic, f_dict_logistic_regression
    from smc_sampler_functions.target_distributions import targetlogdens_probit, targetgradlogdens_probit, f_dict_logistic_regression
    from smc_sampler_functions.target_distributions import targetlogdens_normal_mix, targetgradlogdens_normal_mix

    from smc_sampler_functions.target_distributions import priorlogdens_mix, priorgradlogdens_mix, priorsampler_mix
    from smc_sampler_functions.target_distributions_logcox import priorlogdens_log_cox, priorgradlogdens_log_cox, priorsampler_log_cox
    from smc_sampler_functions.target_distributions_logcox import f_dict_log_cox, targetlogdens_log_cox, targetgradlogdens_log_cox

    parameters_logistic = f_dict_logistic_regression(dim)
    parameters.update(parameters_logistic)
    #parameters_log_cox = f_dict_log_cox(int(dim**0.5))
    #parameters.update(parameters_log_cox)
    #from smc_sampler_functions.target_distributions import targetlogdens_ring, targetgradlogdens_ring

    priordistribution = {'logdensity' : priorlogdens_log_cox, 'gradlogdensity' : priorgradlogdens_log_cox, 'priorsampler': priorsampler_log_cox}
    targetdistribution1 = {'logdensity' : targetlogdens_log_cox, 'gradlogdensity' : targetgradlogdens_log_cox, 'target_name': 'log_cox'}

    #priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
    #priordistribution = {'logdensity' : priorlogdens_mix, 'gradlogdensity' : priorgradlogdens_mix, 'priorsampler': priorsampler_mix}
    #targetdistribution1 = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}
    #targetdistribution1 = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}
    #targetdistribution1 = {'logdensity' : targetlogdens_logistic, 'gradlogdensity' : targetgradlogdens_logistic, 'target_name': 'logistic'}
    #targetdistribution2 = {'logdensity' : targetlogdens_probit, 'gradlogdensity' : targetgradlogdens_probit, 'target_name': 'probit'}
    #targetdistribution4 = {'logdensity' : targetlogdens_ring, 'gradlogdensity' : targetgradlogdens_ring, 'target_name': 'ring'}
    #targetdistribution1 = {'logdensity' : targetlogdens_normal_mix, 'gradlogdensity' : targetgradlogdens_normal_mix, 'target_name': 'normal_mix'}

    #target_dist_list = [targetdistribution1, targetdistribution2]
    target_dist_list = [targetdistribution1]
    #target_dist_list = [targetdistribution2, targetdistribution3
    for target_dist in target_dist_list: 
        temperedist = sequence_distributions(parameters, priordistribution, target_dist)
        #import yappi
        #yappi.start()
        # sample and compare the results
        #res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict1)
        #yappi.get_func_stats().print_all()
        #import ipdb; ipdb.set_trace()
        res_repeated_sampling, res_first_iteration = repeat_sampling(samplers_list_dict, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = target_dist['target_name'])
        #import ipdb; ipdb.set_trace()
        #res_repeated_sampling_is, res_first_iteration_is = repeat_sampling_is(samplers_list_dict_is, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = target_dist['target_name'])
        #from smc_sampler_functions.functions_smc_plotting import plot_repeated_simulations, plot_results_single_simulation
        #plot_repeated_simulations(res_repeated_sampling)
        #plot_results_single_simulation(res_first_iteration)
        #import ipdb; ipdb.set_trace()


