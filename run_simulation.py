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
    dim = 30
N_particles = 2**10
T_time = 1000
move_steps_hmc = 1
move_steps_rw_mala = 10
ESStarget = 0.99
M_num_repetions = 1
epsilon = .1
epsilon_hmc = .1
verbose = True
#rs = np.random.seed(1)
targetmean = np.ones(dim)*2
targetvariance = (0.1*(np.diag(np.arange(dim))/float(dim) +0.7*np.ones((dim, dim))))
#targetvariance = np.eye(dim)*0.1
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
#from smc_sampler_functions.cython.cython_target_distributions import priorlogdens, priorgradlogdens
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens
from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student
from smc_sampler_functions.target_distributions import targetlogdens_logistic, targetgradlogdens_logistic, f_dict_logistic_regression
parameters_logistic = f_dict_logistic_regression(dim)
#import ipdb; ipdb.set_trace()
parameters.update(parameters_logistic)
#parameters['dim'] = parameters_logistic['X_all'].shape[1]
#from smc_sampler_functions.target_distributions import targetlogdens_student as targetlogdens_student_py
#from smc_sampler_functions.target_distributions import targetgradlogdens_student as targetgradlogdens_student_py

#import ipdb; ipdb.set_trace()
#particles_test = np.random.randn(N_particles, dim)
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens}
#targetdistribution = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}
targetdistribution = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}
#targetdistribution = {'logdensity' : targetlogdens_logistic, 'gradlogdensity' : targetgradlogdens_logistic, 'target_name': 'logistic'}

temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)

# prepare the kernels and specify parameters
from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_parallel
from smc_sampler_functions.functions_smc_main import smc_sampler
from smc_sampler_functions.standard_mh_sampler import parallel_mh_sampler

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



#print temperatures
#import yappi
#yappi.start()
# sample and compare the results
#res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict2)
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


if __name__ == '__main__':
    if False:
        mh_sampler = parallel_mh_sampler(temperedist, parameters, hmcdict2)
        #import ipdb; ipdb.set_trace()
        parameters['N_particles'] = 2**10
        res_dict_hmc = smc_sampler(temperedist,  parameters, hmcdict1)
        res_dict_rw = smc_sampler(temperedist,  parameters, rwdict)
        res_dict_mala = smc_sampler(temperedist,  parameters, maladict)
        
        from matplotlib import pyplot as plt
        plt.plot(mh_sampler['particles'][0,1,:]); plt.show()
        plt.plot(mh_sampler['particles'][0,1,:].cumsum()/np.arange(1, T_time+1)); plt.show()
        # mean
        print(mh_sampler['particles'][0,0,int(T_time/2):].mean())
        # acceptance rate
        print(np.mean(np.diff(mh_sampler['particles'][0,1,int(T_time/2):], 1) > 0))
        plt.hist(mh_sampler['particles'][0,1,int(T_time/2):]); plt.show()

        import ipdb; ipdb.set_trace()

    from smc_sampler_functions.functions_smc_main import repeat_sampling
    #samplers_list_dict = [rwdict, hmcdict2, maladict, rwdict]
    samplers_list_dict = [hmcdict2, rwdict, maladict]
    #samplers_list_dict = [hmcdict1, hmcdict2]
    res_repeated_sampling, res_first_iteration = repeat_sampling(samplers_list_dict, temperedist,  parameters, M_num_repetions=M_num_repetions, save_res=True, save_name = targetdistribution['target_name'])
    from smc_sampler_functions.functions_smc_plotting import plot_repeated_simulations, plot_results_single_simulation
    plot_repeated_simulations(res_repeated_sampling)
    plot_results_single_simulation(res_first_iteration)
    import ipdb; ipdb.set_trace()

