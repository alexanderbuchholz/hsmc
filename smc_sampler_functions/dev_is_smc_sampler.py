# dev sqmc sampler
# Notebook for smc sampler 
from __future__ import print_function
from __future__ import division

import numpy as np
import sys
sys.path.append("/home/alex/python_programming/help_functions")
sys.path.append("/home/alex/Dropbox/smc_hmc/python_smchmc")
from smc_sampler_functions.functions_smc_help import sequence_distributions

import numpy as np
from smc_sampler_functions.functions_smc_help import logincrementalweights, reweight, resample, ESS, ESS_target_dichotomic_search, sequence_distributions, tune_mcmc_parameters
from smc_sampler_functions.functions_smc_help import logincrementalweights_is, reweight_is, ESS_target_dichotomic_search_is
from functools import partial

import sys
sys.path.append("../help")
from help import resampling
from help import dichotomic_search
import inspect
import time
import copy
import pickle
import datetime
import os

from help.f_rand_seq_gen import random_sequence_qmc, random_sequence_rqmc, random_sequence_mc
from help.gaussian_densities_etc import gaussian_vectorized
from smc_sampler_functions.functions_smc_help import hilbert_sampling, resampling_is


# define the parameters
dim = 5
N_particles = 2**10
T_time = 10
move_steps = 2
ESStarget = 0.9
#rs = np.random.seed(1)
targetmean = np.ones(dim)*-3
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
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student

priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
#targetdistribution = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}
targetdistribution = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}

temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)

# prepare the kernels and specify parameters
from smc_sampler_functions.proposal_kernels import proposalmala, proposalrw, proposalhmc, proposalhmc_is
from smc_sampler_functions.functions_smc_main import smc_sampler


epsilon_hmc = 0.03
verbose = False
move_steps_hmc = 2
hmcdict = {'proposalkernel_tune': proposalhmc,
                      'proposalkernel_sample': proposalhmc,
                      'proposalname' : 'HMC',
                      'target_probability' : 0.9,
                      'covariance_matrix' : np.eye(dim), 
                      'L_steps' : 100,
                      'epsilon' : np.array([epsilon_hmc]),
                      'epsilon_max' : np.array([epsilon_hmc]),
                      'accept_reject' : True,
                      'tune_kernel': False,
                      'sample_eps_L' : True,
                      'parallelize' : False,
                      'verbose' : verbose,
                      'move_steps': move_steps_hmc,
                      'mean_L' : False
                      }

hmcdict_is_mc = {'proposalkernel_tune': proposalhmc_is,
                      'proposalkernel_sample': proposalhmc_is,
                      'proposalname' : 'HMC IS',
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
                      'mean_L' : True,
                      'unif_sampler' : random_sequence_mc,
                      'trajectory_selector_energy' : True
                      }
hmcdict_is_qmc = copy.copy(hmcdict_is_mc)
hmcdict_is_qmc["unif_sampler"] = random_sequence_qmc

from functions_smc_is_main import smc_sampler_is_qmc
from matplotlib import pyplot as plt

#random_sequence_qmc, random_sequence_rqmc, random_sequence_mc
#res_qmc = sqmc_sampler(temperedist, parameters, hmcdict_is)

if True: 
    M_rep = 1
    summary_particles_mean_mc = []
    summary_particles_var_mc = []
    summary_particles_norm_mc = []
    summary_particles_mean_qmc = []
    summary_particles_var_qmc = []
    summary_particles_norm_qmc = []
    for m_iter in range(M_rep):
        print('repetition %s' %(m_iter))
        #import ipdb; ipdb.set_trace()
        res_mc = smc_sampler_is_qmc(temperedist, parameters, hmcdict_is_mc)
        import ipdb; ipdb.set_trace()
        res_qmc = smc_sampler_is_qmc(temperedist, parameters, hmcdict_is_qmc)
        import ipdb; ipdb.set_trace()
        #particles_array_mcmc, __ = smc_sampler(temperedist, parameters, hmcdict)
        summary_particles_mean_qmc.append(res_qmc['particles'].mean(axis=0))
        summary_particles_mean_mc.append(res_mc['particles'].mean(axis=0))

        summary_particles_var_qmc.append(res_qmc['particles'].var(axis=0))
        summary_particles_var_mc.append(res_mc['particles'].var(axis=0)) 

        summary_particles_norm_qmc.append(res_qmc['norm_constant_array'].prod())
        summary_particles_norm_mc.append(res_mc['norm_constant_array'].prod())

    
    plt.scatter(x=res_qmc['particles'][:,0], y=res_qmc['particles'][:,1]); plt.show()
    plt.plot(np.array(res_qmc['var_array'])[:,0]); plt.title('var'); plt.show()
    plt.plot(np.array(res_qmc['mean_array'])[:,0]); plt.title('mean');  plt.show()
    plt.plot(np.array(res_qmc['norm_constant_array'])); plt.title('norm constant'); plt.show()

    plt.plot((np.array(summary_particles_mean_qmc).var(axis=0)), label="qmc")
    plt.plot(np.array(summary_particles_mean_mc).var(axis=0), label='mc'); plt.yscale('log')
    plt.legend()
    plt.show()

    plt.plot((np.array(summary_particles_norm_qmc).var(axis=0)), label="qmc")
    plt.plot(np.array(summary_particles_norm_mc).var(axis=0), label='mc'); plt.yscale('log')
    plt.legend()
    plt.show()
import ipdb; ipdb.set_trace()