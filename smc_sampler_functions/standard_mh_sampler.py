# functions smc for standard metropolis hasting

from __future__ import print_function
import numpy as np
from functions_smc_help import logincrementalweights, reweight, resample, ESS, ESS_target_dichotomic_search, sequence_distributions, tune_mcmc_parameters
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


def parallel_mh_sampler(temperedist, parameters, proposalkerneldict):
    """
    implements the smc sampler
    """
    #import ipdb; ipdb.set_trace()
    #assert isinstance(temperedist, sequence_distributions)
    N_particles = parameters['N_particles']
    dim = parameters['dim']
    T_time = parameters['T_time']
    proposalkerneldict_temp = copy.copy(proposalkerneldict)
    
    proposalkernel_sample = proposalkerneldict_temp['proposalkernel_sample']

    assert callable(proposalkernel_sample)
    assert isinstance(proposalkerneldict_temp, dict)
    perf_list = []
    acceptance_counter = 0

    
    # intialize sampler
    particles = np.random.normal(size=(N_particles, dim, T_time))

    print('Now runing mh sampler with %s kernel' %proposalkerneldict_temp['proposalname'])
    time_start = time.time()
    for k in range(1,T_time):
        
        # resample
        
        print("now sampling iteration %s" %(k), end='\r')
        
        particles[:,:,k], perfkerneldict = proposalkernel_sample(particles[:,:,k-1], proposalkerneldict_temp, temperedist, 1.)
        perf_list.append(perfkerneldict)
        acceptance_counter += perfkerneldict['acceptance_rate']
    #import pdb; pdb.set_trace()

    ESJD = (np.linalg.norm(np.diff(particles, axis=2), axis=1)**2).mean()
    time_end = time.time()
    run_time = time_end-time_start
    print('Sampler ended at time %s after %s seconds \n' %(T_time, run_time))
    res_dict = {
        'particles' : particles, 
        'perf_list' : perf_list,
        'acceptance_rate' : acceptance_counter/T_time,
        'ESJD' : ESJD
        }
    return res_dict
