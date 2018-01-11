# IS version of the SMC sampler
# smc sampler
from __future__ import print_function
from __future__ import division

import numpy as np
import sys
sys.path.append("/home/alex/python_programming/help_functions")
sys.path.append("/home/alex/Dropbox/smc_hmc/python_smchmc")

import numpy as np
from smc_sampler_functions.functions_smc_help import logincrementalweights_is, reweight_is, ESS_target_dichotomic_search_is, tune_mcmc_parameters, ESS
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




def smc_sampler_is_qmc(temperedist, parameters, proposalkerneldict, verbose=False):
    """
    sqmc sampler, takes as input 
    the temperedist (class instance)
    parameters (dict)
    proposalkerneldict_temp (dict)
    """
    N_particles = parameters['N_particles']
    dim = parameters['dim']
    T_time = parameters['T_time']
    proposalkerneldict_temp = copy.copy(proposalkerneldict)
    if not parameters['autotempering']:
        temperatures = np.linspace(0,1,T_time)
        temperatures = np.hstack((temperatures, 1.)).flatten()
    
    proposalkernel_tune = proposalkerneldict_temp['proposalkernel_tune']
    proposalkernel_sample = proposalkerneldict_temp['proposalkernel_sample']
    move_steps = proposalkerneldict['move_steps']
    unif_sampler = proposalkerneldict['unif_sampler']
    assert callable(proposalkernel_tune)
    assert callable(proposalkernel_sample)
    assert isinstance(proposalkerneldict_temp, dict)
    
    # prepare for the results
    Z_list = []; mean_list = []; var_list = []
    ESS_list = []; acceptance_rate_list = []
    temp_list = []; perf_list = []

    print('Starting sqmc is sampler')
    time_start = time.time()
    
    if not parameters['autotempering']:
        temperatures = np.linspace(0,1,T_time)
        temperatures = np.hstack((temperatures, 1.)).flatten()
    
    # pre allocate data
    u_randomness = unif_sampler(dim*2, 0, N_particles)
    particles_initial = gaussian_vectorized(u_randomness[:, :dim])
    particles, perfkerneldict = proposalkernel_sample(particles_initial, u_randomness[:, dim:], proposalkerneldict_temp, temperedist, 0)

    weights_normalized = np.ones(N_particles)/N_particles
    weights = reweight_is(particles, particles_initial, temperedist, [0, 0], weights_normalized, perfkerneldict)

    max_weights = np.max(weights)
    Z_hat = max_weights+np.log(np.exp(weights-max_weights).sum())
    weights_normalized = np.exp(weights -(max_weights +np.log(np.exp(weights-max_weights).sum())))

    # store results
    Z_list.append(np.copy(Z_hat))
    ESS_list.append(ESS(weights_normalized))
    acceptance_rate_list.append(perfkerneldict['acceptance_rate'])
    means_weighted = np.average(particles, weights=weights_normalized, axis=0)
    variances_weighted = np.cov(particles, rowvar=False, aweights=weights_normalized)
    mean_list.append(means_weighted)
    var_list.append(variances_weighted)
    temp_list.append(0)
    # delete some attributes from the perfkerneldict, 
    del perfkerneldict['energy']; del perfkerneldict['squarejumpdist']
    perf_list.append(perfkerneldict)


    #loop sampler
    temp_curr = 0.
    temp_next = 0.
    counter_while = 0
    print('Now runing smc sampler with %s kernel' %proposalkerneldict_temp['proposalname'])
    time_start = time.time()

    while temp_curr <= 1.:

        # generate randomness
        u_randomness = unif_sampler(dim+1, 0, N_particles)
        # hilbert sampling
        if 'qmc' in unif_sampler.__name__:
            particles_resampled, weights_normalized, u_randomness_ordered = hilbert_sampling(particles, weights_normalized, u_randomness)
        else: 
            particles_resampled, weights_normalized, u_randomness_ordered = resampling_is(particles, weights_normalized, u_randomness)


        # propagate
        if parameters['adaptive_covariance'] and temp_curr != 0.:
            proposalkerneldict_temp['covariance_matrix'] = np.diag(np.diag(var_list[-1]))
        
        # tune the kernel
        if proposalkerneldict_temp['tune_kernel']:
            if verbose: 
                print("now tuning")
            # tune the parameters 
            proposalkerneldict_temp['L_steps'] = np.copy(proposalkerneldict['L_steps'])
            proposalkerneldict_temp['epsilon_sampled'] = np.random.random((N_particles,1))*proposalkerneldict_temp['epsilon_max']
            particles, perfkerneldict = proposalkernel_tune(particles_resampled, u_randomness_ordered[:, 1:], proposalkerneldict_temp, temperedist, temp_curr)
            perfkerneldict['temp'] = temp_curr
            del proposalkerneldict_temp['epsilon_sampled'] # deleted because also available in output perfkerneldict
            
            results_tuning = tune_mcmc_parameters(perfkerneldict, proposalkerneldict_temp)
            proposalkerneldict_temp['epsilon'] = results_tuning['epsilon_next']
            proposalkerneldict_temp['epsilon_max'] = results_tuning['epsilon_max']
            proposalkerneldict_temp['L_steps'] = results_tuning['L_next']
            #import ipdb; ipdb.set_trace()
        if verbose: 
            print("now sampling")

        particles, perfkerneldict = proposalkernel_sample(particles_resampled, u_randomness_ordered[:, 1:], proposalkerneldict_temp, temperedist, temp_curr)

        if not parameters['autotempering']:
            counter_while += 1
            temp_curr, temp_next = temperatures[counter_while-1], temperatures[counter_while]
        elif parameters['autotempering']:
            ESStarget = parameters['ESStarget']
            #partial_ess_target = partial(ESS_target_dichotomic_search, temperatureprevious=temp_curr, ESStarget=ESStarget, particles=particles_resampled, temperedist=temperedist, weights_normalized=weights_normalized)
            #temp_next = dichotomic_search.f_dichotomic_search(np.array([temp_curr,1.]), partial_ess_target, N_max_steps=100)
            #import ipdb; ipdb.set_trace()
            partial_ess_target = partial(ESS_target_dichotomic_search_is, 
                    temperatureprevious=temp_curr, 
                    ESStarget=ESStarget, particles=particles, 
                    particles_previous=particles_resampled, 
                    temperedist=temperedist, 
                    weights_normalized=weights_normalized, 
                    perfkerneldict=perfkerneldict)
            temp_next = dichotomic_search.f_dichotomic_search(np.array([temp_curr,1.]), partial_ess_target, N_max_steps=100)
            if partial_ess_target(temp_next)<-0.01:
                #import ipdb; ipdb.set_trace()
                temp_next = temp_curr
            print('temperature %s' %(temp_next), end='\r')
            assert temp_next <= 1.
            if temp_next > 1.:
                raise ValueError('temp greater than 1')
            if temp_curr == temp_next and temp_next < 1.:
                #import ipdb; ipdb.set_trace()
                print('not able to increase temperature')
        else:
            raise ValueError('tempering must be either auto or not')
        #import ipdb; ipdb.set_trace()
        

        weights = reweight_is(particles, particles_resampled, temperedist, [temp_curr, temp_next], weights_normalized, perfkerneldict)
        max_weights = np.max(weights)
        Z_hat = max_weights+np.log(np.exp(weights-max_weights).sum())
        weights_normalized = np.exp(weights -(max_weights +np.log(np.exp(weights-max_weights).sum())))

        # store results
        Z_list.append(np.copy(Z_hat))
        ESS_list.append(ESS(weights_normalized))
        acceptance_rate_list.append(perfkerneldict['acceptance_rate'])
        means_weighted = np.average(particles, weights=weights_normalized, axis=0)
        variances_weighted = np.cov(particles, rowvar=False, aweights=weights_normalized)
        mean_list.append(means_weighted)
        var_list.append(variances_weighted)
        temp_list.append(temp_curr)
        # delete some attributes from the perfkerneldict, 
        del perfkerneldict['energy']; del perfkerneldict['squarejumpdist']
        perf_list.append(perfkerneldict)



        #import ipdb; ipdb.set_trace()
        print('sampling for temperature %s, current ESS %s' % (temp_curr, ESS(weights_normalized)), end='\r')
        for move in range(move_steps):
            u_randomness = unif_sampler(dim+1, 0, N_particles)
            # hilbert sampling or multinomial sampling
            if 'qmc' in unif_sampler.__name__:
                particles_resampled, weights_normalized, u_randomness_ordered = hilbert_sampling(particles, weights_normalized, u_randomness)
            else: 
                particles_resampled, weights_normalized, u_randomness_ordered = resampling_is(particles, weights_normalized, u_randomness)        # propagate


            # propagate
            particles, perfkerneldict = proposalkernel_sample(particles_resampled, u_randomness_ordered[:, 1:], proposalkerneldict_temp, temperedist, temp_next)
            weights = reweight_is(particles, particles_resampled, temperedist, [temp_next, temp_next], weights_normalized, perfkerneldict)
            max_weights = np.max(weights)
            Z_hat = max_weights+np.log(np.exp(weights-max_weights).sum())
            weights_normalized = np.exp(weights -(max_weights +np.log(np.exp(weights-max_weights).sum())))

            # store results
            Z_list.append(np.copy(Z_hat))
            ESS_list.append(ESS(weights_normalized))
            acceptance_rate_list.append(perfkerneldict['acceptance_rate'])
            means_weighted = np.average(particles, weights=weights_normalized, axis=0)
            variances_weighted = np.cov(particles, rowvar=False, aweights=weights_normalized)
            mean_list.append(means_weighted)
            var_list.append(variances_weighted)
            temp_list.append(temp_curr)
            # delete some attributes from the perfkerneldict, 
            del perfkerneldict['energy']; del perfkerneldict['squarejumpdist']
            perf_list.append(perfkerneldict)
            
            
            
        
        if temp_curr == 1.:
            break
        temp_curr = np.copy(temp_next)
            
            
    particles_resampled = particles
    time_end = time.time()
    run_time = time_end-time_start
    print('Sampler ended at time %s after %s seconds \n' %(len(temp_list), run_time))
    res_dict = {
        'mean_list' : mean_list,
        'var_list' : var_list,
        'particles_resampled' : particles_resampled, 
        'weights_normalized' : weights_normalized, 
        'Z_list' : Z_list, 
        'ESS_list' : ESS_list, 
        'acceptance_rate_list' : acceptance_rate_list,
        'temp_list' : temp_list,
        'parameters' : parameters,
        'proposal_kernel': proposalkerneldict_temp,
        'run_time' : run_time,
        'perf_list' : perf_list,
        'target_name' : temperedist.target_name
        }
    #import ipdb; ipdb.set_trace()
    return res_dict