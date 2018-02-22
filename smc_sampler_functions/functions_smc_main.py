# smc sampler
from __future__ import print_function
import numpy as np
from functions_smc_help import logincrementalweights, reweight, resample, ESS, ESS_target_dichotomic_search, sequence_distributions, tune_mcmc_parameters, test_continue_sampling, tune_mcmc_parameters_fearnhead_taylor, ESS_target_dichotomic_search_simplified
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
import pandas as pd


def smc_sampler(temperedist, parameters, proposalkerneldict, verbose=False):
    """
    implements the smc sampler
    """
    #import ipdb; ipdb.set_trace()
    #assert isinstance(temperedist, sequence_distributions)
    N_particles = parameters['N_particles']
    dim = parameters['dim']
    T_time = proposalkerneldict['T_time']
    proposalkerneldict_temp = copy.copy(proposalkerneldict)
    if not proposalkerneldict['autotempering']:
        temperatures = np.linspace(0,1,T_time)
        temperatures = np.hstack((temperatures, 1.)).flatten()
    
    proposalkernel_tune = proposalkerneldict_temp['proposalkernel_tune']
    proposalkernel_sample = proposalkerneldict_temp['proposalkernel_sample']
    move_steps = proposalkerneldict['move_steps']
    assert callable(proposalkernel_tune)
    assert callable(proposalkernel_sample)
    assert isinstance(proposalkerneldict_temp, dict)
    
    # prepare for the results
    Z_list = []; mean_list = []; var_list = []
    ESS_list = []; acceptance_rate_list = []
    temp_list = []; perf_list = []; test_dict_list = []
    
    # intialize sampler
    u_randomness = np.random.random(size=(N_particles, dim))
    particles = temperedist.priorsampler(parameters, u_randomness)
    weights_normalized = np.ones(N_particles)/N_particles
    #Z_hat = 0.#np.log((2*np.pi)**(dim/2.))
    #Z_list.append(Z_hat)
    mean_list.append(np.average(particles, weights=weights_normalized, axis=0))

    # loop sampler
    temp_curr = 0.
    temp_next = 0.
    counter_while = 0
    print('Now runing smc sampler with %s kernel' %proposalkerneldict_temp['proposalname'])
    time_start = time.time()
    while temp_curr <= 1.:
        
        # resample
        particles_resampled, weights_normalized = resample(particles, weights_normalized)
        
        # propagate
        if proposalkerneldict['adaptive_covariance'] and temp_curr != 0.:
            proposalkerneldict_temp['covariance_matrix'] = np.diag(np.diag(var_list[-1]))
        
        # our tuning
        if proposalkerneldict_temp['tune_kernel'] == True:
            if verbose: 
                print("now tuning")
            # tune the parameters 
            proposalkerneldict_temp['L_steps'] = np.copy(proposalkerneldict['L_steps'])
            proposalkerneldict_temp['epsilon_sampled'] = np.random.random((N_particles,1))*proposalkerneldict_temp['epsilon_max']
            particles, perfkerneldict = proposalkernel_tune(particles_resampled, proposalkerneldict_temp, temperedist, temp_curr)
            perfkerneldict['temp'] = temp_curr
            del proposalkerneldict_temp['epsilon_sampled'] # deleted because also available in output perfkerneldict
            
            results_tuning = tune_mcmc_parameters(perfkerneldict, proposalkerneldict_temp)
            proposalkerneldict_temp['epsilon'] = results_tuning['epsilon_next']
            proposalkerneldict_temp['epsilon_max'] = results_tuning['epsilon_max']
            proposalkerneldict_temp['L_steps'] = results_tuning['L_next']
            #import ipdb; ipdb.set_trace()
        
        # tuning as in the fearnhead and taylor paper
        elif proposalkerneldict_temp['tune_kernel'] == 'fearnhead_taylor':
            if temp_curr == 0.:
                proposalkerneldict_temp['epsilon'] = np.random.random((N_particles,1))*proposalkerneldict_temp['epsilon_max']
                if proposalkerneldict['L_steps']>1:
                    proposalkerneldict_temp['L_steps'] = np.random.randint(1, proposalkerneldict['L_steps'], N_particles)
                else: 
                    proposalkerneldict_temp['L_steps'] = 1.
            else: 
                results_tuning = tune_mcmc_parameters_fearnhead_taylor(perfkerneldict, proposalkerneldict_temp)
                # perturb the kernels
                #import ipdb; ipdb.set_trace()
                eps_perturbed = results_tuning['epsilon_next']+np.random.normal(loc=0.0, scale=0.015, size=results_tuning['epsilon_next'].shape)
                eps_perturbed = np.clip(eps_perturbed, 0, 100)
                proposalkerneldict_temp['epsilon'] = eps_perturbed
                proposalkerneldict_temp['epsilon_max'] = results_tuning['epsilon_max']

                L_perturbed = results_tuning['L_next']+np.random.random_integers(low=-1,high=1, size=results_tuning['L_next'].shape)
                L_perturbed = np.clip(L_perturbed, 1, 200)
                proposalkerneldict_temp['L_steps'] = L_perturbed

        if verbose: 
            print("now sampling")
        summary_particles_list = []
        summary_particles_list.append(particles_resampled.sum(axis=1))
        particles, perfkerneldict = proposalkernel_sample(particles_resampled, proposalkerneldict_temp, temperedist, temp_curr)
        summary_particles_list.append(particles.sum(axis=1))
        
        for move in range(move_steps):
            #import ipdb; ipdb.set_trace()
            test_dict = test_continue_sampling(particles, summary_particles_list, temp_curr, temperedist, proposalkerneldict['quantile_test'])
            test_dict['temp'] = temp_curr
            test_dict_list.append(test_dict)
            if not test_dict['test_decision']:
                break
            else: 
                particles, __ = proposalkernel_sample(particles, proposalkerneldict_temp, temperedist, temp_curr)
                summary_particles_list.append(particles.sum(axis=1))
                temp_list.append(temp_curr)

            #import ipdb; ipdb.set_trace()
            #np.random.shuffle(proposalkerneldict_temp['epsilon'])
            #np.random.shuffle(proposalkerneldict_temp['L_steps'])
        

        # choose weights adaptively
        if not proposalkerneldict['autotempering']:
            counter_while += 1
            temp_curr, temp_next = temperatures[counter_while-1], temperatures[counter_while]
        elif proposalkerneldict['autotempering']:
            ESStarget = proposalkerneldict['ESStarget']

            # old verison
            #partial_ess_target = partial(ESS_target_dichotomic_search, temperatureprevious=temp_curr, ESStarget=ESStarget, particles=particles, temperedist=temperedist, weights_normalized=weights_normalized)
            # new version 
            precalc_dict = temperedist.precalc_logdensity(particles)
            partial_ess_target = partial(ESS_target_dichotomic_search_simplified, temperatureprevious=temp_curr, ESStarget=ESStarget, precalc_dict=precalc_dict)
            #import ipdb; ipdb.set_trace()
            temp_next = dichotomic_search.f_dichotomic_search(np.array([temp_curr,1.]), partial_ess_target, N_max_steps=15)
            print('temperature %s' %(temp_next), end='\r')
            assert temp_next <= 1.
            if temp_next > 1.:
                raise ValueError('temp greater than 1')
            if temp_curr == temp_next and temp_next < 1.:
                #import ipdb; ipdb.set_trace()
                print('not able to increase temperature')
        else:
            raise ValueError('tempering must be either auto or not')
        
        # reweight
        weights = reweight(particles, temperedist, [temp_curr, temp_next], weights_normalized)
        #if np.isinf(np.exp(weights)).any():
        #    import ipdb; ipdb.set_trace()
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
        #del perfkerneldict['energy']; del perfkerneldict['squarejumpdist']
        perf_list.append(perfkerneldict)

        # break routine
        if temp_curr == 1.:
            break
        temp_curr = np.copy(temp_next)

    # resample and remove in the end
    summary_particles_list = []
    summary_particles_list.append(particles.sum(axis=1)+(particles**2).sum(axis=1))
    particles, perfkerneldict = proposalkernel_sample(particles, proposalkerneldict_temp, temperedist, temp_curr)
    summary_particles_list.append(particles.sum(axis=1)+(particles**2).sum(axis=1))
    temp_list.append(temp_curr)

    for move in range(move_steps):
        test_dict = test_continue_sampling(particles, summary_particles_list, temp_curr, temperedist, proposalkerneldict['quantile_test'])
        test_dict['temp'] = temp_curr
        test_dict_list.append(test_dict)
        if not test_dict['test_decision']:
            break
        else: 
            particles, __ = proposalkernel_sample(particles, proposalkerneldict_temp, temperedist, temp_curr)
            summary_particles_list.append(particles.sum(axis=1)+(particles**2).sum(axis=1))
            #temp_list.append(temp_curr)
        #import ipdb; ipdb.set_trace()
        #np.random.shuffle(proposalkerneldict_temp['epsilon'])
        #np.random.shuffle(proposalkerneldict_temp['L_steps'])
    #pdb.set_trace()
    particles_resampled, weights_normalized = resample(particles, weights_normalized)
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
        'target_name' : temperedist.target_name,
        'L_mean' : np.array([iteration['L'] for iteration in perf_list]).mean(),
        'epsilon_mean' : np.array([iteration['epsilon'] for iteration in perf_list])[-1,:,:].mean(),
        'test_dict_list' : test_dict_list
        }
    #import ipdb; ipdb.set_trace()
    return res_dict




def repeat_sampling(samplers_list_dict, temperedist, parameters, M_num_repetions=50, save_res=True, save_res_intermediate=False, save_name=''):
    # function that repeats the sampling
    len_list = len(samplers_list_dict)
    dim = parameters['dim']
    N_particles = parameters['N_particles']
    norm_constant_list = np.zeros((len_list, M_num_repetions))
    mean_array = np.zeros((len_list, M_num_repetions, dim))
    var_array = np.zeros((len_list, M_num_repetions, dim, dim))
    ESJD_array = np.zeros((len_list, M_num_repetions))
    temp_steps_array = np.zeros((len_list, M_num_repetions))
    temp_steps_array_single = np.zeros((len_list, M_num_repetions))
    particles_array = np.zeros((N_particles, dim, len_list, M_num_repetions))
    names_samplers = [sampler['proposalname'] for sampler in samplers_list_dict]
    runtime_list = np.zeros((len_list, M_num_repetions))
    L_mean_array = np.zeros((len_list, M_num_repetions))
    epsilon_mean_array = np.zeros((len_list, M_num_repetions))
    ESS_dict_all = {name : [] for name in names_samplers}

    root_folder = os.getcwd()
    if save_res:
        now = datetime.datetime.now().isoformat()
        os.mkdir('results_simulation_%s_%s'%(temperedist.target_name, now))
        os.chdir('results_simulation_%s_%s'%(temperedist.target_name, now))
    # run the samplers
    res_first_iteration = []
    for m_repetition in range(M_num_repetions):
        print("repetition %s of %s" %(m_repetition, M_num_repetions), end='\n')
        for k, sampler_dict in enumerate(samplers_list_dict):
            res_dict = smc_sampler(temperedist,  parameters, sampler_dict)
            # save the first instance
            if m_repetition == 0:
                res_first_iteration.append(res_dict)
            norm_constant_list[k, m_repetition] = np.sum(res_dict['Z_list'])
            mean_array[k, m_repetition,:] = res_dict['mean_list'][-1]
            ESJD_array[k, m_repetition] = res_dict['perf_list'][-1]['squarejumpdist_realized'].mean()
            L_mean_array[k, m_repetition] = res_dict['L_mean']
            epsilon_mean_array[k, m_repetition] = res_dict['epsilon_mean']
            temp_steps_array[k, m_repetition] = len(res_dict['temp_list'])
            temp_steps_array_single[k, m_repetition] = len(np.unique(res_dict['temp_list'])) # single steps
            inter_frame = pd.DataFrame({'ESS' : res_dict['ESS_list'], 'temp' : np.unique(res_dict['temp_list'])})
            ESS_dict_all[sampler_dict['proposalname']].append(inter_frame)
            #import ipdb; ipdb.set_trace()
            var_array[k, m_repetition,:,:] = res_dict['var_list'][-1]
            runtime_list[k, m_repetition] = res_dict['run_time']
            particles_array[:,:,k, m_repetition] = res_dict['particles_resampled']
            if save_res_intermediate:
                pickle.dump(res_dict, open('%ssampler_%s_rep_%s_dim_%s.p'%(save_name, names_samplers[k], m_repetition, parameters['dim']), 'wb'))
            all_dict = {'parameters': parameters, 
                        'norm_const' : norm_constant_list, 
                        'mean_array' : mean_array, 
                        'var_array' :  var_array, 
                        'names_samplers' : names_samplers,
                        'M_num_repetions' : M_num_repetions,
                        'target_name' : temperedist.target_name, 
                        'particles_array' : particles_array, 
                        'runtime_list' : runtime_list, 
                        'ESJD_list': ESJD_array, 
                        'temp_steps' : temp_steps_array, 
                        'temp_steps_single' : temp_steps_array_single,
                        'L_mean' : L_mean_array, 
                        'epsilon_mean' : epsilon_mean_array,
                        'ESS_dict_all' : ESS_dict_all
                        }
            if save_res:
                pickle.dump(all_dict, open('%s_%s_all_dict_sampler_dim_%s.p' %(temperedist.target_name, save_name, parameters['dim']), 'wb'))
    os.chdir(root_folder)
    #root_folder = os.getcwd()
    #import ipdb; ipdb.set_trace()
    return(all_dict, res_first_iteration)


if __name__ == '__main__':
    print(resampling.multinomial_resample([0.5, 0.5]))
    now = datetime.datetime.now().isoformat()
    os.mkdir('results_simulation_%s'%(now))

