# run the simulation for the student model
from __future__ import division, print_function

import copy
import cPickle as pickle
import os
import sys

import numpy as np

from setup_simulations_server_student import prepare_samplers
from smc_sampler_functions.functions_smc_help import sequence_distributions



from smc_sampler_functions.functions_smc_main import single_simulation_over_samplers_dims
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
targetdistribution = {'logdensity' : targetlogdens_student, 'gradlogdensity' : targetgradlogdens_student, 'target_name': 'student'}

if __name__ == '__main__':

    if sys.argv[2] == 'test':
        dim_list = [10]#, 295]
        M = 1
        print('Run test loop student')
    else: 
        dim_list = [300, 400]#[10, 20, 50, 100, 200, 300, 400]
        M = 40 
        print('Run full loop student')

    for dim in dim_list:
        parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive_simple, __, __ = prepare_samplers(dim)

        #samplers_list_dict_adaptive = [hmcdict_ours_adaptive_simple, hmcdict_ft_adaptive, rwdict, maladict]
        samplers_list_dict_adaptive = [hmcdict_ours_adaptive_simple, maladict]
        hmcdict_ours_non_adaptive = copy.copy(hmcdict_ours_adaptive_simple)
        maladict_non_adaptive = copy.copy(maladict)

        temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)
        save_name=targetdistribution['target_name']

        samplers_list_dict_non_adaptive = []
        if sys.argv[1] == 'loop':
            for m_repetition in range(M):
                print('Repeated simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
                #import ipdb; ipdb.set_trace()
                res_dict_hmc = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
                res_dict_mala = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[1]['proposalname'], m_repetition, parameters['dim']), 'rb'))
                # preapare simulation
                #import ipdb; ipdb.set_trace()
                mala_mean_steps = int(np.ceil(len(res_dict_mala['temp_list'])/len(np.unique(res_dict_mala['temp_list']))))
                maladict_non_adaptive['proposalname'] = 'MALA_non_adaptive'
                maladict_non_adaptive['move_steps'] = mala_mean_steps
                maladict_non_adaptive['quantile_test'] = 0.00001
                
                hmc_mean_steps = int(np.ceil(len(res_dict_hmc['temp_list'])/len(np.unique(res_dict_hmc['temp_list']))))
                hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
                hmcdict_ours_non_adaptive['move_steps'] = hmc_mean_steps
                hmcdict_ours_non_adaptive['quantile_test'] = 0.00001

                samplers_list_dict_non_adaptive = [maladict_non_adaptive, hmcdict_ours_non_adaptive]
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)

        elif int(sys.argv[1])>=0: # one single iteration, for the server
            m_repetition = int(sys.argv[1])
            print('Single simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
            res_dict_hmc = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
            res_dict_mala = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[1]['proposalname'], m_repetition, parameters['dim']), 'rb'))
            # preapare simulation
            #import ipdb; ipdb.set_trace()
            mala_mean_steps = int(np.ceil(len(res_dict_mala['temp_list'])/len(np.unique(res_dict_mala['temp_list']))))
            maladict_non_adaptive['proposalname'] = 'MALA_non_adaptive'
            maladict_non_adaptive['move_steps'] = mala_mean_steps
            maladict_non_adaptive['quantile_test'] = 0.00001
            
            hmc_mean_steps = int(np.ceil(len(res_dict_hmc['temp_list'])/len(np.unique(res_dict_hmc['temp_list']))))
            hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
            hmcdict_ours_non_adaptive['move_steps'] = hmc_mean_steps
            hmcdict_ours_non_adaptive['quantile_test'] = 0.00001

            samplers_list_dict_non_adaptive = [maladict_non_adaptive, hmcdict_ours_non_adaptive]
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
        
        else: raise ValueError('require loop or other')

