# run the simulation for the student model
from __future__ import division, print_function

import copy
import cPickle as pickle
import os
import sys

import numpy as np

from setup_simulations_server_student import prepare_samplers
from smc_sampler_functions.functions_smc_help import sequence_distributions

dim_list = [300, 500]#[10, 20, 50, 100, 200]#, 300, 500]
M = 40



from smc_sampler_functions.functions_smc_main import single_simulation_over_samplers_dims
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
targetdistribution = {'logdensity' : targetlogdens_normal, 'gradlogdensity' : targetgradlogdens_normal, 'target_name': 'normal'}

if __name__ == '__main__':
    for dim in dim_list:
        parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive_simple, hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive = prepare_samplers(dim)

        samplers_list_dict_adaptive = [hmcdict_ours_adaptive_simple, hmcdict_ft_adaptive, rwdict, maladict]
        temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)
        save_name=targetdistribution['target_name']

        samplers_list_dict_non_adaptive = []
        if sys.argv[1] == 'loop':
            for m_repetition in range(M):
                print('Repeated simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
                #import ipdb; ipdb.set_trace()
                res_dict = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
                # preapare simulation
                T_time_non_adaptive = len(res_dict['temp_list'])
                hmcdict_ft_non_adaptive['T_time'] = T_time_non_adaptive
                hmcdict_ft_non_adaptive['proposalname'] = 'HMC_L_random_ft_non_adaptive'
                hmcdict_ours_non_adaptive['T_time'] = T_time_non_adaptive
                hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
                samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)

        elif int(sys.argv[1])>=0: # one single iteration, for the server
            m_repetition = int(sys.argv[1])
            print('Single simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
            #import ipdb; ipdb.set_trace()
            res_dict = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
            # preapare simulation
            T_time_non_adaptive = len(res_dict['temp_list'])
            hmcdict_ft_non_adaptive['T_time'] = T_time_non_adaptive
            hmcdict_ft_non_adaptive['proposalname'] = 'HMC_L_random_ft_non_adaptive'
            hmcdict_ours_non_adaptive['T_time'] = T_time_non_adaptive
            hmcdict_ours_non_adaptive['proposalname'] = 'HMC_L_random_ours_non_adaptive'
            samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
        
        else: raise ValueError('require loop or other')

