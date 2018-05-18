# run the simulation for the student model
from __future__ import division, print_function

import copy
import cPickle as pickle
import os
import sys

import numpy as np

from setup_simulations_server_log_cox import prepare_samplers
from smc_sampler_functions.functions_smc_help import sequence_distributions





from smc_sampler_functions.functions_smc_main import single_simulation_over_samplers_dims
from smc_sampler_functions.target_distributions_logcox import priorlogdens_log_cox, priorgradlogdens_log_cox, priorsampler_log_cox
from smc_sampler_functions.target_distributions_logcox import f_dict_log_cox, targetlogdens_log_cox, targetgradlogdens_log_cox
priordistribution = {'logdensity' : priorlogdens_log_cox, 'gradlogdensity' : priorgradlogdens_log_cox, 'priorsampler': priorsampler_log_cox}
targetdistribution = {'logdensity' : targetlogdens_log_cox, 'gradlogdensity' : targetgradlogdens_log_cox, 'target_name': 'log_cox'}

if __name__ == '__main__':
    if sys.argv[2] == 'test':
        dim_list = [10**2]
        M = 1
        print('Run test loop log cox')
    else: 
        dim_list = [45**2]#[25**2, 30**2, 40**2]#[10**2, 20**2, 30**2, 40**2]#, 64**2]
        M = 40
        print('Run full loop log cox')

    for dim in dim_list:
        parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive, hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive = prepare_samplers(dim)
        parameters_log_cox = f_dict_log_cox(int(dim**0.5))
        parameters.update(parameters_log_cox)

        #samplers_list_dict_adaptive = [hmcdict_ours_adaptive, hmcdict_ft_adaptive, rwdict, maladict]
        samplers_list_dict_adaptive = [hmcdict_ours_adaptive, hmcdict_ft_adaptive, maladict]
        temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)
        save_name=targetdistribution['target_name']

        samplers_list_dict_non_adaptive = []
        if sys.argv[1] == 'loop': # loop otherwise
            #import ipdb; ipdb.set_trace()
            for m_repetition in range(M):
                print('Repeated simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
                #import ipdb; ipdb.set_trace()
                #res_dict = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
                # preapare simulation
                #adjusted_steps = int(np.ceil(float(len(res_dict['temp_list'] ))/len(np.unique(res_dict['temp_list']))))

                #hmcdict_ours_non_adaptive['move_steps'] = adjusted_steps
                #hmcdict_ft_non_adaptive['move_steps'] = adjusted_steps

                #samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
                #single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)

        elif int(sys.argv[1])>=0: # one single iteration, for the server
            m_repetition = int(sys.argv[1])
            print('Single simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
            #import ipdb; ipdb.set_trace()
            res_dict = pickle.load(open('results_simulation_%s/'%(temperedist.target_name)+'%ssampler_%s_rep_%s_dim_%s.p'%(save_name, samplers_list_dict_adaptive[0]['proposalname'], m_repetition, parameters['dim']), 'rb'))
            # preapare simulation
            #import ipdb; ipdb.set_trace()
            #adjusted_steps = int(np.ceil(float(len(res_dict['temp_list'] ))/len(np.unique(res_dict['temp_list']))))

            #hmcdict_ours_non_adaptive['move_steps'] = adjusted_steps
            #hmcdict_ft_non_adaptive['move_steps'] = adjusted_steps

            
            #samplers_list_dict_non_adaptive = [hmcdict_ft_non_adaptive, hmcdict_ours_non_adaptive]
            #single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_non_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
        
        else: raise ValueError('require loop or other')

