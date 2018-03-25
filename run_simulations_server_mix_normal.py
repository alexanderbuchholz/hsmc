# run the simulation for the student model
from __future__ import division, print_function

import copy
import cPickle as pickle
import os
import sys

import numpy as np

from setup_simulations_server_mix_normal import prepare_samplers
from smc_sampler_functions.functions_smc_help import sequence_distributions

dim_list = [5, 10, 15, 20, 30]
M = 40



from smc_sampler_functions.functions_smc_main import single_simulation_over_samplers_dims
from smc_sampler_functions.target_distributions import targetlogdens_normal_mix, targetgradlogdens_normal_mix
from smc_sampler_functions.target_distributions import priorlogdens_mix, priorgradlogdens_mix, priorsampler_mix

priordistribution = {'logdensity' : priorlogdens_mix, 'gradlogdensity' : priorgradlogdens_mix, 'priorsampler': priorsampler_mix}
targetdistribution = {'logdensity' : targetlogdens_normal_mix, 'gradlogdensity' : targetgradlogdens_normal_mix, 'target_name': 'normal_mix'}

if __name__ == '__main__':
    for dim in dim_list:
        parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive = prepare_samplers(dim)

        samplers_list_dict_adaptive = [hmcdict_ours_adaptive, hmcdict_ft_adaptive, rwdict, maladict]
        temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)
        save_name=targetdistribution['target_name']
        if sys.argv[1] == 'loop':
            for m_repetition in range(M):
                print('Repeated simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)
        elif int(sys.argv[1])>=0: # one single iteration, for the server
            m_repetition = int(sys.argv[1])
            print('Single simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name, seed=m_repetition)


        else: raise ValueError('require loop or other')
