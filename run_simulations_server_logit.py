# run the simulation for the logit and probit model
from __future__ import division, print_function

import copy
import cPickle as pickle
import os
import sys

import numpy as np

from setup_simulations_server_logit_probit import prepare_samplers
from smc_sampler_functions.functions_smc_help import sequence_distributions

dim_list = [25, 31, 301]
M = 40 



from smc_sampler_functions.functions_smc_main import single_simulation_over_samplers_dims
from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens, priorsampler
from smc_sampler_functions.target_distributions import targetlogdens_logistic, targetgradlogdens_logistic, f_dict_logistic_regression
from smc_sampler_functions.target_distributions import targetlogdens_probit, targetgradlogdens_probit
priordistribution = {'logdensity' : priorlogdens, 'gradlogdensity' : priorgradlogdens, 'priorsampler': priorsampler}
targetdistribution = {'logdensity' : targetlogdens_logistic, 'gradlogdensity' : targetgradlogdens_logistic, 'target_name': 'logistic'}


if __name__ == '__main__':
    for dim in dim_list:
        parameters, maladict, rwdict, hmcdict_ft_adaptive, hmcdict_ours_adaptive = prepare_samplers(dim)

        parameters_logistic = f_dict_logistic_regression(dim)
        parameters.update(parameters_logistic)

        samplers_list_dict_adaptive = [hmcdict_ours_adaptive, hmcdict_ft_adaptive, rwdict, maladict]
        temperedist = sequence_distributions(parameters, priordistribution, targetdistribution)
        save_name=targetdistribution['target_name']


        if sys.argv[1] == 'loop':
            for m_repetition in range(M):
                print('Repeated simulation: now running repetition %s in dimension %s' %(m_repetition, dim))
                single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name)
                #import ipdb; ipdb.set_trace()
        elif int(sys.argv[1])>=0:
            m_repetition = int(sys.argv[1])
            print('Single simulation : now running repetition %s in dimension %s' %(m_repetition, dim))
            single_simulation_over_samplers_dims(m_repetition, samplers_list_dict_adaptive, temperedist, parameters, save_name=save_name)

        else: raise ValueError('require loop or other')