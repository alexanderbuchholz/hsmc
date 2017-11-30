# test numba

from smc_sampler_functions.target_distributions import priorlogdens, priorgradlogdens
from smc_sampler_functions.target_distributions import targetlogdens_normal, targetgradlogdens_normal
from smc_sampler_functions.target_distributions import targetlogdens_student, targetgradlogdens_student

from smc_sampler_functions.cython.leapfrog_cython import loop_first_cython
from smc_sampler_functions.cython.cython_target_distributions import targetlogdens_student_new
loop_first_cython
from numba import jit
import numpy as np

from run_simulation import parameters

particles = np.random.randn( parameters['N_particles'], parameters['dim'])
N_particles = parameters['N_particles']
dim = parameters['dim']
#import ipdb; ipdb.set_trace()
targetlogdens_student(particles, parameters)

def loop_first(particles, parameters, targetlogdens_student):
    for n_particle in range(parameters['N_particles']):
        targetlogdens_student(np.atleast_2d(particles[n_particle,:]), parameters)

#@jit(nopython=True)
def loop_first_numba(particles, parameters, targetlogdens_student):
    for n_particle in range(parameters['N_particles']):
        targetlogdens_student(np.atleast_2d(particles[n_particle,:]), parameters)

_mean, inv_covar, covar, df= parameters['targetmean'], parameters['targetvariance_inv'], parameters['targetvariance'], parameters['df']
loop_first(particles, parameters, targetlogdens_student)
loop_first_cython(particles, targetlogdens_student_new, _mean, inv_covar, covar, df, dim)
targetlogdens_student(particles, parameters)

#ipdb.set_trace()