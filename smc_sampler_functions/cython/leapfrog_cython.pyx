# leapfrog cython
import numpy as np
cimport numpy as np
from cpython cimport bool
from cython.parallel import prange

DTYPE = np.ndarray

def leapfr_mom(np.ndarray x, np.ndarray cur_p, np.ndarray epsilon, gradf, float temperature, bool half=False):
    """ makes a leapfrog step for the momentum, vectorized """
    cdef float factor
    cdef np.ndarray rval
    if half:
        factor = 0.5
    else:
        factor = 1
    #x = np.atleast_2d(x)
    #cur_p = np.atleast_2d(cur_p)
    #assert cur_p.shape == x.shape
    rval = cur_p + factor*epsilon*gradf(x, temperature)
    return rval

def leapfr_pos(np.ndarray cur_x, np.ndarray p, np.ndarray epsilon, np.ndarray inv_mass):
    """ makes a leapfrog step for the position, vectorized """
    #cur_x = np.atleast_2d(cur_x)
    #p = np.atleast_2d(p)
    #assert cur_x.shape == p.shape
    cdef np.ndarray rval
    rval = cur_x + epsilon*p.dot(inv_mass)
    return rval

def leapfr_mom_pos(np.ndarray cur_x, np.ndarray cur_p, np.ndarray epsilon, gradf, float temperature, np.ndarray inv_mass):
    """ makes a leapfrog step for both the momentum and the position, vectorized """
    #cur_x = np.atleast_2d(cur_x)
    #cur_p = np.atleast_2d(cur_p)
    #assert cur_x.shape == cur_p.shape
    new_p = leapfr_mom(cur_x, cur_p, epsilon, gradf, temperature)
    new_x = leapfr_pos(cur_x, new_p, epsilon, inv_mass)
    #TODO add the bounce routine here
    return (new_x, new_p)


def leapfrog_transition_individual(L_value, np.ndarray x_L, np.ndarray p_L, np.ndarray epsilon_L, loggradient, float temperature, np.ndarray covariance_matrix):
    """
    individual leapfrog step
    """
    #import ipdb; ipdb.set_trace()
    for m_leapfrog_step in range(1, L_value):
        (x_L, p_L) = leapfr_mom_pos(x_L, p_L, epsilon_L, loggradient, temperature, covariance_matrix)
    p_L = leapfr_mom(x_L, p_L, epsilon_L, loggradient, temperature, half=True)
    return np.atleast_2d(x_L), np.atleast_2d(p_L)

# np.ndarray 

def loop_leapfrog(L_list, np.ndarray x, np.ndarray p, np.ndarray epsilon, loggradient, float temperature, np.ndarray covariance_matrix):
    cdef int L_range_max
    cdef np.ndarray x_L
    cdef np.ndarray p_L
    cdef np.ndarray epsilon_L
    cdef np.ndarray x_next
    cdef np.ndarray p_next
    L_range_max = len(L_list)
    for counter_index in range(L_range_max):#, nogil=True):
        L_value = int(L_list[counter_index])
        x_L, p_L = np.atleast_2d(x[counter_index, :, 1]), np.atleast_2d(p[counter_index, :, 1])
        epsilon_L = epsilon[counter_index]
        x[counter_index, :, 2], p[counter_index, :, 2] =  leapfrog_transition_individual(L_value, x_L, p_L, epsilon_L, loggradient, temperature, covariance_matrix)
        #x[counter_index, :, 2], p[counter_index, :, 2] = x_next, p_next
        #x[counter_index, :, 2] = res[0]
        #p[counter_index, :, 2] = res[1]
        #res_list.append(res)
    return x, p
    #return res_list

def leapfrog_transition_individual_parallel(index_L, L_dict, x_all, p_all, epsilon_all, loggradient, temperature, covariance_matrix):
    """
    individual leapfrog step
    """
    #import ipdb; ipdb.set_trace()
    iteration_L = int(L_dict[index_L]['iteration_L'])
    L_step = int(L_dict[index_L]['L_step'])
    x, p = x_all[iteration_L, :, 1], p_all[iteration_L, :, 1]
    epsilon = epsilon_all[iteration_L]
    for m_leapfrog_step in range(1, L_step):
        (x, p) = leapfr_mom_pos(x, p, epsilon, loggradient, temperature, covariance_matrix)
    p = leapfr_mom(x, p, epsilon, loggradient, temperature, half=True)
    return(np.atleast_2d(x), np.atleast_2d(p), iteration_L)

def loop_first_cython(np.ndarray particles, targetlogdens_student_new, 
                            _mean, 
                            inv_covar, 
                            covar, 
                            df,
                            dim):
    cdef int N_particles 
    N_particles = particles.shape[0]
    for n_particle in range(N_particles):
        targetlogdens_student_new(np.atleast_2d(particles[n_particle,:]), _mean, inv_covar, covar, df, dim)
