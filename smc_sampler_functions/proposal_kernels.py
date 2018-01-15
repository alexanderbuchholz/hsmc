# proposal kernels
import warnings
from functools import partial
from help.functions_parallelize import parallelize_partial_over_chunks
import numpy as np
from scipy.stats import multivariate_normal
from multiprocessing.dummy import Pool as ThreadPool 
pool = ThreadPool(8) 

def proposalrw(particles, parametersmcmc, temperedist, temperature):
    """
    random walk proposal
    """
    assert isinstance(parametersmcmc, dict)
    assert temperature <= 1.
    assert temperature >= 0.
    N_particles, dim = particles.shape
    covariance_matrix = parametersmcmc['covariance_matrix']
    #import ipdb; ipdb.set_trace()
    l_matrix = np.linalg.cholesky(covariance_matrix)
    l_matrix_inv = np.linalg.inv(l_matrix)

    size_rw = particles.shape

    if 'epsilon_sampled' in parametersmcmc.keys():
        epsilon = parametersmcmc['epsilon_sampled']
    else:
        epsilon = parametersmcmc['epsilon']
    if epsilon.shape[0] == 1:
        epsilon = np.ones((N_particles,1))*epsilon


    noise = np.random.normal(size=size_rw).dot(l_matrix_inv)*epsilon
    #pdb.set_trace()
    particles_proposed = particles+noise
    
    weights_numerator = temperedist.logdensity(particles_proposed, temperature=temperature)

    
    weights_denominator = temperedist.logdensity(particles, temperature=temperature)

    mh_ratio = weights_numerator-weights_denominator
    u_accept = np.random.random(particles.shape[0])
    
    accept_reject_selector = np.log(u_accept) < mh_ratio
    particles_next = np.zeros(size_rw)
    
    particles_next[accept_reject_selector,:] = particles_proposed[accept_reject_selector,:]
    particles_next[~accept_reject_selector,:] = particles[~accept_reject_selector,:]

    squarejumpdist = np.linalg.norm(particles-particles_proposed, axis=1)**2
    jumping_distance_realized = np.linalg.norm(particles-particles_next, axis=1)**2
    performance_kernel_dict = {'energy': mh_ratio, 
                                'squarejumpdist':squarejumpdist,
                                'acceptance_rate':accept_reject_selector.mean(),
                                'squarejumpdist_realized':jumping_distance_realized,
                                'epsilon':epsilon,
                                'L':1}
    if parametersmcmc['verbose']:
        print('acceptance rate: %s, esjd: %s, epsilon mean: %s, L mean: %s' %(accept_reject_selector.mean(), jumping_distance_realized.mean(), np.mean(epsilon), np.mean(1)))
    return particles_next, performance_kernel_dict

    
def proposalmala(particles, parametersmcmc, temperedist, temperature):
    """
    mala proposal, takes the gradient information into account
    """
    assert isinstance(parametersmcmc, dict)
    assert temperature <= 1.
    assert temperature >= 0.
    N_particles, dim = particles.shape
    

    size_drift = particles.shape
    dim = size_drift[1]
    if 'epsilon_sampled' in parametersmcmc.keys():
        epsilon = parametersmcmc['epsilon_sampled']
    else:
        epsilon = parametersmcmc['epsilon']
    if epsilon.shape[0] == 1:
        epsilon = np.ones((N_particles,1))*epsilon

    covariance_matrix = parametersmcmc['covariance_matrix']
    l_matrix = np.linalg.cholesky(covariance_matrix)
    l_matrix_inv = np.linalg.inv(l_matrix)


    drift = np.random.normal(size=size_drift).dot(l_matrix)*epsilon
    mu_proposed = particles+0.5*epsilon**2*temperedist.gradlogdensity(particles, temperature=temperature).dot(covariance_matrix)
    particles_proposed = mu_proposed+drift
    mu_reversed = particles_proposed+0.5*epsilon**2*temperedist.gradlogdensity(particles_proposed, temperature=temperature).dot(covariance_matrix)

    weights_numerator1 = temperedist.logdensity(particles_proposed, temperature=temperature)
    weights_numerator2 = multivariate_normal.logpdf((mu_reversed-particles).dot(l_matrix_inv)/epsilon, cov=np.eye(dim))
    weights_numerator = weights_numerator1+weights_numerator2
    
    weights_denominator1 = temperedist.logdensity(particles, temperature=temperature)
    weights_denominator2 = multivariate_normal.logpdf((mu_proposed-particles_proposed).dot(l_matrix_inv)/epsilon, cov=np.eye(dim))
    weights_denominator = weights_denominator1+weights_denominator2
    
    mh_ratio = weights_numerator-weights_denominator
    u_accept = np.random.random(particles.shape[0])
    
    accept_reject_selector = np.log(u_accept) < mh_ratio
    particles_next = np.zeros(size_drift)
    
    particles_next[accept_reject_selector,:] = particles_proposed[accept_reject_selector,:]
    particles_next[~accept_reject_selector,:] = particles[~accept_reject_selector,:]
    
    squarejumpdist = np.linalg.norm(particles-particles_proposed, axis=1)**2
    jumping_distance_realized = np.linalg.norm(particles-particles_next, axis=1)**2
    performance_kernel_dict = {'energy': mh_ratio, 
                                'squarejumpdist':squarejumpdist,
                                'squarejumpdist_realized':jumping_distance_realized,
                                'acceptance_rate':accept_reject_selector.mean(),
                                'epsilon':epsilon, 
                                'L':1}
    if parametersmcmc['verbose']:
        print('acceptance rate: %s, esjd: %s, epsilon mean: %s, L mean: %s' %(accept_reject_selector.mean(), jumping_distance_realized.mean(), np.mean(epsilon), np.mean(1)))
    return particles_next, performance_kernel_dict


def leapfr_mom(x, cur_p, epsilon, gradf, temperature, half=False):
    """ makes a leapfrog step for the momentum, vectorized """
    if half:
        factor = 0.5
    else:
        factor = 1
    x = np.atleast_2d(x)
    cur_p = np.atleast_2d(cur_p)
    assert cur_p.shape == x.shape
    rval = cur_p + factor*epsilon*gradf(x, temperature)
    if (~np.isfinite(rval)).any():
        warnings.warn('some divergent behaviour')
    return rval

def leapfr_pos(cur_x, p, epsilon, inv_mass):
    """ makes a leapfrog step for the position, vectorized """
    cur_x = np.atleast_2d(cur_x)
    p = np.atleast_2d(p)
    assert cur_x.shape == p.shape
    rval = cur_x + epsilon*p.dot(inv_mass)
    if (~np.isfinite(rval)).any():
        warnings.warn('some divergent behaviour')
    #assert(not(~np.isfinite(rval)).any())
    return rval

def leapfr_mom_pos(cur_x, cur_p, epsilon, gradf, temperature, inv_mass):
    """ makes a leapfrog step for both the momentum and the position, vectorized """
    cur_x = np.atleast_2d(cur_x)
    cur_p = np.atleast_2d(cur_p)
    assert cur_x.shape == cur_p.shape
    new_p = leapfr_mom(cur_x, cur_p, epsilon, gradf, temperature)
    new_x = leapfr_pos(cur_x, new_p, epsilon, inv_mass)
    #TODO add the bounce routine here
    return (new_x, new_p)
    


def f_energy_kinetic(momentum, inv_mass):
    momentum = np.atleast_2d(momentum)
    energy_kinetic = 0.5*(momentum.dot(inv_mass)*momentum).sum(axis=1)
    return energy_kinetic

def f_energy(particles, momentum, log_density, temperature, inv_mass):
    momentum = np.atleast_2d(momentum)
    particles = np.atleast_2d(particles)
    energy_kinetic = f_energy_kinetic(momentum, inv_mass)
    energy_potential = -log_density(particles, temperature)
    return energy_kinetic, energy_potential


    
def proposalhmc(particles, parametersmcmc, temperedist, temperature):
    """
    function that computes the entire trajectory of leapfrog steps
    """
    assert isinstance(parametersmcmc, dict)
    assert temperature <= 1.
    assert temperature >= 0.

    covariance_matrix = parametersmcmc['covariance_matrix']
    #import ipdb; ipdb.set_trace()
    l_matrix = np.linalg.cholesky(covariance_matrix)
    l_matrix_inv = np.linalg.inv(l_matrix)
    N_particles, dim = particles.shape
    L_steps = int(np.mean(parametersmcmc['L_steps']))
    if 'epsilon_sampled' in parametersmcmc.keys():
        epsilon = parametersmcmc['epsilon_sampled']
    else:
        epsilon = parametersmcmc['epsilon']
    if epsilon.shape[0] == 1:
        epsilon = np.ones((N_particles,1))*epsilon

    #import ipdb; ipdb.set_trace()
    accept_reject = parametersmcmc['accept_reject']
    #import ipdb; ipdb.set_trace()
    #gradient_log_density = partial(temperedist.gradlogdensity, temperature=temperature)
    #log_density = partial(temperedist.logdensity, temperature=temperature)
    #mass_matrix = np.linalg.inv(covariance_matrix)
    
    # preallocate the arrays
    
    x = np.zeros((N_particles, dim, L_steps+1))
    energy_kinetic = np.zeros((N_particles, L_steps+1))
    energy_potential = np.zeros((N_particles, L_steps+1))
    marginal_ESJD = np.zeros((N_particles, L_steps+1))
    ESJD = np.zeros((N_particles, L_steps+1))
    p = np.zeros((N_particles, dim, L_steps+2)) # velocity

    # start with t = 0
    x[:, :, 0] = particles
    particles_next = np.zeros(particles.shape)
    p[:, :, 0] = np.random.normal(size=particles.shape).dot(l_matrix_inv)

    #pdb.set_trace()
    energy_kinetic[:, 0], energy_potential[:, 0] = f_energy(x[:, :, 0], p[:, :, 0], temperedist.logdensity, temperature, covariance_matrix)

    # First half-step of leapfrog.
    p[:, :, 1] = leapfr_mom(x[:, :, 0], p[:, :, 0], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
    x[:, :, 1] = leapfr_pos(x[:, :, 0], p[:, :, 1], epsilon, covariance_matrix)
    inter_momentum = leapfr_mom(x[:, :, 1], p[:, :, 1], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
    energy_kinetic[:, 1], energy_potential[:, 1] = f_energy(x[:, :, 1], inter_momentum, temperedist.logdensity, temperature, covariance_matrix)

    #import ipdb; ipdb.set_trace()
    for m_leapfrog_step in range(1, L_steps):
        #pdb.set_trace()
        (x[:, :, m_leapfrog_step+1], p[:, :, m_leapfrog_step+1]) = leapfr_mom_pos(x[:, :, m_leapfrog_step], p[:, :, m_leapfrog_step], epsilon, temperedist.gradlogdensity, temperature, covariance_matrix)
        # half step for the energy
        inter_momentum = leapfr_mom(x[:, :, m_leapfrog_step+1], p[:, :, m_leapfrog_step+1], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
        energy_kinetic[:, m_leapfrog_step+1], energy_potential[:, m_leapfrog_step+1] = f_energy(x[:, :, m_leapfrog_step+1], inter_momentum, temperedist.logdensity, temperature, covariance_matrix)
        marginal_ESJD[:, m_leapfrog_step] = ((x[:, :, m_leapfrog_step] - x[:, :, 0])*-p[:, :, m_leapfrog_step]).sum(axis=1)
        ESJD[:, m_leapfrog_step] = ((x[:, :, m_leapfrog_step] - x[:, :, 0])*(x[:, :, m_leapfrog_step] - x[:, :, 0])).sum(axis=1)
    #pdb.set_trace()
    #import ipdb; ipdb.set_trace()
    p[:, :, -1] = leapfr_mom(x[:, :, -1], p[:, :, -2], epsilon, temperedist.gradlogdensity, temperature, half=True)
    energy_kinetic[:, -1], energy_potential[:, -1] = f_energy(x[:, :, -1], p[:, :, -1], temperedist.logdensity, temperature, covariance_matrix)
    marginal_ESJD[:, -1] = ((x[:, :, -1] - x[:, :, 0])*-p[:, :, -1]).sum(axis=1)
    ESJD[:, -1] = ((x[:, :, -1] - x[:, :, 0])*(x[:, :, -1] - x[:, :, 0])).sum(axis=1)

    # accept reject routine
    energy_total = energy_kinetic + energy_potential
    #TODO: is there an error with the tuning? ESJD not actualized ? 
    if accept_reject: 
        Hamiltonian_cur = energy_total[:, 0]
        # chooses only the energy that corresponds to the number of leapfrog steps
        Hamiltonian_new = energy_total[:, -1]
        logprobaccept = Hamiltonian_cur-Hamiltonian_new
        unif_accept = np.log(np.random.rand(N_particles))
        # accept reject step
        if (~np.isfinite(logprobaccept)).any():
            warnings.warn('some divergent behaviour on #particles '+str(sum(~np.isfinite(logprobaccept))))
            selector = ~np.isfinite(logprobaccept)
            logprobaccept[selector] = -np.inf
            energy_total[selector] = np.inf
        accepted = unif_accept < logprobaccept
        rejected_particles = np.isnan(accepted)
        accepted = np.logical_and(accepted, ~rejected_particles)
        #pdb.set_trace()
        particles_next[accepted, :] = x[accepted, :, -1]
        particles_next[~accepted, :] = x[~accepted, :, 0]
        jumping_distance_realized = ((particles_next-particles)*(particles_next-particles)).sum(axis=1)
    # accept all particles if using HMC IS
    else: 
        #pdb.set_trace()
        accepted = np.ones(N_particles)
        particles_next = x[:, :, -1]
        jumping_distance_realized = ((particles_next-particles)*(particles_next-particles)).sum(axis=1)
        
    performance_kernel_dict = {'energy': energy_total, 
                                'squarejumpdist':ESJD,
                                'squarejumpdist_realized':jumping_distance_realized,
                                'acceptance_rate':accepted.mean(),
                                'epsilon':epsilon,
                                'L':L_steps}
    #import ipdb; ipdb.set_trace()
    if parametersmcmc['verbose']:
        print('acceptance rate: %s, esjd: %s, epsilon mean: %s, L mean: %s' %(accepted.mean(), jumping_distance_realized.mean(), np.mean(epsilon), np.mean(L_steps)))
    return particles_next, performance_kernel_dict


#from .cython.python_smchmc.smc_sampler_functions.cython.leapfrog_cython import loop_leapfrog as cython_loop_leapfrog
from .cython.leapfrog_cython import loop_leapfrog as cython_loop_leapfrog
from .cython.leapfrog_cython import leapfrog_transition_individual_parallel
from numba import jit

@jit()
def leapfrog_transition_individual(L_dict, x_all, p_all, epsilon_all, loggradient, temperature, covariance_matrix):
    """
    individual leapfrog step
    """
    #import ipdb; ipdb.set_trace()
    #iteration_L = int(L_dict[index_L]['iteration_L'])
    #L_step = int(L_dict[index_L]['L_step'])
    iteration_L = int(L_dict['iteration_L'])
    L_step = int(L_dict['L_step'])
    x, p = x_all[iteration_L, :, 1], p_all[iteration_L, :, 1]
    epsilon = epsilon_all[iteration_L]
    for m_leapfrog_step in range(1, L_step):
        (x, p) = leapfr_mom_pos(x, p, epsilon, loggradient, temperature, covariance_matrix)
    p = leapfr_mom(x, p, epsilon, loggradient, temperature, half=True)
    return(np.atleast_2d(x), np.atleast_2d(p), iteration_L)

@jit()
def loop_leapfrog(L_dict_list, x, p, epsilon, loggradient, temperature, covariance_matrix):
    for L_index, L_dict in enumerate(L_dict_list):
        #L_dict = {'iteration_L':iteration_L, 'L_step': L_step}
        #partial_leapfrog_transition(L_index)
        x_next, p_next, index_inter =  leapfrog_transition_individual(L_dict, x, p, epsilon, loggradient, temperature, covariance_matrix)
        x[L_index, :, 2], p[L_index, :, 2] = x_next, p_next
    return x, p
    

#from joblib import Parallel, delayed

def proposalhmc_parallel(particles, parametersmcmc, temperedist, temperature):
    """
    function that computes the entire trajectory of leapfrog steps
    """
    assert isinstance(parametersmcmc, dict)
    assert temperature <= 1.
    assert temperature >= 0.

    covariance_matrix = parametersmcmc['covariance_matrix']
    #import ipdb; ipdb.set_trace()
    l_matrix = np.linalg.cholesky(covariance_matrix)
    l_matrix_inv = np.linalg.inv(l_matrix)
    N_particles, dim = particles.shape
    L_steps = parametersmcmc['L_steps']
    
    if 'epsilon_sampled' in parametersmcmc.keys():
        epsilon = parametersmcmc['epsilon_sampled']
    else:
        epsilon = parametersmcmc['epsilon']
    if epsilon.shape[0] == 1:
        epsilon = np.ones((N_particles,1))*epsilon
    #import ipdb; ipdb.set_trace()
    if parametersmcmc['mean_L']:
        L_steps = np.ones(N_particles, dtype=int)*int(np.mean(L_steps))
    accept_reject = parametersmcmc['accept_reject']
    #import ipdb; ipdb.set_trace()
    gradient_log_density = partial(temperedist.gradlogdensity, temperature=temperature)
    log_density = partial(temperedist.logdensity, temperature=temperature)


    # preallocate the arrays
    x = np.zeros((N_particles, dim, 3))
    energy_kinetic = np.zeros((N_particles, 2))
    energy_potential = np.zeros((N_particles, 2))
    ESJD = np.zeros((N_particles, 1))
    p = np.zeros((N_particles, dim, 3)) # velocity

    # start with t = 0
    x[:, :, 0] = particles
    particles_next = np.zeros(particles.shape)
    p[:, :, 0] = np.random.normal(size=particles.shape).dot(l_matrix_inv)

    #pdb.set_trace()
    energy_kinetic[:, 0], energy_potential[:, 0] = f_energy(x[:, :, 0], p[:, :, 0], temperedist.logdensity, temperature, covariance_matrix)

    # First half-step of leapfrog.
    p[:, :, 1] = leapfr_mom(x[:, :, 0], p[:, :, 0], epsilon, temperedist.gradlogdensity, temperature, half=True)
    x[:, :, 1] = leapfr_pos(x[:, :, 0], p[:, :, 1], epsilon, covariance_matrix)

    if isinstance(L_steps, int):
        L_steps = np.ones(N_particles, dtype=int)*L_steps
    #L_steps = np.ones(N_particles, dtype=int)*int(np.mean(L_steps))
    
    #L_dict_list = [{'iteration_L': iteration_L, 'L_step': L_step} for iteration_L, L_step in enumerate(L_steps)]
    # if parametersmcmc['parallelize'] == 'mutltiprocess': 
    #     print('run parallelized code')
    #     try:
    #         partial_leapfrog_transition = partial(leapfrog_transition_individual, L_dict=L_dict_list, x_all=x, p_all=p, epsilon_all=epsilon, loggradient=temperedist.gradlogdensity, temperature=temperature, covariance_matrix=covariance_matrix)
    #         #res_parallel = parallelize_partial_over_chunks(partial_leapfrog_transition, range(len(L_dict_list)))
    #         #res_parallel = pool.map(partial_leapfrog_transition, range(len(L_dict_list)))
    #         res_parallel = []
    #         for i_index in range(len(L_dict_list)):
    #             res_parallel.append(partial_leapfrog_transition(i_index))
    #         #import ipdb; ipdb.set_trace()
    #         p_inter = np.array([ires[1] for ires in res_parallel])[:,0,:]
    #         x_inter = np.array([ires[0] for ires in res_parallel])[:,0,:]
    #         index_inter = np.array([ires[2] for ires in res_parallel])
    #         assert all([index_inter[i] <= index_inter[i+1] for i in range(len(index_inter)-1)])
    #         x[:, :, 2], p[:, :, 2] = x_inter, p_inter
    #     except:
    #         import ipdb; ipdb.set_trace()
    # elif parametersmcmc['parallelize'] == 'cython': 
    #     #import ipdb; ipdb.set_trace()
    #     x, p = loop_leapfrog(L_dict_list, x, p, epsilon, temperedist.gradlogdensity, temperature, covariance_matrix)
    #     #x, p = cython_loop_leapfrog(L_steps, x, p, epsilon, temperedist.gradlogdensity, temperature, covariance_matrix)
    #else:
    x_finished = np.zeros((N_particles, dim))
    p_finished = np.zeros((N_particles, dim))
    x_start = x[:, :, 1]
    p_start = p[:, :, 1]
    indicator_just_finished = np.zeros(len(L_steps), dtype=bool)
    indicator_active = np.ones(len(L_steps), dtype=bool)
    #import ipdb; ipdb.set_trace()
    for m_leapfrog_step in range(1, max(L_steps)+1):
        indicator_active = (L_steps >= m_leapfrog_step)
        (x_start[indicator_active,:], p_start[indicator_active,:]) = leapfr_mom_pos(x_start[indicator_active,:], p_start[indicator_active,:], epsilon[indicator_active], temperedist.gradlogdensity, temperature, covariance_matrix)
        # half step for the energy
        indicator_just_finished = L_steps==m_leapfrog_step
        if np.any(indicator_just_finished):
            p_finished[indicator_just_finished,:] = leapfr_mom(x_start[indicator_just_finished,:], p_start[indicator_just_finished,:], epsilon[indicator_just_finished], temperedist.gradlogdensity, temperature=temperature, half=True)
            x_finished[indicator_just_finished,:] = x_start[indicator_just_finished,:]
    x[:, :, -1] = x_finished
    p[:, :, -1] = p_finished
    #import ipdb; ipdb.set_trace()
    energy_kinetic[:, -1], energy_potential[:, -1] = f_energy(x[:, :, -1], p[:, :, -1], temperedist.logdensity, temperature, covariance_matrix)
    ESJD[:, -1] = np.linalg.norm(x[:, :, -1] - x[:, :, 0])**2
    

    # accept reject routine
    energy_total = energy_kinetic + energy_potential
    #TODO: is there an error with the tuning? ESJD not actualized ? 
    if accept_reject: 
        Hamiltonian_cur = energy_total[:, 0]
        # chooses only the energy that corresponds to the number of leapfrog steps
        Hamiltonian_new = energy_total[:, -1]
        logprobaccept = Hamiltonian_cur-Hamiltonian_new
        unif_accept = np.log(np.random.rand(N_particles))
        # accept reject step
        if (~np.isfinite(logprobaccept)).any():
            warnings.warn('some divergent behaviour on #particles '+str(sum(~np.isfinite(logprobaccept))))
            selector = ~np.isfinite(logprobaccept)
            logprobaccept[selector] = -np.inf
            energy_total[selector] = np.inf
        accepted = unif_accept < logprobaccept
        rejected_particles = np.isnan(accepted)
        accepted = np.logical_and(accepted, ~rejected_particles)
        #pdb.set_trace()
        particles_next[accepted, :] = x[accepted, :, -1]
        particles_next[~accepted, :] = x[~accepted, :, 0]
        jumping_distance_realized = ((particles_next-particles)*(particles_next-particles)).sum(axis=1)
    # accept all particles if using HMC IS
    else: 
        #pdb.set_trace()
        accepted = np.ones(N_particles)
        particles_next = x[:, :, -1]
        jumping_distance_realized = ((particles_next-particles)*(particles_next-particles)).sum(axis=1)
        
    performance_kernel_dict = {'energy': energy_total, 
                                'squarejumpdist':ESJD,
                                'squarejumpdist_realized':jumping_distance_realized,
                                'acceptance_rate':accepted.mean(),
                                'epsilon':epsilon,
                                'L':L_steps}
    #import ipdb; ipdb.set_trace()
    if parametersmcmc['verbose']: 
        print('acceptance rate: %s, esjd: %s, epsilon mean: %s, L mean: %s' %(accepted.mean(), jumping_distance_realized.mean(), np.mean(epsilon), np.mean(L_steps)))
    return particles_next, performance_kernel_dict

from help.gaussian_densities_etc import gaussian_vectorized

def proposalhmc_is(particles, u_randomness, parametersmcmc, temperedist, temperature):
    """
    function that computes the entire trajectory of leapfrog steps
    """
    assert isinstance(parametersmcmc, dict)
    assert temperature <= 1.
    assert temperature >= 0.

    covariance_matrix = parametersmcmc['covariance_matrix']
    #import ipdb; ipdb.set_trace()
    l_matrix = np.linalg.cholesky(covariance_matrix)
    l_matrix_inv = np.linalg.inv(l_matrix)
    N_particles, dim = particles.shape

    L_steps = int(np.mean(parametersmcmc['L_steps']))
    #L_steps = int(np.percentile(parametersmcmc['L_steps'], 0.5))
    if 'epsilon_sampled' in parametersmcmc.keys():
        epsilon = parametersmcmc['epsilon_sampled']
    else:
        #epsilon = np.atleast_2d(parametersmcmc['epsilon']).mean(axis=0)
        epsilon = np.atleast_2d(np.percentile(parametersmcmc['epsilon'], 50))
    #import ipdb; ipdb.set_trace()
    if epsilon.shape[0] == 1:
        epsilon = np.ones((N_particles,1))*epsilon

    
    # preallocate the arrays
    
    x = np.zeros((N_particles, dim, L_steps+1))
    energy_kinetic = np.zeros((N_particles, L_steps+1))
    energy_potential = np.zeros((N_particles, L_steps+1))
    marginal_ESJD = np.zeros((N_particles, L_steps+1))
    ESJD = np.zeros((N_particles, L_steps+1))
    p = np.zeros((N_particles, dim, L_steps+2)) # velocity

    # start with t = 0
    x[:, :, 0] = particles
    particles_next = np.zeros(particles.shape)
    if particles.shape != u_randomness.shape: 
        raise ValueError('shape differ')
    p[:, :, 0] = gaussian_vectorized(u_randomness).dot(l_matrix_inv)

    energy_kinetic[:, 0], energy_potential[:, 0] = f_energy(x[:, :, 0], p[:, :, 0], temperedist.logdensity, temperature, covariance_matrix)

    # First half-step of leapfrog.
    p[:, :, 1] = leapfr_mom(x[:, :, 0], p[:, :, 0], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
    x[:, :, 1] = leapfr_pos(x[:, :, 0], p[:, :, 1], epsilon, covariance_matrix)
    inter_momentum = leapfr_mom(x[:, :, 1], p[:, :, 1], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
    energy_kinetic[:, 1], energy_potential[:, 1] = f_energy(x[:, :, 1], inter_momentum, temperedist.logdensity, temperature, covariance_matrix)

    for m_leapfrog_step in range(1, L_steps):
        #pdb.set_trace()
        (x[:, :, m_leapfrog_step+1], p[:, :, m_leapfrog_step+1]) = leapfr_mom_pos(x[:, :, m_leapfrog_step], p[:, :, m_leapfrog_step], epsilon, temperedist.gradlogdensity, temperature, covariance_matrix)
        # half step for the energy
        inter_momentum = leapfr_mom(x[:, :, m_leapfrog_step+1], p[:, :, m_leapfrog_step+1], epsilon, temperedist.gradlogdensity, temperature=temperature, half=True)
        energy_kinetic[:, m_leapfrog_step+1], energy_potential[:, m_leapfrog_step+1] = f_energy(x[:, :, m_leapfrog_step+1], inter_momentum, temperedist.logdensity, temperature, covariance_matrix)
        marginal_ESJD[:, m_leapfrog_step] = ((x[:, :, m_leapfrog_step] - x[:, :, 0])*-p[:, :, m_leapfrog_step]).sum(axis=1)
        ESJD[:, m_leapfrog_step] = ((x[:, :, m_leapfrog_step] - x[:, :, 0])*(x[:, :, m_leapfrog_step] - x[:, :, 0])).sum(axis=1)

    p[:, :, -1] = leapfr_mom(x[:, :, -1], p[:, :, -2], epsilon, temperedist.gradlogdensity, temperature, half=True)
    energy_kinetic[:, -1], energy_potential[:, -1] = f_energy(x[:, :, -1], p[:, :, -1], temperedist.logdensity, temperature, covariance_matrix)
    marginal_ESJD[:, -1] = ((x[:, :, -1] - x[:, :, 0])*-p[:, :, -1]).sum(axis=1)
    ESJD[:, -1] = ((x[:, :, -1] - x[:, :, 0])*(x[:, :, -1] - x[:, :, 0])).sum(axis=1)

    # accept all particles if using HMC IS
    energy_total = energy_kinetic + energy_potential
    Hamiltonian_cur = energy_total[:, 0]
    # chooses only the energy that corresponds to the number of leapfrog steps
    Hamiltonian_new = energy_total[:, -1]
    logprobaccept = Hamiltonian_cur-Hamiltonian_new
    unif_accept = np.log(np.random.rand(N_particles))
    # accept reject step
    if (~np.isfinite(logprobaccept)).any():
        warnings.warn('some divergent behaviour on #particles '+str(sum(~np.isfinite(logprobaccept))))
        selector = ~np.isfinite(logprobaccept)
        logprobaccept[selector] = -np.inf
        energy_total[selector] = np.inf

    accepted = np.ones(N_particles, dtype=bool)
    #pdb.set_trace()
    particles_next[:, :] = x[accepted, :, -1]
    jumping_distance_realized = ((particles_next-particles)*(particles_next-particles)).sum(axis=1)
            
    performance_kernel_dict = {'energy': energy_total, 
                                'energy_kinetic' : energy_kinetic,
                                'energy_potential' : energy_potential,
                                'squarejumpdist':ESJD,
                                'squarejumpdist_realized':jumping_distance_realized,
                                'acceptance_rate':accepted.mean(),
                                'epsilon':epsilon,
                                'L':L_steps}
    #import ipdb; ipdb.set_trace()
    if parametersmcmc['verbose']:
        print('acceptance rate: %s, esjd: %s, epsilon mean: %s, L mean: %s' %(accepted.mean(), jumping_distance_realized.mean(), np.mean(epsilon), np.mean(L_steps)))
    return particles_next, performance_kernel_dict

