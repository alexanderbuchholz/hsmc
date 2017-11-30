# target distribution
from __future__ import division

import numpy as np
cimport numpy as np
from libc.math cimport tgamma
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma

def priorlogdens(np.ndarray particles):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    cdef np.ndarray covar
    covar = np.eye(particles.shape[1])
    return(multivariate_normal.logpdf(particles, cov=covar))

def priorgradlogdens(np.ndarray particles):
    """
    particles [N_partiles, dim]
    """
    return -particles

def targetlogdens_normal(np.ndarray particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    multivariate normal
    """
    cdef np.ndarray _mean
    _mean = parameters['targetmean']
    cdef np.ndarray covar
    covar = parameters['targetvariance']
    return multivariate_normal.logpdf(particles, mean=_mean, cov=parameters['targetvariance'])

def targetgradlogdens_normal(np.ndarray particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    """
    cdef np.ndarray meaned_particles
    cdef np.ndarray l_inv_cov
    meaned_particles = particles - parameters['targetmean']
    l_inv_cov = parameters['l_targetvariance_inv']
    return -meaned_particles.dot(l_inv_cov.transpose())


def targetlogdens_student(np.ndarray particles, parameters):
    """
    the log density of the t distribution, unnormalized
    """
    cdef np.ndarray factor_particles1 
    cdef np.ndarray factor_particles2 
    cdef np.ndarray _mean
    cdef np.ndarray inv_covar
    cdef np.ndarray covar
    #cdef int df
    #cdef int dim
    cdef float Z
    cdef float half
    half = 0.5
    df = float(parameters['df'])
    dim = float(parameters['dim'])
    _mean = parameters['targetmean']
    inv_covar = parameters['targetvariance_inv']
    covar = parameters['targetvariance']

    factor_particles1 = (particles-_mean).dot(inv_covar)*(particles-_mean)
    factor_particles2 = factor_particles1.sum(axis=1)
    sum1 = float(df+dim)
    factor_ind1 = np.multiply(sum1, half, casting='unsafe', dtype=np.float64)
    factor_ind2 = np.multiply(df, half, casting='unsafe', dtype=np.float64)
    factor_ind3 = np.multiply(df, np.pi, casting='unsafe', dtype=np.float64)
    factor_ind4 = np.multiply(dim, half, casting='unsafe', dtype=np.float64)
    Z = tgamma(factor_ind1)/tgamma(factor_ind2)*(factor_ind3**(factor_ind4))*np.linalg.det(covar)**0.5
    return(-(df+dim)/2.*np.log(1+(1./df)*factor_particles2)+np.log(Z))

def targetlogdens_student_new(np.ndarray particles, 
                            np.ndarray _mean, 
                            np.ndarray inv_covar, 
                            np.ndarray covar, 
                            int df,
                            int dim):
    """
    the log density of the t distribution, unnormalized
    """
    cdef np.ndarray factor_particles1 
    cdef np.ndarray factor_particles2 
    #cdef np.ndarray _mean
    #cdef np.ndarray inv_covar
    #cdef np.ndarray covar
    #cdef int df
    #cdef int dim
    cdef float Z
    cdef float half
    half = 0.5
    #df = float(parameters['df'])
    #dim = float(parameters['dim'])
    #_mean = parameters['targetmean']
    #inv_covar = parameters['targetvariance_inv']
    #covar = parameters['targetvariance']

    factor_particles1 = (particles-_mean).dot(inv_covar)*(particles-_mean)
    factor_particles2 = factor_particles1.sum(axis=1)
    sum1 = float(df+dim)
    factor_ind1 = np.multiply(sum1, half, casting='unsafe', dtype=np.float64)
    factor_ind2 = np.multiply(df, half, casting='unsafe', dtype=np.float64)
    factor_ind3 = np.multiply(df, np.pi, casting='unsafe', dtype=np.float64)
    factor_ind4 = np.multiply(dim, half, casting='unsafe', dtype=np.float64)
    Z = tgamma(factor_ind1)/tgamma(factor_ind2)*(factor_ind3**(factor_ind4))*np.linalg.det(covar)**0.5
    return(-(df+dim)/2.*np.log(1+(1./df)*factor_particles2)+np.log(Z))



def targetgradlogdens_student(np.ndarray particles, parameters):
    """
    computes the gradient of the t distribution
    """
    cdef np.ndarray factor_particles1 
    cdef np.ndarray factor_particles2 
    cdef np.ndarray _mean
    cdef np.ndarray inv_covar
    cdef np.ndarray covar
    cdef float df
    cdef float dim
    cdef float Z

    cdef float factor_1
    cdef np.ndarray factor_2
    cdef np.ndarray factor_3

    df = float(parameters['df'])
    dim = float(parameters['dim'])
    _mean = parameters['targetmean']
    inv_covar = parameters['targetvariance_inv']
    covar = parameters['targetvariance']

    factor_1 = -(df+dim)/2.
    factor_particles1 = (particles-_mean).dot(inv_covar)*(particles-_mean)
    factor_particles2 = factor_particles1.sum(axis=1)
    factor_2 = 1+(1./df)*factor_particles2
    factor_3 = (2./df)*(particles-_mean).dot(inv_covar)
    return(factor_1*factor_3/factor_2[:,np.newaxis])
