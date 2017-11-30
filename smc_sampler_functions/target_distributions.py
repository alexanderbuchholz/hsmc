# target distribution
from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma

def priorlogdens(particles):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    return(multivariate_normal.logpdf(particles, cov=np.eye(particles.shape[1])))

def priorgradlogdens(particles):
    """
    particles [N_partiles, dim]
    """
    return -particles

def targetlogdens_normal(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    multivariate normal
    """
    return multivariate_normal.logpdf(particles, mean=parameters['targetmean'], cov=parameters['targetvariance'])

def targetgradlogdens_normal(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    """
    meaned_particles = particles - parameters['targetmean']
    l_inv_cov = parameters['l_targetvariance_inv']
    return -meaned_particles.dot(l_inv_cov.transpose())


def targetlogdens_student(particles, parameters):
    """
    the log density of the t distribution, unnormalized
    """
    factor_particles = (particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])*(particles-parameters['targetmean'])
    factor_particles = factor_particles.sum(axis=1)
    Z = gamma((parameters['df']+parameters['dim'])/2.)/(gamma((parameters['df'])/2.)*(parameters['df']*np.pi)**(parameters['dim']/2.)*np.linalg.det(parameters['targetvariance'])**0.5)
    return(-(parameters['df']+parameters['dim'])/2*np.log(1+(1./parameters['df'])*factor_particles)+np.log(Z))


def targetgradlogdens_student(particles, parameters):
    """
    computes the gradient of the t distribution
    """
    factor_1 = -(parameters['df']+parameters['dim'])/2
    factor_particles = (particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])*(particles-parameters['targetmean'])
    factor_particles = factor_particles.sum(axis=1)
    factor_2 = 1+(1./parameters['df'])*factor_particles
    factor_3 = (2./parameters['df'])*(particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])
    return(factor_1*factor_3/factor_2[:,np.newaxis])
