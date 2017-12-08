# target distributions log gaussian cox model

import numpy as np
import numexpr as ne
from numba import jit
from matplotlib import pyplot as plt

# parameters of the model
def f_dict_log_cox(N):
    '''
    N is the gridsize
    '''
    np.random.seed(1)
    beta = 1./33.
    sigma2 = 1.91
    mu = np.log(126)-sigma2/2
    # dim/ gridsize
    dim = N**2
    mu_mean = np.ones((dim,1))*mu
    parameters = {'beta' : beta, 'sigma2' : sigma2, 'mu' : mu, 'dim': dim, 'mu_mean' : mu_mean, 'N' :N}
    covariance_matrix = f_covariance_matrix(N, beta, sigma2)
    covar_reshaped = covariance_matrix.reshape(dim, dim)
    inv_covar = np.linalg.inv(covar_reshaped)
    parameters['covar'] = covar_reshaped
    parameters['inv_covar'] = inv_covar
    parameters['l_covar'] = np.linalg.cholesky(covar_reshaped)
    X_true = np.random.normal(size=(dim)).dot(parameters['l_covar'])+mu_mean.flatten()
    delta = (1./dim)*np.exp(X_true)
    Y = np.random.poisson(delta)[:,np.newaxis]
    parameters['Y'] = Y
    parameters['X_true'] = X_true
    return parameters

#parameters = f_dict_log_cox(30)

from numba import jit # use jit, so that loops are fast

@jit(nopython=True)
def covariance_function(i, j, i_prime, j_prime, beta, sigma2):
    exponent1 = (i-i_prime)**2+(j-j_prime)**2
    exponent2 = -exponent1**0.5
    res = sigma2*np.exp(exponent2/(64*beta))
    return(res)


@jit(nopython=True)
def f_covariance_matrix(N, beta, sigma2):
    covariance_matrix = np.zeros((N, N, N, N))
    for i in range(N):
        for i_prime in range(N):
            for j in range(N):
                for j_prime in range(N):
                    covariance_matrix[i, j, i_prime, j_prime] = covariance_function(i, j, i_prime, j_prime, beta, sigma2)
    return(covariance_matrix)


def targetlogdens_log_cox(X, parameters):
    """
    X is a matrix of size [dim, particles]
    Y is a matrix of size [dim, 1]
    """
    X = X.transpose()
    Y = parameters['Y']
    assert X.shape[1]>=1
    assert Y.shape[1]==1
    dim = parameters['dim']
    assert X.shape[0]==dim
    assert Y.shape[0]==dim
    inv_covar = parameters['inv_covar']
    mu_mean = parameters['mu_mean']
    assert mu_mean.shape[1]==1
    part1 = ne.evaluate('Y*X')
    part2 = ne.evaluate('-(1/dim)*exp(X)')
    part3 = (part1+part2).sum(axis=0)
    meaned_x = ne.evaluate('X-mu_mean')
    part4 = -0.5*((inv_covar).dot(meaned_x)*meaned_x).sum(axis=0)
    res = part3+part4
    #import pdb; pdb.set_trace()
    return(res)


def priorlogdens_log_cox(X, parameters):
    X = X.transpose()
    dim = parameters['dim']
    inv_covar = parameters['inv_covar']
    mu_mean = parameters['mu_mean']
    meaned_x = ne.evaluate('X-mu_mean')
    res = -0.5*(inv_covar.dot(meaned_x)*meaned_x).sum(axis=0)
    return(res)

def priorsampler_log_cox(parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    N_particles = parameters['N_particles']
    dim = parameters['dim']
    mu_mean = parameters['mu_mean']
    res = np.random.normal(size=(N_particles, dim)).dot(parameters['l_covar'])+mu_mean.flatten()
    return(res)


def targetgradlogdens_log_cox(X, parameters):
    """
    X is a matrix of size [dim, particles]
    Y is a matrix of size [dim, 1]
    parameters is the dictionary with the relevant variables
    """
    X = X.transpose()
    Y = parameters['Y']
    assert X.shape[1]>=1
    assert Y.shape[1]==1
    dim = parameters['dim']
    assert X.shape[0]==dim
    assert Y.shape[0]==dim
    inv_covar = parameters['inv_covar']
    mu_mean = parameters['mu_mean']
    assert mu_mean.shape[1]==1
    
    part1 = ne.evaluate('-(1/dim)*exp(X)')
    part2 = Y+part1
    meaned_x = ne.evaluate('X-mu_mean')
    part3 = -(inv_covar).dot(meaned_x)
    res = part2+part3
    #import pdb; pdb.set_trace()
    return(res.transpose())


def priorgradlogdens_log_cox(X, parameters):
    X = X.transpose()
    dim = parameters['dim']
    inv_covar = parameters['inv_covar']
    mu_mean = parameters['mu_mean']
    meaned_x = ne.evaluate('X-mu_mean')
    res = -inv_covar.dot(meaned_x)
    return(res.transpose())