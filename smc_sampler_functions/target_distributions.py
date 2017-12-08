# target distribution


#from __future__ import division

import numpy as np
import numexpr as ne
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma
import sys
sys.path.append("../help/")

#from help.log_sum_exp import logplus_one
#from log_sum_exp import logplus_one
from numba import jit
from functools import partial

#@jit(nopython=True)
def h_sigmoid(x):
    return(1./(1+np.exp(-x)))


def priorlogdens(particles, parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    return(multivariate_normal.logpdf(particles, cov=np.eye(particles.shape[1])))

def priorsampler(parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    N_particles = parameters['N_particles']
    dim = parameters['dim']
    res = np.random.normal(size=(N_particles, dim))
    return(res)


def priorgradlogdens(particles, parameters):
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
    inv_cov = parameters['targetvariance_inv']
    return -meaned_particles.dot(inv_cov)


def targetlogdens_ring(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    multivariate normal
    """
    return -0.5*(np.linalg.norm(particles, axis=1)**2 - 4)**2

def targetgradlogdens_ring(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    """
    return -(np.linalg.norm(particles, axis=1)**2 - 4)[:,np.newaxis]*particles




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


#@jit(nopython=True)

#@profile
def targetlogdens_logistic_help(particles, X, y):
    """
    likelihood of the logistic regression
    """
    #import ipdb; ipdb.set_trace()
    particles = np.atleast_2d(particles)
    dot_product = np.dot(X, particles.transpose())
    #sigmoid_value = logplus_one(dot_product)
    #sigmoid_value1 = logexp(dot_product)
    sigmoid_value1 = ne.evaluate('log(1+exp(-dot_product))')
    # likelihood_value = (-y*sigmoid_value1 - (1-y)*(dot_product+sigmoid_value1)).sum(axis=0)
    likelihood_value = ((y-1)*dot_product-sigmoid_value1).sum(axis=0)
    #import ipdb; ipdb.set_trace()
    return (likelihood_value-0.5*np.linalg.norm(particles, axis=1)**2)


def targetlogdens_logistic_help_safe(particles, X, y):
    """
    likelihood of the logistic regression
    """
    #import ipdb; ipdb.set_trace()
    particles = np.atleast_2d(particles)
    dot_product = np.dot(X, particles.transpose())
    #sigmoid_value = logplus_one(dot_product)
    sigmoid_value1 = np.log(1./(1+np.exp(-dot_product)))
    sigmoid_value2 = np.log(1-1./(1+np.exp(-dot_product)))
    likelihood_value = (y*sigmoid_value1 + (1-y)*sigmoid_value2).sum(axis=0)
    return (likelihood_value-0.5*np.linalg.norm(particles, axis=1)**2)


def targetlogdens_logistic(particles, parameters):
    return targetlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])

#
#@jit(nopython=True)
def logit(x):
    return np.exp(-x)/(1.+np.exp(-x))
    #return ne.evaluate('exp(-x)/(1+exp(-x))')

logit(1)

#@jit(nopython=True)
def logexp(x):
    #return ne.evaluate('log(1+exp(x))')
    return np.log(1.+np.exp(-x))
logexp(1)


#@profile
def targetgradlogdens_logistic_help(particles, X, y):
    #import ipdb; ipdb.set_trace()
    particles = np.atleast_2d(particles)
    dot_product = np.dot(X, particles.transpose())
    part1 = ((y-1)*X).sum(axis=0)
    part2 = X[:,:,np.newaxis]
    part3 = ne.evaluate('exp(-dot_product)/(1+exp(-dot_product))')
    #part3 = logit(dot_product)
    part3 = part3[:, np.newaxis]
    part4 = ne.evaluate('part2*part3')
    part5 = (part4).sum(axis=0)
    grad_new = part1[:,np.newaxis]+part5
    gradient_pi_0 = -particles.transpose()
    grad_final = (grad_new+gradient_pi_0).transpose()
    return grad_final

def targetgradlogdens_logistic(particles, parameters):
    return targetgradlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])

def f_dict_logistic_regression(dim):
    if dim == 31:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X_all = data.data
        N_obs = X_all.shape[0]
        X_all = (X_all-X_all.mean(axis=0))/X_all.var(axis=0)
        X_all = np.hstack((np.ones((N_obs,1)), X_all))
        y_all = data.target
        y_all = y_all[:, np.newaxis]
    else: 
        np.random.seed(1)
        N = 100
        X_all = np.random.randn(N, dim)
        beta = np.ones(dim)
        proba = 1./(1+np.exp(-np.dot(X_all, beta)))
        #import ipdb as pdb; pdb.set_trace()
        y_all = (proba > np.random.random(N))*1
        y_all = y_all[:, np.newaxis]
    parameters = {'X_all': X_all, 'y_all': y_all}
    return(parameters)

def targetgradlogdens_logistic_notjit(particles, X, y):
    #pdb.set_trace()
    #import ipdb; ipdb.set_trace()
    dot_product = np.dot(X, particles.transpose())[np.newaxis,:,:]
    part_1 = (y*X).transpose()[:,:,np.newaxis]*(1.-h_sigmoid(dot_product))
    part_2 = ((y-1)*X).transpose()[:,:,np.newaxis]*h_sigmoid(dot_product)
    grad_new = (part_1 + part_2).sum(axis=1)
    gradient_pi_0 = -particles.transpose()
    #pdb.set_trace()
    #grad_function_single = grad(self.log_density)
    #test = np.array([grad_function_single(particles[k:k+1,:]) for k in range(particles.shape[0])])[:,0,:]
    #return(tempering*np.array([grad_function_single(particles[k:k+1,:]) for k in range(particles.shape[0])])[:,0,:])
    grad_final = (grad_new+gradient_pi_0).transpose()
    return grad_final


def approx_gradient(function, x, h=0.00001):
    dim = x.shape[1]
    grad_vector = np.zeros(x.shape)
    for i in range(dim):
        x_1 = np.copy(x)
        x_2 = np.copy(x)
        x_1[:,i] = x[:,i]+h
        x_2[:,i] = x[:,i]-h
        grad_vector[:,i] = (function(x_1)-function(x_2))/(2*h)
    return(grad_vector)



if __name__ == '__main__':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X_all = data.data
        N_obs = X_all.shape[0]

        X_all = (X_all-X_all.mean(axis=0))/X_all.var(axis=0)
        X_all = np.hstack((np.ones((N_obs,1)), X_all))

        y_all = data.target
        #selector = y_all==0
        #y_all[selector] = -1
        y_all = y_all[:, np.newaxis]
        dim = X_all.shape[1]
        N_obs = X_all.shape[0]
        #particles = np.random.normal(size=(1, dim))
        #parameters = {'X_all': X_all, 'y_all': y_all}
        #import ipdb as pdb; pdb.set_trace()
        parameters = f_dict_logistic_regression(10)
        #particles = np.ones((1,parameters['X_all'].shape[1]))
        particles = np.random.normal(size=(1, parameters['X_all'].shape[1]))

        #logistic_log_likelihood_jit = jit(logistic_log_likelihood, nopython=True)
        #logistic_log_likelihood_jit(particles, X_all, y_all)
        targetlogdens_logistic(particles, parameters)
        targetgradlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])
        #import yappi
        #yappi.start()
        targetgradlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])
        #yappi.get_func_stats().print_all()
        targetgradlogdens_logistic(particles, parameters)
        #import ipdb; ipdb.set_trace()
        #targetgradlogdens_logistic_notjit(particles, X_all, y_all)
        #
        partial_target_max = partial(targetlogdens_logistic, parameters=parameters) 
        diff = approx_gradient(partial_target_max, particles) - targetgradlogdens_logistic(particles, parameters)
        assert np.linalg.norm(diff)<0.00001
        diff = targetlogdens_logistic_help_safe(particles, parameters['X_all'], parameters['y_all']) - targetlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])
        assert np.linalg.norm(diff)<0.00001
        import ipdb; ipdb.set_trace()
        particles_test = np.random.normal(size=(100, parameters['X_all'].shape[1]))
        for N in range(99):
            diff = approx_gradient(partial_target_max, particles_test[N:N+1], 0.0000001) - targetgradlogdens_logistic(particles_test, parameters)[N:N+1,:]
            assert np.linalg.norm(diff)<0.00001