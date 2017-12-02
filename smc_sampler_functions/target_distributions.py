# target distribution
from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.special import gamma
import sys
sys.path.append("../help/")
sys.path.append("/home/alex/Dropbox/smc_hmc/python_smchmc/help")

#from help.log_sum_exp import logplus_one
from log_sum_exp import logplus_one
from numba import jit

@jit(nopython=True)
def h_sigmoid(x):
    return(1./(1+np.exp(-x)))


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


#@jit(nopython=True)
def targetlogdens_logistic_help(particles, X, y):
    """
    likelihood of the logistic regression
    """
    import ipdb; ipdb.set_trace()
    dot_product = np.dot(X, particles.transpose())
    #sigmoid_value = logplus_one(dot_product)
    sigmoid_value = np.log(1+np.exp(-dot_product))
    likelihood_value = (-y*sigmoid_value + (1-y)*dot_product*sigmoid_value).sum(axis=0)
    return likelihood_value-np.linalg.norm(particles)**2

def targetlogdens_logistic(particles, parameters):
    return targetlogdens_logistic_help(particles, parameters['X_all'], parameters['y_all'])

#@jit(nopython=True)
def targetgradlogdens_logistic_help(particles, X, y):
    dot_product = np.dot(X, particles.transpose())
    part_1 = np.dot((y*X).transpose(), (1.-h_sigmoid(dot_product)))
    part_2 = np.dot(((y-1)*X).transpose(), h_sigmoid(dot_product))
    grad_new = (part_1 + part_2)
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
        N = 1000
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
        particles = np.random.normal(size=(1, dim))
        parameters = {'X_all': X_all, 'y_all': y_all}

        parameters = f_dict_logistic_regression(2)
        particles = np.ones((2, parameters['X_all'].shape[1]))
        #logistic_log_likelihood_jit = jit(logistic_log_likelihood, nopython=True)
        #logistic_log_likelihood_jit(particles, X_all, y_all)
        #import yappi
        #yappi.start()
        targetlogdens_logistic(particles, parameters)
        targetgradlogdens_logistic(particles, parameters)
        targetgradlogdens_logistic_notjit(particles, X_all, y_all)
        #yappi.get_func_stats().print_all()
        import ipdb as pdb; pdb.set_trace()