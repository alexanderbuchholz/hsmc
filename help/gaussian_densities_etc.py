# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:55:00 2016

@author: alex
"""
import math
import numpy as np
from scipy.stats import norm
from scipy.stats import t as t_student
#import ipdb as pdb

pi = math.pi
#from numba import jit


def gaussian(u, mu=0, sigma=np.array([1])):
    """
    function that generates a gaussian based on a realization of u
    handle two cases: general generation of a random variable,
                    and simple case in one dimension
    """
    ## needs still to get implemented!
    #if (mu==0 and sigma==np.array([1])):
    # calculate the cholesky decomposition
    #pdb.set_trace()
    if len(sigma.shape) != 1:
        sigmaL = np.linalg.cholesky(np.atleast_2d(sigma))
    else : sigmaL = np.diag(np.ones(u.shape)*np.sqrt(sigma))
    if np.all(mu==0):
        mu = np.zeros(u.shape)
    if u.shape != mu.shape:
        ValueError('the dimension of mu is not correct! need dimension = dimension u')
    z = norm.ppf(u)#sqrt(-2*log(u[0]))*cos(2*pi*u[1])
    return(np.squeeze(np.transpose(sigmaL.dot(z))+mu))

def gaussian_vectorized(u):
    """
    vectorized version of gaussian
    """
    z = norm.ppf(u)
    return(z)

def gaussian_standard(mu=0, sigma=np.array([1])):
    """
    with this sampler we do not control the sampling mechanism, only MC sampling
    """
    return np.random.multivariate_normal(mu, sigma)

def gaussian_density(x, mu, sigma):
    """
    return the value of the gaussian density
    """
    #pdb.set_trace()
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ (math.pow((2*pi), float(size)/2) * math.pow(det, 1.0/2))
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def student(u, mu=0, sigma=np.array([1]), df=3):
    """
    function that generates a t-student based on a realization of u
    handle two cases: general generation of a random variable,
                    and simple case in one dimension
    """
    #pdb.set_trace()
    ## needs still to get implemented!
    #if (mu==0 and sigma==np.array([1])):
    # calculate the cholesky decomposition
    if len(sigma.shape) != 1:
        sigmaL = np.linalg.cholesky(np.atleast_2d(sigma))
    else: sigmaL = np.diag(np.ones(u.shape)*np.sqrt(sigma))
    if np.all(mu==0):
        mu = np.zeros(u.shape)
    if u.shape != mu.shape:
        ValueError('the dimension of mu is not correct! need dimension = dimension u')
    z = t_student.ppf(u, df=df)#sqrt(-2*log(u[0]))*cos(2*pi*u[1])
    #pdb.set_trace()
    return(np.squeeze(np.transpose(sigmaL.dot(z))+mu))


def student_density(x, mu, sigma, df=5):
    """
    multivariate student density, evaluated at x
    """
    #pdb.set_trace()
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        Num = math.gamma(1. * (size+df)/2)
        Denom = ( math.gamma(1.*df/2) * pow(df*math.pi,1.*size/2) * pow(np.linalg.det(sigma),1./2) * pow(1 + (1./df)*np.dot(np.dot((x - mu),np.linalg.inv(sigma)), (x - mu)),1.* (size+df)/2))
        density_output = 1. * Num / Denom

        return density_output
    else:
        raise NameError("The dimensions of the input don't match")


def f_kernel_value(epsilon_t, delta_values, f_kernel):
    """
    function that returns the kernel values for an array of several y values
    """
    #multiple_y = delta_values.shape[0]
    #kernel_values = np.zeros(multiple_y)
    #pdb.set_trace()
    kernel_values = epsilon_t*f_kernel(delta_values/ epsilon_t)
    #for i_multiple_y in range(multiple_y):
    #    kernel_values[i_multiple_y] = epsilon_t*f_kernel(delta_values[i_multiple_y]/ epsilon_t)
    return kernel_values


def gaussian_move(theta, u, sigma):
    """
        the kernel that moves theta
        need to use an adaptive version of sigma
    """
    theta_prop = gaussian(u, theta, sigma)
    return(theta_prop)
# the new theta needs to be within a given range

def student_move(theta, u, sigma):
    """
        the kernel that moves theta
        need to use an adaptive version of sigma
    """
    theta_prop = student(u, theta, sigma)
    return(theta_prop)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def weighted_choice(weights, u):
    """
    a function that choses a random ancestor
    """
    #pdb.set_trace()
    if sum(weights) > 1.001:
        raise ValueError('weights greater than one!')
    upto = 0
    counter = 0
    for w in weights:
        if upto + w >= u:
            return counter
        upto += w
        counter += 1
    assert False, "Shouldn't get here"

# define kernels that can be used for the ABC
def uniform_kernel(x):
    return(0.5*(np.abs(x)<1))

def gaussian_kernel(x):
    return(1./np.sqrt(2.*pi)*np.exp(-0.5*x**2))

def break_if_nan(myarray):
    if np.isnan(myarray).any():
        print "nan values present!"
        #pdb.set_trace()
    else: pass

def break_if_negative(myarray):
    if (myarray<0.).any():
        print "negative values present!"
        pdb.set_trace()
    else: pass

if __name__ == '__main__':
    p = 5
    n = 1000
    x = np.random.normal(size = (n,p))
    mu = np.zeros(p)
    #x = np.ones(p)

    sigma = np.eye(p)
    from scipy.stats import multivariate_normal
    multivariate_normal.pdf(x, mean=mu, cov=sigma)
    np.array([gaussian_density(x[i,:], mu, sigma) for i in range(n)])
