# target distribution


#from __future__ import division

import numpy as np
import numexpr as ne
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import zscore
from scipy.special import gamma, erf, gammaln
from scipy.special import erf as serf
import sys
sys.path.append("../help/")
sys.path.append("../smc_sampler_functions/")
sys.path.append("../smc_sampler_functions/data/")
if True: 
    try:
        from help.gaussian_densities_etc import gaussian_vectorized
    except:
        from gaussian_densities_etc import gaussian_vectorized
#from help.log_sum_exp import logplus_one
#from log_sum_exp import logplus_one
import pandas as pd
import numba
from numba import jit, f8, float64
from functools import partial


#@jit(nopython=True)
def h_sigmoid(x):
    return(1./(1+np.exp(-x)))


def priorlogdens(particles, parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    if 'prior_mean' in parameters.keys() and 'prior_var' in parameters.keys():
        #import ipdb; ipdb.set_trace()
        res = multivariate_normal.logpdf(particles, mean=parameters['prior_mean'].flatten(), cov=parameters['prior_var'])
    else: 
        factor_variance = parameters['factor_variance']
        res = multivariate_normal.logpdf(particles, mean= np.zeros(parameters['dim']), cov=factor_variance*np.eye(particles.shape[1]))
    #return(multivariate_normal.logpdf(particles, cov=np.eye(particles.shape[1])))
    return(res)

def priorsampler(parameters, u_randomness):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    #N_particles = parameters['N_particles']
    #dim = parameters['dim']
    #res = np.random.normal(size=(N_particles, dim))
    if 'prior_mean' in parameters.keys() and 'prior_var' in parameters.keys():
        res = gaussian_vectorized(u_randomness).dot(np.linalg.cholesky(parameters['prior_var']))+parameters['prior_mean'].transpose()
        #import ipdb; ipdb.set_trace()
    else:
        factor_variance = parameters['factor_variance']
        res = gaussian_vectorized(u_randomness)*(factor_variance**0.5)
    #res = gaussian_vectorized(u_randomness)
    return(res)


def priorgradlogdens(particles, parameters):
    """
    particles [N_partiles, dim]
    """
    if 'prior_mean' in parameters.keys() and 'prior_var' in parameters.keys():
        res = -(particles-parameters['prior_mean'].transpose()).dot(parameters['prior_inv_var'])
    else: 
        factor_variance = parameters['factor_variance']
        res = -particles/factor_variance
    return res
    #return -particles


def priorlogdens_student(particles, parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    df = parameters['df']
    dim = parameters['dim']
    dict_prior = {'targetmean': np.zeros(dim),
        'targetvariance_inv' : np.eye(dim),
        'targetvariance' : np.eye(dim),
        'logdet_targetvariance': np.zeros(1),
        'dim' : dim,
        'df': 3
        }
    res = targetlogdens_student(particles, dict_prior)
    return(res)

def priorsampler_student(parameters, u_randomness):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    #N_particles = parameters['N_particles']
    #dim = parameters['dim']
    #res = np.random.normal(size=(N_particles, dim))
    df = parameters['df']
    dim = parameters['dim']
    N_particles = parameters['N_particles']
    res = np.random.standard_t(df, size=(N_particles, dim))
    return(res)


def priorgradlogdens_student(particles, parameters):
    """
    particles [N_partiles, dim]
    """
    df = parameters['df']
    dim = parameters['dim']
    dict_prior = {'targetmean': np.zeros(dim),
        'targetvariance_inv' : np.eye(dim),
        'targetvariance' : np.eye(dim),
        'logdet_targetvariance': np.zeros(1),
        'dim' : dim,
        'df': 3
        }
    res = targetgradlogdens_student(particles, dict_prior)
    return res
    #return -particles



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

def priorlogdens_mix(particles, parameters):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    shift = parameters['mean_shift']
    return(multivariate_normal.logpdf(particles-shift, cov=5.*np.eye(particles.shape[1])))

def priorsampler_mix(parameters, u_randomness):
    """
    particles [N_partiles, dim]
    multivariate normal
    """
    #N_particles = parameters['N_particles']
    #dim = parameters['dim']
    #res = np.random.normal(size=(N_particles, dim))
    shift = parameters['mean_shift']
    res = gaussian_vectorized(u_randomness)*(5.**0.5)+shift
    return(res)


def priorgradlogdens_mix(particles, parameters):
    """
    particles [N_partiles, dim]
    """
    shift = parameters['mean_shift']
    return -(particles-shift)/5.

def targetlogdens_normal_mix(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    multivariate normal
    """
    mean1 = parameters['targetmean']
    mean2 = mean1*-1.
    proportion = parameters['proportion']
    return np.log(proportion*multivariate_normal.pdf(particles, mean=mean1, cov=parameters['targetvariance1'])+(1.-proportion)*multivariate_normal.pdf(particles, mean=mean2, cov=parameters['targetvariance2']))
    #return multivariate_normal.logpdf(particles, mean=parameters['targetmean'], cov=parameters['targetvariance'])

def targetgradlogdens_normal_mix(particles, parameters):
    """
    particles [N_partiles, dim]
    parameters dict 'targetmean' 'targetvariance'
    """
    mean1 = parameters['targetmean']
    mean2 = mean1*-1.

    meaned_particles1 = particles - mean1
    meaned_particles2 = particles - mean2
    inv_cov1 = parameters['targetvariance_inv1']
    inv_cov2 = parameters['targetvariance_inv2']
    proportion = parameters['proportion']
    #import pdb; pdb.set_trace()
    pdf1 = proportion*np.atleast_2d(multivariate_normal.pdf(particles, mean=mean1, cov=parameters['targetvariance1'])).transpose()
    pdf2 = (1-proportion)*np.atleast_2d(multivariate_normal.pdf(particles, mean=mean2, cov=parameters['targetvariance2'])).transpose()
    numerator = -meaned_particles1.dot(inv_cov1)*pdf1 - meaned_particles2.dot(inv_cov2)*pdf2
    denominator = pdf1+pdf2
    return numerator/denominator



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



#@profile
def targetlogdens_student(particles, parameters):
    """
    the log density of the t distribution, unnormalized
    """
    df = parameters['df']
    dim = parameters['dim']
    targetmean = parameters['targetmean']
    targetvariance_inv = parameters['targetvariance_inv']
    logdet_targetvariance = parameters['logdet_targetvariance']
    #det_targetvariance = parameters['det_targetvariance']
    
    particles_meaned = particles-targetmean
    mat_prod = np.dot(particles_meaned, targetvariance_inv)
    factor_particles = mat_prod*particles_meaned
    factor_particles = factor_particles.sum(axis=1)
    #import ipdb; ipdb.set_trace()
    #Z = gamma((df+dim)/2.)/(gamma((df)/2.)*(df*np.pi)**(dim/2.)* det_targetvariance**0.5)
    #Z = gamma((df+dim)/2.)/(gamma((df)/2.)*(df*np.pi)**(dim/2.))
    log_Z = gammaln((df+dim)/2.)-gammaln((df)/2.)-(dim/2.)*np.log(df*np.pi)
    #return(-(df+dim)/2*np.log(1.+(1./df)*factor_particles)+np.log(Z))
    factor1 = -(df+dim)/2
    factor2 = ne.evaluate('log(1.+(1./df)*factor_particles)')
    #return(factor1*factor2+np.log(Z)-0.5*logdet_targetvariance)
    return(factor1*factor2+log_Z-0.5*logdet_targetvariance)

#@profile
def targetlogdens_student_old(particles, parameters):
    """
    the log density of the t distribution, unnormalized
    """
    factor_particles = (particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])*(particles-parameters['targetmean'])
    factor_particles = factor_particles.sum(axis=1)
    Z = gamma((parameters['df']+parameters['dim'])/2.)/(gamma((parameters['df'])/2.)*(parameters['df']*np.pi)**(parameters['dim']/2.)*np.linalg.det(parameters['targetvariance'])**0.5)
    return(-(parameters['df']+parameters['dim'])/2*np.log(1+(1./parameters['df'])*factor_particles)+np.log(Z))


#@profile
def targetgradlogdens_student(particles, parameters):
    """
    computes the gradient of the t distribution
    """
    df = parameters['df']
    dim = parameters['dim']
    targetmean = parameters['targetmean']
    targetvariance_inv = parameters['targetvariance_inv']

    dim_bound = 10000
    if dim >= dim_bound:
        particles_meaned = ne.evaluate('particles-targetmean')
    else: 
        particles_meaned = particles-targetmean
    factor_1 = -(df+dim)/2.
    mat_prod = np.dot(particles_meaned, targetvariance_inv)
    if dim >= dim_bound:
        factor_particles = ne.evaluate('mat_prod*particles_meaned')
    else: 
        factor_particles = mat_prod*particles_meaned
    factor_particles = factor_particles.sum(axis=1)
    factor_2 = (1+(1./df)*factor_particles)[:,np.newaxis]
    factor_3 = (2./df)*mat_prod
    if dim >= dim_bound:
        return ne.evaluate('factor_1*factor_3/factor_2')
    else: 
        return factor_1*factor_3/factor_2

#@profile
def targetgradlogdens_student_old(particles, parameters):
    """
    computes the gradient of the t distribution
    """
    factor_1 = -(parameters['df']+parameters['dim'])/2.
    factor_particles = (particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])*(particles-parameters['targetmean'])
    factor_particles = factor_particles.sum(axis=1)
    factor_2 = 1+(1./parameters['df'])*factor_particles
    factor_3 = (2./parameters['df'])*(particles-parameters['targetmean']).dot(parameters['targetvariance_inv'])
    return(factor_1*factor_3/factor_2[:,np.newaxis])


def targetlogdens_banana(particles, parameters):
    return( -(100*(particles[:, 1::2]**2 - particles[:, ::2])**2 + (particles[:, 1::2] - 1)**2).sum(axis=1))

def targetgradlogdens_banana(particles, parameters):
    uneven = -200*2*particles[:, 1::2]*(particles[:, 1::2]**2 - particles[:, ::2]) + 2*(particles[:, 1::2]-1)
    even = 200*(particles[:, 1::2]**2 - particles[:, ::2])
    gradients = np.zeros(particles.shape)
    gradients[:, 1::2] = uneven
    gradients[:, ::2] = even
    #pdb.set_trace()
    return(gradients)


def targetlogdens_warped(particles, parameters):
    #return( -(100*(particles[:, 1::2]**2 - particles[:, ::2])**2 + (particles[:, 1::2] - 1)**2).sum(axis=1))
    return( -((0.05*particles[:, 1::2]**2 + particles[:, ::2]-5)**2 + 0.01*(particles[:, 1::2])**2).sum(axis=1))

def targetgradlogdens_warped(particles, parameters):
    #uneven = -200*2*particles[:, 1::2]*(particles[:, 1::2]**2 - particles[:, ::2]) + 2*(particles[:, 1::2]-1)
    uneven = -0.05*2*particles[:, 1::2]*(0.05*particles[:, 1::2]**2 + particles[:, ::2]-5) + 0.01*2*(particles[:, 1::2])
    #even = 200*(particles[:, 1::2]**2 - particles[:, ::2])
    even = -0.05*(particles[:, 1::2]**2 + particles[:, ::2]-5)
    gradients = np.zeros(particles.shape)
    gradients[:, 1::2] = uneven
    gradients[:, ::2] = even
    #pdb.set_trace()
    return(gradients)



#@jit(nopython=True)

#@profile
def targetlogdens_logistic_help(particles, X, y):
    """
    likelihood of the logistic regression
    """
    #import ipdb; ipdb.set_trace()
    particles = np.atleast_2d(particles)
    dot_product = np.dot(X, particles.transpose())
    #dot_product_min = np.min(dot_product)
    #dot_prod_less_min = dot_product-dot_product_min
    #sigmoid_value = logplus_one(dot_product)
    #sigmoid_value1 = logexp(dot_product)

    sigmoid_value1 = ne.evaluate('log(1.+exp(-dot_product))')
    #sigmoid_value1 = ne.evaluate('log(1+exp(-dot_prod_less_min))')
    # likelihood_value = (-y*sigmoid_value1 - (1-y)*(dot_product+sigmoid_value1)).sum(axis=0)
    likelihood_value = ((y-1)*dot_product-sigmoid_value1).sum(axis=0)
    #import ipdb; ipdb.set_trace()
    return (likelihood_value-0.5*np.linalg.norm(particles, axis=1)**2)

#@profile
def targetlogdens_logistic_help_old(particles, X, y):
    """
    likelihood of the logistic regression
    """
    #import ipdb; ipdb.set_trace()
    particles = np.atleast_2d(particles)
    dot_product = np.dot(X, particles.transpose())
    #dot_product_min = np.min(dot_product)
    #dot_prod_less_min = dot_product-dot_product_min
    #sigmoid_value = logplus_one(dot_product)
    #sigmoid_value1 = logexp(dot_product)

    #sigmoid_value1 = ne.evaluate('log(1+exp(-dot_product))')
    sigmoid_value1 = np.log(1.+np.exp(-dot_product))
    #sigmoid_value1 = ne.evaluate('log(1+exp(-dot_prod_less_min))')
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


#@jit(nopython=True)
def logexp(x):
    #return ne.evaluate('log(1+exp(x))')
    return np.log(1.+np.exp(-x))


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

#@profile
def targetgradlogdens_logistic_help_old(particles, X, y):
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

def f_dict_logistic_regression(dim, save=False, load_mean_var=False, model_type='logit'):
    if dim == 31:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X_data = data.data
        N_obs = X_data.shape[0]
        X_std = (X_data-X_data.mean(axis=0))/(X_data.var(axis=0)**0.5)
        X_all = np.hstack((np.ones((N_obs,1)), X_std))
        y_all = data.target
        y_all = y_all[:, np.newaxis]
        name_data = "breast_cancer"

    elif dim == 95: # previous 166
        #import ipdb; ipdb.set_trace()
        data = pd.read_csv('./smc_sampler_functions/data/musk/clean1.data', header=None)
        X = data.iloc[:,2:-1]
        
        from sklearn.linear_model import LinearRegression
        lin_model = LinearRegression(normalize=True)
        list_indices = range(X.shape[1])
        list_to_reduce = range(X.shape[1])
        list_rsquares = []
        #import ipdb; ipdb.set_trace()
        if False: # previous version; before comments from james
            for i_index in list_indices: # run over list of indices
                list_to_reduce.remove(i_index) # remove current index
                lin_model.fit(X=X.iloc[:,list_to_reduce], y= X.iloc[:,i_index])
                R_square = lin_model.score(X=X.iloc[:,list_to_reduce], y= X.iloc[:,i_index])
                list_rsquares.append(R_square)
                if R_square>0.99:
                    pass # do nothing and keep a short list
                else: 
                    list_to_reduce.append(i_index) # add the variable back to the list
            ipdb.set_trace()
            assert len(list_to_reduce) == 105
        else: 
            for i_index in list_indices: # run over list of indices
                
                list_to_reduce.remove(i_index) # remove current index
                lin_model.fit(X=X.iloc[:,list_to_reduce], y= X.iloc[:,i_index])
                R_square = lin_model.score(X=X.iloc[:,list_to_reduce], y= X.iloc[:,i_index])
                list_rsquares.append(R_square)
                list_to_reduce.append(i_index) # add the variable back to the list
            #ipdb.set_trace()
            sort_indeces = np.argsort(list_rsquares)
            list_to_reduce = sort_indeces[:94]
            assert len(list_to_reduce) == 94
        # procedure gives a list of lenght 105
        #import ipdb; ipdb.set_trace()
        X_reduced = X.iloc[:,list_to_reduce]
        N_obs = X_reduced.shape[0]
        X_all  = zscore(X_reduced)
        X_all = np.hstack((np.ones((N_obs,1)), X_all))
        y_all = data.iloc[:,-1][:, np.newaxis]
        name_data = "musk"
        #import ipdb; ipdb.set_trace()

    elif dim == 61:
        #import ipdb; ipdb.set_trace()
        data = pd.read_csv('./smc_sampler_functions/data/sonar/sonar.all-data', header=None)
        X = data.iloc[:,:-1]
        N_obs = data.shape[0]
        X_all  = zscore(X)
        X_all = np.hstack((np.ones((N_obs,1)), X_all))
        y_all = (data.iloc[:,-1]=='R')*1.
        y_all = y_all[:, np.newaxis]
        name_data = "sonar"

    elif dim == 295: 
        from german_credit import data_z_2way as data
        N_obs = data.shape[0]

        X_all = np.hstack((np.ones((N_obs,1)), data[:,:-1]))
        y_all = (data[:, -1]+1)/2
        y_all = y_all[:, np.newaxis]
        name_data = "german_credit_interactions"
        if False:
            from sklearn.linear_model import LinearRegression
            lin_model = LinearRegression(normalize=True)
            list_indices = range(X_all.shape[1])
            list_to_reduce = range(X_all.shape[1])
            list_rsquares = []
            if False: 
                for i_index in list_indices: # run over list of indices
                    
                    list_to_reduce.remove(i_index) # remove current index
                    lin_model.fit(X=X_all[:,list_to_reduce], y= X_all[:,i_index])
                    R_square = lin_model.score(X=X_all[:,list_to_reduce], y= X_all[:,i_index])
                    list_rsquares.append(R_square)
                    list_to_reduce.append(i_index) # add the variable back to the list
                sort_indeces = np.argsort(list_rsquares)
                list_to_reduce = sort_indeces[:94]
                import ipdb; ipdb.set_trace()
            if True:
                for i_index in list_indices: # run over list of indices
                    list_to_reduce.remove(i_index) # remove current index
                    lin_model.fit(X=X_all[:,list_to_reduce], y= X_all[:,i_index])
                    R_square = lin_model.score(X=X_all[:,list_to_reduce], y= X_all[:,i_index])
                    list_rsquares.append(R_square)
                    if R_square>0.99:
                        pass # do nothing and keep a short list
                    else: 
                        list_to_reduce.append(i_index) # add the variable back to the list
                import ipdb; ipdb.set_trace()


    elif dim == 25:
        from german_credit import data_z as data
        N_obs = data.shape[0]
        X_all = np.hstack((np.ones((N_obs,1)), data[:,:-1]))
        y_all = (data[:, -1]+1)/2
        y_all = y_all[:, np.newaxis]
        name_data = "german_credit"

    else: 
        np.random.seed(1)
        N = 100
        X_all = np.random.randn(N, dim)
        beta = np.ones(dim)
        proba = 1./(1+np.exp(-np.dot(X_all, beta)))
        #import ipdb as pdb; pdb.set_trace()
        y_all = (proba > np.random.random(N))*1
        y_all = y_all[:, np.newaxis]
        np.random.seed(None)
        name_data = "simulated_dim_"+str(dim)
    parameters = {'X_all': X_all, 'y_all': y_all, 'name_data': name_data}
    if save:
        df = pd.DataFrame(X_all)
        df['target'] = y_all
        df.to_csv(name_data+'.csv', index=False)
    if load_mean_var:
        #import ipdb; ipdb.set_trace()
        #import pdb; pdb.set_trace()
        prior_mean = pd.read_csv('smc_sampler_functions/data/'+name_data+'_mean_'+model_type+'.csv', index_col=0)
        prior_var = pd.read_csv('smc_sampler_functions/data/'+name_data+'_covar_'+model_type+'.csv', index_col=0)
        log_Z = pd.read_csv('smc_sampler_functions/data/'+name_data+'_log_Z_'+model_type+'.csv', index_col=0)

        #prior_mean = pd.read_csv('/'+name_data+'_mean_'+model_type+'.csv', index_col=0)
        #prior_var = pd.read_csv('/'+name_data+'_covar_'+model_type+'.csv', index_col=0)
        #log_Z = pd.read_csv('/'+name_data+'_log_Z_'+model_type+'.csv', index_col=0)
        
        parameters['prior_mean'] = prior_mean.as_matrix()
        parameters['prior_var'] = prior_var
        parameters['prior_inv_var'] = np.linalg.inv(prior_var)
        parameters['prior_log_Z'] = log_Z
        #import ipdb; ipdb.set_trace()
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


#################################################################
################## PROBIT #######################################
#################################################################


def fast_norm_pdf(x):
    y = ne.evaluate('(1./(sqrt(2.*pi)))*exp(-x*x/2.)')
    return y

#@profile
def targetgradlogdens_probit(particles, parameters):
    """
    the gradient of the logdensity of a probit model
    """
    particles = np.atleast_2d(particles)
    y = parameters['y_all']
    X = parameters['X_all']
    X_new = X[:,:,np.newaxis]
    factor_yx = ne.evaluate('y*X')[:,:,np.newaxis]
    dotprod = np.dot(X, particles.transpose())
    factordensity = fast_norm_pdf(dotprod)[:,np.newaxis,:]
    #factorProb_inter = phi(X.dot(particles.transpose()))
    #factorProb = np.clip(factorProb_inter[:,np.newaxis,:], 4e-16, 1-4e-16)
    phi_dot_prod = phi_scipy(dotprod)
    factorProb = np.clip(phi_dot_prod[:,np.newaxis,:], 4e-16, 1-4e-16)
    numerator =  ne.evaluate('(factor_yx - X_new*factorProb)*factordensity')
    #numerator = (factor_yx - X[:,:,np.newaxis]*factorProb)*factordensity
    denominator = ne.evaluate('(1.-factorProb)*factorProb')
    ratio = ne.evaluate('numerator/denominator').sum(axis=0).transpose()
    gradient_pi_0 = -particles
    return ne.evaluate('ratio+gradient_pi_0')

#@profile
def targetgradlogdens_probit_old(particles, parameters):
    """
    the gradient of the logdensity of a probit model
    """
    particles = np.atleast_2d(particles)
    y = parameters['y_all']
    X = parameters['X_all']
    factor_yx = ne.evaluate('y*X')[:,:,np.newaxis]
    factordensity = norm.pdf(X.dot(particles.transpose()))[:,np.newaxis,:]
    #factorProb_inter = phi(X.dot(particles.transpose()))
    #factorProb = np.clip(factorProb_inter[:,np.newaxis,:], 4e-16, 1-4e-16)
    factorProb = np.clip(norm.cdf(X.dot(particles.transpose()))[:,np.newaxis,:], 4e-16, 1-4e-16)
    numerator =  ne.evaluate('factor_yx*factordensity') - X[:,:,np.newaxis]*ne.evaluate('factordensity*factorProb')
    denominator = ne.evaluate('(1.-factorProb)*factorProb')
    gradient_pi_0 = -particles
    return (numerator/denominator).sum(axis=0).transpose()+gradient_pi_0


def targetlogdens_probit_old(particles, parameters):
    """
    the gradient of the logdensity of a probit model
    """
    particles = np.atleast_2d(particles)
    y = parameters['y_all']
    X = parameters['X_all']

    factorProb = norm.cdf(X.dot(particles.transpose()))
    part1 = y*np.log(np.clip(factorProb, 4e-16, 1-4e-16))
    part2 = (1-y)*np.log(1-np.clip(factorProb, 4e-16, 1-4e-16))
    res = (part1+part2).sum(axis=0)-0.5*np.linalg.norm(particles, axis=1)**2
    return res


from math import *
root = np.sqrt(2.0)
one_over_root = 1./root 

@jit(nopython=True)
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    I, J = x.shape
    res = np.zeros(x.shape)
    root= sqrt(2.)
    for i in range(I):
        for j in range(J):
            #res[i,j] = ((1.0 + erf(x[i,j] / root)) / 2.0)
            res[i,j] = erf(x[i,j] / root)
    res = (1.0 + res)/2.0
    return res

#'f8(f8[:,:])', 
#@jit(float64(float64[:,:]), nopython=True, parallel=True)
def erf_handcoded(z):
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    # use Horner's method
    ans = 1 - t * np.exp( -z*z -  1.26551223 + t * ( 1.00002368 +t * ( 0.37409196 + t * ( 0.09678418 + t * (-0.18628806 + t * ( 0.27886807 + t * (-1.13520398 + t * ( 1.48851587 + t * (-0.82215223 + t * ( 0.17087277))))))))))
    is_positive = 2.*(z >= 0.0)-1.
    return ans*is_positive
    #else:
    #    return -ans

#@jit()
def phi_scipy(x):
    # same function but based on scipy special
    res = (1.+serf(x/sqrt(2.)))/2.
    return res

#@jit(nopython=True)
def phi_scipy_new(x):
    # same function but slightly changed
    return 0.5*(1.+serf(x*one_over_root))
    #res = ne.evaluate('(1.+x)/2.')
    #return res

@jit(nopython=True)
def dot_numba(X, particles):
    return np.dot(X, particles.transpose())

@jit(nopython=True)
def sum_numba(part1, part2, particles):
    return (part1+part2).sum(axis=0)-0.5*(particles*particles).sum(axis=1)

#@profile
def core_targetlogdens_probit(particles, y, X):
    """
    speed up version
    """
    #factorProb = norm.cdf(X.dot(particles.transpose()))
    #dotprod = dot_numba(X, particles)
    dotprod = np.dot(X, particles.transpose())
    factorProb = phi_scipy(dotprod)
    clipped_value = np.clip(factorProb, 4e-16, 1-4e-16)
    #part1 = ne.evaluate('y*log(clipped_value)')
    #part2 = ne.evaluate('(1-y)*log(1-clipped_value)')
    part0 = ne.evaluate('y*log(clipped_value)+(1-y)*log(1-clipped_value)')
    #res = (part1+part2).sum(axis=0)-0.5*np.linalg.norm(particles, axis=1)**2
    #res = (part1+part2).sum(axis=0)-0.5*(particles*particles).sum(axis=1)
    #res = sum_numba(part1, part2, particles)
    res = part0.sum(axis=0)-0.5*(particles*particles).sum(axis=1)
    return res

#@profile
def core_targetlogdens_probit_jit(particles, y, X):
    #factorProb = norm.cdf(X.dot(particles.transpose()))
    dotprod = np.dot(X, particles.transpose())
    factorProb = phi(dotprod)
    clipped_value = np.clip(factorProb, 4e-16, 1-4e-16)
    part0 = y*np.log(clipped_value)+(1.-y)*np.log(1.-clipped_value)
    #part1 = y*np.log(clipped_value)
    #part2 = (1-y)*np.log(1-clipped_value)
    #res = (part1+part2).sum(axis=0)-0.5*(particles*particles).sum(axis=1)
    res = part0.sum(axis=0)-0.5*(particles*particles).sum(axis=1)
    return res


def targetlogdens_probit(particles, parameters):
    """
    the gradient of the logdensity of a probit model
    """
    particles = np.atleast_2d(particles)
    y = parameters['y_all']
    X = parameters['X_all']

    return core_targetlogdens_probit(particles, y, X)


if __name__ == '__main__':
    # save the data so that it can be used with R
    f_dict_logistic_regression(5, save=True)
    f_dict_logistic_regression(10, save=True)
    #f_dict_logistic_regression(25, save=True)
    #f_dict_logistic_regression(60, save=True)
    #f_dict_logistic_regression(166, save=True)
    #f_dict_logistic_regression(295, save=True)

    dim = 295
    particles = np.random.normal(size=(1000, dim))
    if False: 
        
        #particles = np.random.normal(size=(1, dim))
        #parameters = {'X_all': X_all, 'y_all': y_all}
        #import ipdb as pdb; pdb.set_trace()
        parameters = f_dict_logistic_regression(dim)
        #particles = np.ones((1,parameters['X_all'].shape[1]))
        particles = np.random.normal(size=(1, parameters['X_all'].shape[1]))
        particles = np.random.normal(size=(1000, parameters['X_all'].shape[1]))
        #import ipdb; ipdb.set_trace()

        #import yappi
        #yappi.start()

        targetlogdens_probit(particles, parameters)
        X = parameters['X_all']
        y = parameters['y_all']

        
        
        # for the student test
        targetmean = np.ones(dim)*2.
        #targetvariance = np.eye(dim)*0.1
        #targetvariance = (0.1*(np.diag(np.linspace(start=0.01, stop=100, num=dim))/float(dim) +0.7*np.ones((dim, dim))))
        targetvariance = (np.diag(np.linspace(start=0.01, stop=100, num=dim)) +0.7*np.ones((dim, dim)))
        #targetvariance = ((np.diag(np.linspace(start=1, stop=2, num=dim)) +0.7*np.ones((dim, dim))))
        targetvariance_inv = np.linalg.inv(targetvariance)
        l_targetvariance_inv = np.linalg.cholesky(targetvariance_inv)
        parameters = {'dim' : dim, 
                    'targetmean': targetmean, 
                    'targetvariance':targetvariance,
                    'det_targetvariance' : np.linalg.det(targetvariance), 
                    'targetvariance_inv':targetvariance_inv,
                    'l_targetvariance_inv':l_targetvariance_inv,
                    'df' : 5
                    }

        for m in range(100):
            targetlogdens_student(particles, parameters)
            targetlogdens_student_old(particles, parameters)
            #targetgradlogdens_logistic_help_old(particles, parameters['X_all'], parameters['y_all'])
            #targetgradlogdens_probit(particles, parameters)
            #targetgradlogdens_probit_old(particles, parameters)
            #core_targetlogdens_probit(particles, y, X)
            #core_targetlogdens_probit_jit(particles, y, X)

        #yappi.get_func_stats().print_all()

    if False:
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