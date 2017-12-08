# functions for plotting
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_results_single_simulation(results_list):
    """
    plot the results of a single run
    """

    plt.figure(figsize=(20,10))

    # plot acceptance rate kernel
    plt.subplot(331)
    for results_sampler in results_list:
        plt.plot(results_sampler['temp_list'], results_sampler['acceptance_rate_list'], label=results_sampler['proposal_kernel']['proposalname'])
    plt.title("acceptance rate mcmc")
    plt.legend()

    # plot ESS
    plt.subplot(332)
    for results_sampler in results_list:
        plt.plot(results_sampler['temp_list'], results_sampler['ESS_list'], label=results_sampler['proposal_kernel']['proposalname'])
    plt.ylim(ymax = 1.1, ymin = 0.)
    plt.title("ESS")
    plt.legend()

    # plot resampled particles
    plt.subplot(333)
    for results_sampler in results_list:
        plt.scatter(results_sampler['particles_resampled'][:,0], results_sampler['particles_resampled'][:,1], label=results_sampler['proposal_kernel']['proposalname'], alpha=0.2)
    try: plt.scatter(results_sampler['parameters']['targetmean'][0], results_sampler['parameters']['targetmean'][1], color="r")
    except: pass
    plt.title("particles")
    plt.legend()

    # plot estimated normalization constant
    plt.subplot(334)
    plt.title("normalization constant")
    for results_sampler in results_list:
        plt.plot(results_sampler['temp_list'], np.cumsum(results_sampler['Z_list']), label=results_sampler['proposal_kernel']['proposalname'])
    plt.axhline(0) # normalization constant is zero
    plt.legend()

    plt.subplot(335)
    plt.title("distribution first component")
    for results_sampler in results_list:
        sns.distplot(results_sampler['particles_resampled'][:,0], label=results_sampler['proposal_kernel']['proposalname'])
    plt.legend()
    
    plt.subplot(336)
    plt.title("distribution second component")
    for results_sampler in results_list:
        sns.distplot(results_sampler['particles_resampled'][:,1], label=results_sampler['proposal_kernel']['proposalname'])
    plt.legend()

    # plot diagnostics of kernel
    plt.subplot(337)
    plt.title("temp vs epsilon")
    for results_sampler in results_list:
        epsilons = np.array([iteration['epsilon'] for iteration in results_sampler['perf_list']])
        temp = np.array(results_sampler['temp_list'])
        epsilons = epsilons.mean(axis=1).flatten()
        plt.plot(temp, epsilons, label=results_sampler['proposal_kernel']['proposalname'])
    plt.legend()

    plt.subplot(338)
    plt.title("temp vs esjd")
    for results_sampler in results_list:
        ESJD = np.array([iteration['squarejumpdist_realized'] for iteration in results_sampler['perf_list']]).mean(axis=1)
        temp = np.array(results_sampler['temp_list'])
        plt.plot(temp, ESJD, label=results_sampler['proposal_kernel']['proposalname'])
    plt.legend()
    #import ipdb; ipdb.set_trace() 
    plt.savefig('diagnosis_single_simulation_dim_%s_model_%s.png'%(results_list[0]['particles_resampled'].shape[1], results_list[0]['target_name']))
    
    plt.show()
    plt.clf()


def plot_repeated_simulations(results_dict):
    """
    plot the results of the repeated simulations
    """
    norm_constant_list = results_dict['norm_const'] 
    mean_array = results_dict['mean_array']
    var_array = results_dict['var_array'] 
    names_samplers = results_dict['names_samplers']

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    #plt.subplot(221)
    ax1.boxplot(norm_constant_list.transpose(), labels=names_samplers)
    ax1.set_title('normalization constant')

    #plt.subplot(222)
    ax2.boxplot(mean_array.transpose(), labels=names_samplers)
    ax2.set_title('mean first component')

    #plt.subplot(223)
    ax3.boxplot(var_array.transpose(), labels=names_samplers)
    ax3.set_title('var first component')
    #import ipdb; ipdb.set_trace()

    if results_dict['target_name'] == 'normal' or results_dict['target_name'] == 'student':
        # results_dict['particles_array'] = [N_particles, dim, i_sampler, M_repetition]
        mse_list = []
        for i, sampler in enumerate(names_samplers):
            mean_iteration = results_dict['particles_array'][:,:,i,:].mean(axis=0) 
            meaned_values_mean = mean_iteration - results_dict['parameters']['targetmean'][:,np.newaxis]

            mse_mean_iteration = np.mean(np.linalg.norm(meaned_values_mean, axis=0)**2)
            log_mse_mean_iteration = np.log(mse_mean_iteration)

            var_iteration = results_dict['particles_array'][:,:,i,:].var(axis=0) 
            meaned_values_var = var_iteration - np.diag(results_dict['parameters']['targetvariance'])[:,np.newaxis]

            mse_var_iteration = np.mean(np.linalg.norm(meaned_values_var, axis=0)**2)
            log_mse_var_iteration = np.log(mse_var_iteration)

            Z_var = np.log(np.mean(norm_constant_list[i,:]**2))

            mse_list.append([np.round(log_mse_mean_iteration, decimals=3), np.round(log_mse_var_iteration, decimals=3), np.round(Z_var, decimals=3)])
            
        rows = names_samplers
        columns = ['log MSE mean', 'log MSE var', 'log MSE Z']



    else: 
        mse_list = []
        for i, sampler in enumerate(names_samplers):
            mean_iteration = results_dict['particles_array'][:,:,i,:].mean(axis=0).sum(axis=1).var()
            log_mse_mean_iteration = np.log(mean_iteration)
            
            var_iteration = results_dict['particles_array'][:,:,i,:].var(axis=0).sum(axis=1).var()
            log_mse_var_iteration = np.log(var_iteration)
            Z_var = np.log(np.var(norm_constant_list[i,:]))

            mse_list.append([np.round(log_mse_mean_iteration, decimals=3), np.round(log_mse_var_iteration, decimals=3), np.round(Z_var, decimals=3)])
        
        rows = names_samplers
        columns = ['log var mean', 'log var var', 'log var Z']

    ax4.axis('off')
    ax4.table(cellText=mse_list,
                      rowLabels=rows,
                      colLabels=columns, 
                      loc='center')
    #import pdb; pdb.set_trace()
    plt.savefig('repeated_simulation_%s_dim_%s.png' %(results_dict['target_name'], results_dict['parameters']['dim']))
    plt.show()
    plt.clf()


