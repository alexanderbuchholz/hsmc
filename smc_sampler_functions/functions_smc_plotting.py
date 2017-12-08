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

    plt.figure()
    plt.subplot(221)
    plt.boxplot(norm_constant_list.transpose(), labels=names_samplers)
    plt.title('normalization constant')

    plt.subplot(222)
    plt.boxplot(mean_array.transpose(), labels=names_samplers)
    plt.title('mean first component')

    plt.subplot(223)
    plt.boxplot(var_array.transpose(), labels=names_samplers)
    plt.title('var first component')
    import ipdb; ipdb.set_trace()
    plt.savefig('repeated_simulation_%s_dim_%s.png' %(results_dict['target_name'], results_dict['parameters']['dim']))
    plt.show()
    plt.clf()


