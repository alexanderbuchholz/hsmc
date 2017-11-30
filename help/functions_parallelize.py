# function that allows the equally balances parallelization

from multiprocessing import Process, Pipe
from itertools import izip
import multiprocessing

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    results_list = []
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    #import ipdb; ipdb.set_trace()
    #import ipdb; ipdb.set_trace()
    [p.start() for p in proc]
    for (p,c) in pipe:
        results_list.append(p.recv())
    [p.join() for p in proc]
    #[p.terminate() for p in proc]
    
    return results_list

def parallelize_partial_over_chunks(partial_parallel_sampler, list_repetitions):
    NUM_CORES = multiprocessing.cpu_count()
    #pdb.set_trace()
    results_list_intra_sampler = []
    chunks = [list_repetitions[i:i + NUM_CORES] for i in range(0, len(list_repetitions), NUM_CORES)] 
    for chunk_single in chunks:
        F_results = parmap(partial_parallel_sampler, chunk_single)
        results_list_intra_sampler.append(F_results)
    # unlists the results and puts them in one list
    results_list = [item for sublist in results_list_intra_sampler for item in sublist]
    return results_list
