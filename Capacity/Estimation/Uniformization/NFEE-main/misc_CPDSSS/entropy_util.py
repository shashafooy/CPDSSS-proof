"""
Functions for generating the model and entropy
"""
from datetime import timedelta
import gc
import os
from threading import Thread
import time
import re
import multiprocessing as mp

import numpy as np
from scipy import stats
from ent_est import entropy


def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return entropy.tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

def create_model(n_inputs, rng, n_hiddens = [100,100],n_mades=10):
    n_hiddens=n_hiddens
    act_fun='tanh'

    import ml.models.mafs as mafs

    return mafs.MaskedAutoregressiveFlow(
                n_inputs=n_inputs,
                n_hiddens=n_hiddens,
                act_fun=act_fun,
                n_mades=n_mades,
                input_order='random',
                mode='random',
                rng=rng
            )

def calc_entropy(sim_model,base_samples=None,n_samples=100,k=1,val_tol=0.05,patience=10,method='umtkl',n_hiddens=[100,100],n_stages=10):
    H = None
    # patience=10
    #redo learning if calc_ent returns error
    while H is None:
        estimator = learn_model(sim_model,n_samples,val_tol,patience,n_hiddens,n_stages)
        H = knn_entropy(estimator,base_samples,k,method)        


        # net=create_model(sim_model.x_dim, rng=np.random,n_hiddens=n_hiddens)
        # estimator = entropy.UMestimator(sim_model,net)
        # start_time = time.time()
        # # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
        # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim),val_tol=val_tol,patience=patience)
        # end_time = time.time()        
        # print("learning time: ",str(timedelta(seconds = int(end_time - start_time))))
        # estimator.samples = estimator.samples if base_samples is None else base_samples
        # reuse = False if base_samples is None else True
        # start_time = time.time()
        # H,H2 = estimator.calc_ent(reuse_samples=reuse, method=method,k=k)
        # end_time = time.time()        
        # print("knn time: ",str(timedelta(seconds = int(end_time - start_time))))       

        # net.release_shared_data()
        for i in range(3): gc.collect()
    if method=='both':
        return H[0],H[1]
    else:
        return H

def learn_model(sim_model,n_samples=100,val_tol=0.01,patience=10,n_hiddens=[100,100],n_stages=10):
    net=create_model(sim_model.x_dim, rng=np.random,n_hiddens=n_hiddens,n_mades=n_stages)
    estimator = entropy.UMestimator(sim_model,net)
    start_time = time.time()
    # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
    estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim),val_tol=val_tol,patience=patience)
    end_time = time.time()        
    print("learning time: ",str(timedelta(seconds = int(end_time - start_time))))

    return estimator

def knn_entropy(estimator: entropy.UMestimator,base_samples=None,k=1,method='umtkl'):
    import theano
    estimator.samples = estimator.samples if base_samples is None else base_samples
    reuse = False if base_samples is None else True
    start_time = time.time()
    H,H2 = estimator.calc_ent(reuse_samples=reuse, method=method,k=k)
    end_time = time.time()        
    print("knn time: ",str(timedelta(seconds = int(end_time - start_time))))    
    if method=='both':
        return H,H2
    else:
        return H


def thread_exp():
    '''Experiment to try running a theano function in a thread'''
    import theano   
    import simulators.CPDSSS_models as mod

    net=create_model(4, rng=np.random,n_hiddens=[100,100],n_mades=10)
    samples = mod.Laplace(0,2,4).sim(1000)
    return net.calc_random_numbers(samples)

    
