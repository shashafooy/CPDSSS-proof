"""
Functions for generating the model and entropy
"""
from datetime import timedelta
import gc
import os
import time

import numpy as np
from scipy import stats
from ent_est import entropy


def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return entropy.tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

def create_model(n_inputs, rng):
    n_hiddens=[100,100]
    act_fun='tanh'
    n_mades=10

    import ml.models.mafs as mafs

    return mafs.MaskedAutoregressiveFlow(
                n_inputs=n_inputs,
                n_hiddens=n_hiddens,
                act_fun=act_fun,
                n_mades=n_mades,
                mode='random',
                rng=rng
            )

def calc_entropy(sim_model,base_samples=None,n_samples=100,val_tol=0.05):
    H=-1
    patience=10
    #redo learning if calc_ent returns error
    while H==-1:
        net=create_model(sim_model.x_dim, rng=np.random)
        estimator = entropy.UMestimator(sim_model,net)
        start_time = time.time()
        # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
        estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim),val_tol=val_tol,patience=patience)
        end_time = time.time()        
        print("learning time: ",str(timedelta(seconds = int(end_time - start_time))))
        estimator.samples = estimator.samples if base_samples is None else base_samples
        reuse = False if base_samples is None else True
        start_time = time.time()
        H,_,_,_ = estimator.calc_ent(reuse_samples=reuse, method='umtkl',k=1)
        end_time = time.time()        
        print("knn time: ",str(timedelta(seconds = int(end_time - start_time))))       

        net.release_shared_data()
        for i in range(3): gc.collect()
        
    return H

def update_filename(path,old_name,n_samples,today,iter,rename=True):
    new_name ="CPDSSS_data_dump({0}_iter)({1}k_samples)({2})".format(iter,int(n_samples/1000),today)
    unique_name=new_name
    #Check if name already exists, append number to end until we obtain new name
    i=1
    while os.path.isfile(os.path.join(path,unique_name + '.pkl')):
        unique_name = new_name + '_' + str(i)        
        i=i+1
    new_name=unique_name 
    if(rename):
        os.rename(os.path.join(path,old_name + '.pkl'),os.path.join(path,new_name + '.pkl'))
    return new_name

def print_border(msg):
    print("-"*len(msg) + "\n" + msg + "\n" + "-"*len(msg))



