"""
Functions for generating the model and entropy
"""
from datetime import timedelta
import gc
import os
import time
import re

import numpy as np
from scipy import stats
from ent_est import entropy


def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return entropy.tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

def create_model(n_inputs, rng, n_hiddens = [100,100]):
    n_hiddens=n_hiddens
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

def calc_entropy(sim_model,base_samples=None,n_samples=100,k=1,val_tol=0.05,patience=10,method='umtkl',n_hiddens=[100,100]):
    H=-1
    # patience=10
    #redo learning if calc_ent returns error
    while H==-1:
        net=create_model(sim_model.x_dim, rng=np.random,n_hiddens=n_hiddens)
        estimator = entropy.UMestimator(sim_model,net)
        start_time = time.time()
        # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
        estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim),val_tol=val_tol,patience=patience)
        end_time = time.time()        
        print("learning time: ",str(timedelta(seconds = int(end_time - start_time))))
        estimator.samples = estimator.samples if base_samples is None else base_samples
        reuse = False if base_samples is None else True
        start_time = time.time()
        H,H2,_,_ = estimator.calc_ent(reuse_samples=reuse, method=method,k=k)
        end_time = time.time()        
        print("knn time: ",str(timedelta(seconds = int(end_time - start_time))))       

        net.release_shared_data()
        for i in range(3): gc.collect()
    if method=='both':
        return H,H2
    else:
        return H

def update_filename(path,old_name,iter,rename=True):
    reg_pattern = r"\(\d{1,3}_iter\)"
    iter_name = "({}_iter)".format(iter)
    match = re.search(reg_pattern,old_name)
    # Replace iteration number, append if it doesn't exist
    if match:
        new_name = re.sub(reg_pattern,iter_name,old_name)    
    else:
        new_name = old_name + iter_name

    # Attach unique pid to filename
    if str(os.getpid()) not in new_name:
        new_name = new_name + "_" + str(os.getpid())


    # WHILE LOOP SHOULD NOT RUN
    unique_name=new_name
    #Check if name already exists, append number to end until we obtain new name
    i=0
    while os.path.isfile(os.path.join(path,unique_name + '.pkl')):
        unique_name = new_name + '_' + str(i)        
        i=i+1
    #create file if it doesn't exists.
    #Sometimes had race condition of two programs used the same name because 
    #   new name file wasn't used for a few more clock cycles
    #   creating the file right after getting unique name should prevent this
    os.makedirs(path,exist_ok=True)
    open(os.path.join(path,unique_name + '.pkl'),'a').close()
    new_name=unique_name 
    if(rename):
        os.rename(os.path.join(path,old_name + '.pkl'),os.path.join(path,new_name + '.pkl'))
    return new_name

def print_border(msg):
    print("-"*len(msg) + "\n" + msg + "\n" + "-"*len(msg))


def time_exec(func,print_time=True):
    """Time how long the given function takes

    Args:
        func (Lambda): Lambda function with the given code that will run. e.g. lambda: myfunc(x,y)
        print_time (Bool): Set True to print the total time
    """
    start_time = time.time()
    result = func()
    end_time = time.time()
    tot_time = end_time - start_time
    print(f"Elapsed Time: {tot_time:.4f} sec")
    return result, tot_time
    
    
