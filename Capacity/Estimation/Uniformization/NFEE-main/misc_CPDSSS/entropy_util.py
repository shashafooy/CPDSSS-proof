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
import theano

import numpy as np
from scipy import stats
from ent_est import entropy
import ml.models.mafs as mafs
import util.io
from ml.trainers import ModelCheckpointer
import ml.step_strategies as ss


dtype = theano.config.floatX

def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return entropy.tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

def create_model(n_inputs, rng, n_hiddens = [200,200],n_mades=14):
    """Generate a multi stage Masked Autoregressive Flow (MAF) model
    George Papamakarios, Theo Pavlakou, and Iain Murray. “Masked Autoregressive Flow for Density Estimation”

    Args:
        n_inputs (_type_): dimension of the input sample
        rng (_type_): type of rng generator to use
        n_hiddens (list, optional): number of hidden layers and hidden nodes per MAF stage. Defaults to [100,100].
        n_mades (int, optional): number of MAF stages. Defaults to 14.

    Returns:
        _type_: MAF model
    """
    n_hiddens=n_hiddens
    act_fun='tanh'


    return mafs.MaskedAutoregressiveFlow(
                n_inputs=n_inputs,
                n_hiddens=n_hiddens,
                act_fun=act_fun,
                n_mades=n_mades,
                input_order='random',
                mode='random',
                rng=rng
            )

def save_model(model,name,path = 'temp_data/saved_models'):    
    parms = [np.empty_like(p.get_value()) for p in model.parms]
    masks = [np.empty_like(m.get_value()) for m in model.masks]
    for i, p in enumerate(model.parms):
        parms[i] = p.get_value().copy()
    for i, m  in enumerate(model.masks):
        masks[i] = m.get_value().copy()
    util.io.save([parms,masks],os.path.join(path,name))

    # util.io.save((model),os.path.join(path,name))

def load_model(n_inputs, name, path = 'temp_data/saved_models'):

    # model = util.io.load(os.path.join(path,name))
    # return model
    model = create_model(n_inputs, rng=np.random)
    try:
        params,masks = util.io.load(os.path.join(path,name))
    except FileNotFoundError:
        return None

    assert len(params) == len(model.parms),'number of parameters is not the same, likely due to different number of stages'
    assert params[0].shape[0] == model.parms[0].get_value().shape[0], f'invalid model input dimension. Expected {model.parms[0].get_value().shape[0]}, got {params[0].shape[0]}'
    assert params[0].shape[1] == model.parms[0].get_value().shape[1], f'invalid model, number of nodes per hidden layer. Expected {model.parms[0].get_value().shape[1]}, got {params[0].shape[1]}'

    for i, p in enumerate(params):
        model.parms[i].set_value(p.astype(dtype))   

    for i, m in enumerate(masks):
        model.masks[i].set_value(m.astype(dtype))

    return model

def update_best_model(model,samples,best_trn_loss,name,path='temp_data/saved_models'):
    """Compare the given model with the saved model {name} located at {path}. If new model has lower training loss, save to given file

    Args:
        model (MaskedAutoregressiveFlow): new model to compare against
        samples (_type_): samples to find the training loss for
        best_trn_loss (_type_): current best training loss
        name (_type_): name of saved model
        path (str, optional): path to the model file. Defaults to 'temp_data/saved_models'.

    Returns:
        _type_: best error
    """
    new_loss = model.eval_trnloss(samples)
    # if best_trn_loss == np.Inf:
    old_model = load_model(samples.shape[1],name,path)
    if old_model is not None:
        best_trn_loss = old_model.eval_trnloss(samples)
    
    print(f"Saved best test loss: {best_trn_loss:.3f}, new model test loss: {new_loss:.3f}")
    if best_trn_loss < new_loss:
        return best_trn_loss
    else:
        save_model(model,name,path)
        return new_loss


def calc_entropy(sim_model,base_samples=None,n_samples=100,k=1,val_tol=0.001,patience=5,method='umtkl',n_hiddens=[200,200,200],n_stages=14):
    """Calculate entropy by uniformizing the data by training a neural network and evaluating the knn entropy on the uniformized points.
    This method does not implement any speed up from threading

    Args:
        sim_model (_type_): dataset model used to generate points from the target distribution
        base_samples (_type_, optional): data samples used in knn. Defaults to None.
        n_samples (int, optional): number of samples used in training. This will be scaled by the dimensionality of the data. Defaults to 100.
        k (int, optional): k value used in knn. Defaults to 1.
        val_tol (float, optional): tolerance level during training to decide if model has improved. Defaults to 0.001.
        patience (int, optional): number of training epochs that must pass with improvement before stopping training. Defaults to 5.
        method (str, optional): knn type to evaluate. types are 'umtkl','umtksg','both'. Defaults to 'umtkl'.
        n_hiddens (list, optional): number of hidden layers and nodes per stage. Defaults to [100,100].
        n_stages (int, optional): number of stages in the model. Defaults to 14.

    Returns:
        _type_: entropy estimate
    """
    
    H = None
    # patience=10
    #redo learning if calc_ent returns error
    while H is None:
        estimator = learn_model(sim_model,n_samples,val_tol,patience,n_hiddens,n_stages)
        H = knn_entropy(estimator,base_samples,k,method)        

        for i in range(3): gc.collect()
    if method=='both':
        return H[0],H[1]
    else:
        return H
    
def calc_entropy_thread(sim_model,n_train,base_samples,reuse=True,method='umtksg'):
    """Train the MAF model, evaluate the uniformizing correction term, and launch the knn algorithm as a thread

    Args:
        sim_model (_type_): model used to generate points from target distribution. Must have method sim()
        n_train (_type_): number of samples used in training. This will be scaled by the dimensionality of the data.
        base_samples (_type_,optional): samples generated from the target distribution to be used in knn. Default None
        method (str, optional): type of knn metric to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.


    Returns:
        (thread,numpy): return started thread handle used for calculating entropy and the associated entropy correction term
    """

    estimator = learn_model(sim_model, n_train, train_samples= base_samples if reuse else None)
    # estimator.samples=base_samples
    uniform,correction = estimator.uniform_correction(base_samples)
    thread = estimator.start_knn_thread(uniform,method=method)
    return thread,correction,estimator

def learn_model(sim_model, n_samples=100,train_samples = None,val_tol=0.0005,patience=5,n_hiddens=[200,200,200],n_stages=14, mini_batch=256, fine_tune=True, step=ss.Adam()):
    """Create a MAF model and train it with the given parameters

    Args:
        sim_model (_type_): model to generate points from target distribution
        n_samples (int, optional): number of samples to train on. Scaled by sim_model dimension. Defaults to 100.
        val_tol (float, optional): validation tolerance threshold to decide if model has improved. Defaults to 0.001.
        patience (int, optional): number of epochs without improvement before exiting training. Defaults to 5.
        n_hiddens (list, optional): number of hidden layers and nodes in a list. Defaults to [200,200].
        n_stages (int, optional): number of MAF stages. Defaults to 14.
        mini_batch (int, optional): Batch size for training. Defaults to 1024
        fine_tune (bool, optional): Set to True to run training twice, first with large step size, then a smaller step size. Defaults to True.

    Returns:
        entropy.UMestimator: estimator object used for training and entropy calculation
    """
    
    net=create_model(sim_model.x_dim, rng=np.random,n_hiddens=n_hiddens,n_mades=n_stages)
    estimator = entropy.UMestimator(sim_model,net,train_samples)
    start_time = time.time()
    # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
    estimator.learn_transformation(
        n_samples = int(n_samples*sim_model.x_dim),
        val_tol=val_tol,
        patience=patience, 
        fine_tune=fine_tune,
        minibatch=mini_batch,
        step=step
        )
    end_time = time.time()        
    print("learning time: ",str(timedelta(seconds = int(end_time - start_time))))

    return estimator

def knn_entropy(estimator: entropy.UMestimator,base_samples=None,k=1,method='umtksg'):
    """Wrapper function to time knn entropy calculation from the given estimator
    Does not use any threading for speed up

    Args:
        estimator (entropy.UMestimator): estimator containing trained model
        base_samples (_type_, optional): distribution samples to be used for knn. Defaults to None.
        k (int, optional): k neighbors value for knn. Defaults to 1.
        method (str, optional): type of knn metric to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.

    Returns:
        _type_: Entropy value. If method='both' is used, then return tuple with entropy using KL and KSG
    """
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


    
