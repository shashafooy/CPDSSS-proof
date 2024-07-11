import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy import stats
import scipy.special as spl

import sys
import configparser

import ml.trainers as trainers
import ml.step_strategies as ss

from misc_CPDSSS.util import BackgroundThread



def tkl_tksg(y, n=None, k=1, max_k=None, algorithm = 'auto', shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    N, dim = y.shape
    if isinstance(k,list):
        max_k=k[-1]
        k_range = range(1,max_k+1)
    else:
        max_k=k
        k_range = range(k,k+1)
    # k=k if max_k is None else max_k
    
    config = configparser.ConfigParser()
    if config.read('CPDSSS.ini') != []:
        n_jobs = int(config['GLOBAL']['knn_cores']) #number of cores for knn, negative value uses all cores but n+1
    else: 
        n_jobs = 1
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    #Auto algorithm switches to brute after dim=16. For truncated, better to swap at dim=30
    algorithm = 'brute' if dim>30 else 'kd_tree'


    # knn search
    nbrs = NearestNeighbors(n_neighbors=max_k+1, algorithm=algorithm, metric='chebyshev',n_jobs=n_jobs).fit(y)
    dist, idx = nbrs.kneighbors(y)

    # k_range = range(1,max_k+1) if max_k is not None else range(k,k+1)

    h_kl=np.zeros(len(k_range))
    h_ksg=np.zeros(len(k_range))
    for i,k in enumerate(k_range):
        # truncated KL
        zeros_mask = dist[:,k]!=0
        
        r = dist[:,k]
        r = np.tile(r[:, np.newaxis], (1, dim))
        lb = (y-r >= 0)*(y-r) + (y-r < 0)*0
        ub = (y+r <= 1)*(y+r) + (y+r > 1)*1

        zeta = (ub-lb)[zeros_mask] #remove zeros, duplicate points result in 0 distance
        N=zeta.shape[0]
        hh = np.log(np.prod(zeta, axis=1))
            
        h_kl[i] = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)


        # truncated KSG
        # epsilons = np.abs(y-y[idx[:,k]])
        y_dup = np.tile(y[:,np.newaxis,:],(1,k,1)) #duplicate last axis to add k dimension
        epsilons = np.max(np.abs(y_dup-y[idx[:,1:k+1]]),axis=1)
        zeta2 = np.minimum(y+epsilons,1) - np.maximum(y-epsilons,0)
        #remove zeros, invalid data. Zeros occur if points along a dimension are exactly the same
        zeros_mask = ~np.any(zeta2==0,axis=1)
        zeta2=zeta2[zeros_mask]
        hh2=np.sum(np.log(zeta2),axis=1)
        N=zeta2.shape[0]

        # hh3 = np.zeros(n)
        # for j in range(n):
        #     r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
        #     hh3[j] = np.log(np.prod(2*r))

            
        h_ksg[i] = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh2)

    #If we only used 1 k value, return that value, not an array
    (h_kl,h_ksg)=(h_kl,h_ksg) if len(k_range)>1 else (h_kl[0],h_ksg[0])
    
    return h_kl,h_ksg

def kl_ksg(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    y = np.asarray(y, float)
    
    if isinstance(k,list):
        max_k=k[-1]
        k_range=range(1,max_k+1)
    else:
        max_k=k
        k_range=range(k,k+1)

    config = configparser.ConfigParser()
    if config.read('CPDSSS.ini') != []:
        n_jobs = int(config['GLOBAL']['knn_cores']) #number of cores for knn, negative value uses all cores but n+1
    else: 
        n_jobs = 1


    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y/y_std
        
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    # print("starting distance search")
    nbrs = NearestNeighbors(n_neighbors=max_k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)

    h_kl=np.empty(len(k_range))
    h_ksg=np.empty(len(k_range))
    for i,k in enumerate(k_range):

        # KL
        zeros_mask = dist[:,k]!=0
        dist = dist[zeros_mask,:]
        

        if standardize == True:
            hh = dim*np.log(2*dist[:,k])+np.sum(np.log(y_std))
        else:
            hh = dim*np.log(2*dist[:,k])
        N=hh.shape[0]    
        h_kl[i] = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    

        # KSG
        # epsilons=np.abs(y-y[idx[:,k]])
        y_dup = np.tile(y[:,np.newaxis,:],(1,k,1)) #duplicate last axis to add k dimension
        epsilons = np.max(np.abs(y_dup-y[idx[:,1:k+1]]),axis=1)
        zeros_mask = ~np.any(epsilons==0,axis=1)
        epsilons=epsilons[zeros_mask]
        if standardize == True:        
            hh=np.sum(np.log(2*epsilons*y_std),axis=1)            
        else:
            hh=np.sum(np.log(2*epsilons),axis=1)
        N=hh.shape[0]
        h_ksg[i] = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh)
    
    h_kl,h_ksg=(h_kl,h_ksg) if len(k_range)>1 else (h_kl[0],h_ksg[0])

    # print("finished knn")
    return h_kl,h_ksg



def kl(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y/y_std
        
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    zeros_mask = dist[:,k]!=0
    dist = dist[zeros_mask,:]
    N=dist.shape[0]

    if standardize == True:
        hh = dim*np.log(2*dist[:,k])+np.sum(np.log(y_std))
    else:
        hh = dim*np.log(2*dist[:,k])
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)

    return h
    

def tkl(y, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    config = configparser.ConfigParser()
    if config.read('CPDSSS.ini') != []:
        n_jobs = int(config['GLOBAL']['knn_cores']) #number of cores for knn, negative value uses all cores but n+1
    else: 
        n_jobs = 1
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)

    #Auto algorithm switches to brute after dim=16. For truncated, better to swap at dim=30
    algorithm = 'brute' if dim>30 else 'kd_tree'

    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algorithm, metric='chebyshev',n_jobs=n_jobs).fit(y)
    dist, idx = nbrs.kneighbors(y)

    zeros_mask = dist[:,k]!=0
    
    r = dist[:,k]
    r = np.tile(r[:, np.newaxis], (1, dim))
    lb = (y-r >= 0)*(y-r) + (y-r < 0)*0
    ub = (y+r <= 1)*(y+r) + (y+r > 1)*1

    zeta = (ub-lb)[zeros_mask] #remove zeros, duplicate points result in 0 distance
    N=zeta.shape[0]
    hh = np.log(np.prod(zeta, axis=1))
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    
    return h


def mi_kl(y, dim_x, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    #standardize
    y = y/np.std(y, axis=0)
    
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    for i in range(n):
        n_x[i] = np.sum(dist_1[i,1:] < dist[i,k])
        n_y[i] = np.sum(dist_2[i,1:] < dist[i,k])
    
    mi = spl.digamma(k)-np.mean(spl.digamma(n_x+1)+spl.digamma(n_y+1))+spl.digamma(N)
    
    return mi



def ksg(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004 
    """
    
    y = np.asarray(y, float)
    
    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y/y_std
        
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    # hh = np.empty(n)
    # epsilons=np.abs(y-y[idx[:,k]])
    y_dup = np.tile(y[:,np.newaxis,:],(1,k,1)) #duplicate last axis to add k dimension
    epsilons = np.max(np.abs(y_dup-y[idx[:,1:k+1]]),axis=1)
    zeros_mask = ~np.any(epsilons==0,axis=1)
    epsilons=epsilons[zeros_mask]
    if standardize == True:        
        hh=np.sum(np.log(2*epsilons*y_std),axis=1)
        # for j in range(n):
        #     r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
        #     hh[j] = np.log(np.prod(2*r*y_std))
            
    else:
        hh=np.sum(np.log(2*epsilons),axis=1)
        # for j in range(n):
        #     r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
        #     hh[j] = np.log(np.prod(2*r))
    
    N=hh.shape[0]
    h = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh)
    
    return h


def tksg(y, n=None, k=1, shuffle=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004 
    """
    config = configparser.ConfigParser()
    if config.read('CPDSSS.ini') != []:
        n_jobs = int(config['GLOBAL']['knn_cores']) #number of cores for knn, negative value uses all cores but n+1
    else: 
        n_jobs = 1
    y = np.asarray(y, float)
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    #Auto algorithm switches to brute after dim=16. For truncated, better to swap at dim=30
    algorithm = 'brute' if dim>30 else 'kd_tree'

    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm=algorithm, metric='chebyshev',n_jobs=n_jobs).fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    # epsilons = np.abs(y-y[idx[:,k]])
    y_dup = np.tile(y[:,np.newaxis,:],(1,k,1)) #duplicate last axis to add k dimension
    epsilons = np.max(np.abs(y_dup-y[idx[:,1:k+1]]),axis=1)    
    zeta = np.minimum(y+epsilons,1) - np.maximum(y-epsilons,0)
    #remove zeros, invalid data. Zeros occur if points along a dimension are exactly the same
    zeros_mask = ~np.any(zeta==0,axis=1)
    zeta=zeta[zeros_mask]
    hh2=np.sum(np.log(zeta),axis=1)
    N=zeta.shape[0]

    # hh = np.empty(n)
    # for j in range(n):
    #     # r=dist[n,k]
    #     r = np.max(np.abs(y[j]-y[idx[j,1:k+1]]), axis=0)
    #     lb = (y[j]-r >=0)*(y[j]-r) + (y[j]-r < 0)*0
    #     ub = (y[j]+r <=1)*(y[j]+r) + (y[j]+r > 1)*1
    #     hh[j] = np.sum(np.log(ub-lb))
        
    h = -spl.digamma(k)+spl.digamma(N)+(dim-1)/k+np.mean(hh2)
    
    return h


def mi_ksg(y, dim_x, n=None, k=1, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    
    #standardize
    y = y/np.std(y, axis=0)
    
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    for i in range(n):
        r_1 = np.max(np.abs(y[i,:dim_x]-y[idx[i,1:k+1],:dim_x]))
        r_2 = np.max(np.abs(y[i,dim_x:]-y[idx[i,1:k+1],dim_x:]))
        n_x[i] = np.sum(dist_1[i,1:] <= r_1)
        n_y[i] = np.sum(dist_2[i,1:] <= r_2)
    
    mi = spl.digamma(k)-1/k-np.mean(spl.digamma(n_x)+spl.digamma(n_y))+spl.digamma(N)
    
    return mi



def lnc(y, n=None, k=1, alpha=None, shuffle=True, rng=np.random):
    """
    Implements the Local Nonuniformity Correction (LNC) estimator in
    Shuyang Gao, Greg Ver Steeg, and Aram Galstyan, "Efficient Estimation of Mutual Information for
    Strongly Dependent Variables", AISTATS, 2015 
    """
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    # Determine alpha            
    if alpha is None:
        alpha = est_alpha_for_lnc(dim, k)
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    hh = np.empty(n)
    for j in range(n):
        y_loc = y[idx[j,:k+1]]
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV = dim*np.log(2*l_edge)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV_loc = np.log(np.prod(2*l_edge*np.sqrt(pca.explained_variance_)))
        if logV_loc-logV < np.log(alpha):
            hh[j] = logV_loc
        else:
            hh[j] = logV
        
    h = -spl.digamma(k)+spl.digamma(N)+np.mean(hh)
    
    return h


def mi_lnc(y, dim_x, n=None, k=1, alpha=None, shuffle=True, rng=np.random):
    
    y = np.asarray(y, float)
    N, dim = y.shape
    
    if n is None:
        n = N
    else:
        n = min(n, N)
    
    # permute y
    if shuffle is True:
        rng.shuffle(y)
    
    x1 = y[:, :dim_x]
    x2 = y[:, dim_x:]
    # knn search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(y)
    dist, idx = nbrs.kneighbors(y)
    
    nbrs_1 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x1)
    dist_1, idx_1 = nbrs_1.kneighbors(x1)
    
    nbrs_2 = NearestNeighbors(n_neighbors=N, algorithm='auto', metric='chebyshev').fit(x2)
    dist_2, idx_2 = nbrs_2.kneighbors(x2)
    
    n_x = np.empty(n)
    n_y = np.empty(n)
    LNC = np.empty(n)
    for i in range(n):
        y_loc = y[idx[i,:k+1]]
        r_1 = np.max(np.abs(y[i,:dim_x]-y[idx[i,1:k+1],:dim_x]))
        r_2 = np.max(np.abs(y[i,dim_x:]-y[idx[i,1:k+1],dim_x:]))
        logV = dim_x*np.log(2*r_1)+(dim-dim_x)*np.log(2*r_2)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(np.abs(y_loc[1:k+1]-y_loc[0]))
        logV_loc = np.log(np.prod(2*l_edge*np.sqrt(pca.explained_variance_)))
        if logV_loc-logV < np.log(alpha):
            LNC[i] = logV_loc-logV
        else:
            LNC[i] = 0.0
        
        n_x[i] = np.sum(dist_1[i,:] <= r_1)
        n_y[i] = np.sum(dist_2[i,:] <= r_2)
    
    mi = spl.digamma(k)-1/k-np.mean(spl.digamma(n_x)+spl.digamma(n_y))+spl.digamma(N)-np.mean(LNC)
    
    return mi


def est_alpha_for_lnc(dim, k, N=5e5, eps=5e-3, rng=np.random):
    N = int(N)
    a = np.empty(N)
    for i in range(N):
        y_loc = rng.rand(k, dim)
        pca = PCA(n_components=dim, whiten = True)
        pca.fit(y_loc)
        y_loc = pca.transform(y_loc)
        l_edge = np.max(y_loc)
        V_tilde = np.prod(2*l_edge*np.sqrt(pca.explained_variance_))
        a[i] = V_tilde/1
    return np.sort(a)[int(eps*N)]
    
    
    
def learn_density(model, xs, ws=None, regularizer=None, val_frac=0.05, step=ss.Adam(a=1.e-4), minibatch=100, patience=20, monitor_every=1, logger=sys.stdout, rng=np.random, val_tol=None, target=None, show_progress=False):
    """    Train model to learn the density p(x).


    Args:
        model (_type_): model to train
        xs (_type_): samples to train on
        ws (_type_, optional): weights. Defaults to None.
        regularizer (_type_, optional): _description_. Defaults to None.
        val_frac (float, optional): _description_. Defaults to 0.05.
        step (_type_, optional): _description_. Defaults to ss.Adam(a=1.e-4).
        minibatch (int, optional): _description_. Defaults to 100.
        patience (int, optional): Epochs to try after best validation case. Defaults to 20.
        monitor_every (int, optional): _description_. Defaults to 1.
        logger (_type_, optional): _description_. Defaults to sys.stdout.
        rng (_type_, optional): _description_. Defaults to np.random.
        val_tol (_type_, optional): Tolerance if validation loss has improved. Defaults to None.
        target (_type_,optional): Target optimal validation value. Defaults to None

    Returns:
        _type_: Trained model
    """

    xs = np.asarray(xs, np.float32)

    n_data = xs.shape[0]

    # shuffle data, so that training and validation sets come from the same distribution
    idx = rng.permutation(n_data)
    xs = xs[idx]

    # split data into training and validation sets
    n_trn = int(n_data - val_frac * n_data)
    xs_trn, xs_val = xs[:n_trn], xs[n_trn:]

    if ws is None:

        # train model without weights
        trainer = trainers.SGD(
            model=model,
            trn_data=[xs_trn],
            trn_loss=model.trn_loss if regularizer is None else model.trn_loss + regularizer,
            val_data=[xs_val],
            val_loss=model.trn_loss,
            step=step,
            val_target = target
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger,
            val_Tol=val_tol,
            show_progress=show_progress
        )

    else:

        # prepare weights
        ws = np.asarray(ws, np.float32)
        assert ws.size == n_data, 'wrong sizes'
        ws = ws[idx]
        ws_trn, ws_val = ws[:n_trn], ws[n_trn:]

        # train model with weights
        trainer = trainers.WeightedSGD(
            model=model,
            trn_data=[xs_trn],
            trn_losses=-model.L,
            trn_weights=ws_trn,
            trn_reg=regularizer,
            val_data=[xs_val],
            val_losses=-model.L,
            val_weights=ws_val,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger
        )

    return model


def learn_conditional_density(model, xs, ys, ws=None, regularizer=None, val_frac=0.05, step=ss.Adam(a=1.e-4), minibatch=100, patience=20, monitor_every=1, logger=sys.stdout, rng=np.random):
    """
    Train model to learn the conditional density p(y|x).
    """

    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)

    n_data = xs.shape[0]
    assert ys.shape[0] == n_data, 'wrong sizes'

    # shuffle data, so that training and validation sets come from the same distribution
    idx = rng.permutation(n_data)
    xs = xs[idx]
    ys = ys[idx]

    # split data into training and validation sets
    n_trn = int(n_data - val_frac * n_data)
    xs_trn, xs_val = xs[:n_trn], xs[n_trn:]
    ys_trn, ys_val = ys[:n_trn], ys[n_trn:]

    if ws is None:

        # train model without weights
        trainer = trainers.SGD(
            model=model,
            trn_data=[xs_trn, ys_trn],
            trn_loss=model.trn_loss if regularizer is None else model.trn_loss + regularizer,
            trn_target=model.y,
            val_data=[xs_val, ys_val],
            val_loss=model.trn_loss,
            val_target=model.y,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger
        )

    else:

        # prepare weights
        ws = np.asarray(ws, np.float32)
        assert ws.size == n_data, 'wrong sizes'
        ws = ws[idx]
        ws_trn, ws_val = ws[:n_trn], ws[n_trn:]

        # train model with weights
        trainer = trainers.WeightedSGD(
            model=model,
            trn_data=[xs_trn, ys_trn],
            trn_losses=-model.L,
            trn_weights=ws_trn,
            trn_reg=regularizer,
            trn_target=model.y,
            val_data=[xs_val, ys_val],
            val_losses=-model.L,
            val_weights=ws_val,
            val_target=model.y,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger
        )

    return model


class UMestimator:
    
    def __init__(self, sim_model, model):
        
        self.sim_model = sim_model
        self.model = model
        self.samples = None
        self.n_samples = None
        self.xdim = None
        self.target = sim_model.entropy()
        
    def learn_transformation(self, n_samples, logger=sys.stdout, rng=np.random,patience=10,val_tol=None, show_progress=False, minibatch = 128):
        """Learn the transformation to push a gaussian towards target distribution

        Args:
            n_samples (int): number of samples
            logger (_type_, optional): output log type. Defaults to sys.stdout.
            rng (_type_, optional): Defaults to np.random.
            patience (int, optional): How many epochs to try after finding the best validation. Defaults to 10.
            val_tol (int, optional): Tolerance of validation loss to decide when the model improved. Defaults to None.
        """
        
        if self.samples is None:
            xs = self.sim_model.sim(n_samples)
            self.samples = xs
        
        self.n_samples = n_samples
        self.x_dim = self.samples.shape[1]
        
        #Scale so validation occurs at most every 10**5 / minibatch during training
        monitor_every = min(10 ** 5 / float(n_samples), 1.0) 
        logger.write('training model...\n')
        learn_density(
            self.model, 
            self.samples, 
            monitor_every=monitor_every, 
            logger=logger, 
            rng=rng, 
            patience=patience, 
            val_tol=val_tol, 
            minibatch=minibatch,
            target=self.target,
            show_progress=show_progress
            )
        logger.write('training done\n')
        
    def calc_ent(self, k=1, reuse_samples=True, method='umtkl',SHOW_PDF_PLOTS=False):
        import theano #used to attach GPU to a thread
        if reuse_samples:
            samples = self.samples
        else:
            samples = self.sim_model.sim(self.n_samples)
        
        u = self.model.calc_random_numbers(samples)
        #remove extreme data that isn't within 99.9999% of the norm dist
        idx = np.all(np.abs(u)<stats.norm.ppf(1.0-1e-6), axis=1) 
        u = u[idx]

        if(SHOW_PDF_PLOTS==True):
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,3)
            x=np.linspace(-0.1,1.1,100)
            ax[0].plot(x,stats.uniform.pdf(x),lw=5)
            x=np.linspace(stats.norm.ppf(1e-6),stats.norm.ppf(1-1e-6),100)
            ax[1].plot(x,stats.norm.pdf(x),lw=5)
            ax[2].plot(x,stats.norm.pdf(x),lw=5)

            ax[0].hist(stats.norm.cdf(u),bins=40,density=True),ax[0].set_title("Transformed Uniform")
            ax[1].hist(u,bins=40,density=True),ax[1].set_title("Transformed Gaussian")
            ax[2].hist(samples,bins=100,density=True),ax[2].set_title("Original Data")
            
        

        #Made a bad gaussian estimate
        if(u.shape[0]<0.01*self.samples.shape[0]):
            return None,0,0,0
        
        
        z = stats.norm.cdf(u)
        correction1 = - np.mean(np.log(np.prod(stats.norm.pdf(u), axis=1)))

        h2=0
        
        if method == 'umtkl':                        
            h_thread = BackgroundThread(target = tkl,args=(z,None,k))
            # h = tkl(z, k=k) + correction1            
        elif method == 'umtksg':            
            h_thread = BackgroundThread(target = tksg,args=(z,None,k))
            # h = tksg(z, k=k) + correction1
        elif method == 'both':
            h_thread = BackgroundThread(target = tkl_tksg,args=(z,None,k))
            # h,h2 = tkl_tksg(z,k=k) + correction1
        h_thread.start()        
            
        correction2 = -np.mean(self.model.logdet_jacobi_u(samples)[idx])

        result = h_thread.get_result()
        if method == 'both':
            (h,h2)=result + correction1
        else:
            h=result + correction1
            
        # return h+correction2, correction1+correction2, kl(u)+correction2, ksg(u)+correction2
        return h+correction2,h2+correction2
    
    def ksg_ent(self, k=1, reuse_samples=True, method='kl'):
        
        if reuse_samples:
            samples = self.samples
        else:
            samples = self.sim_model.sim(self.n_samples)
        
        if method == 'kl':
            return kl(samples, k=k)
        elif method == 'ksg':
            return ksg(samples, k=k)
        
        
    
class UMestimator_mi:
    
    def __init__(self, sim_model, model_j, model_m):
        
        self.sim_model = sim_model
        self.model_j = model_j
        self.model_m = model_m
        self.samples = None
        self.n_samples = None
        self.xdim = None
        
    def learn_transformation(self, n_samples, logger=sys.stdout, rng=np.random):
        
        if self.samples is None:
            xs = self.sim_model.sim(n_samples)
            self.samples = xs
        
        self.n_samples = n_samples
        self.x_dim = self.samples.shape[1]
        
        monitor_every = min(10 ** 5 / float(n_samples), 1.0)
        logger.write('training joint density network...\n')
        learn_density(self.model_j, self.samples, monitor_every=monitor_every, logger=logger, rng=rng)
        logger.write('training done\n')
        
        xs1 = self.samples[:, :self.sim_model.dim_x]
        xs2 = self.samples[:, self.sim_model.dim_x:]
        xs2 = xs2[rng.permutation(self.n_samples),:]
        ys = np.concatenate((xs1,xs2), axis=1)
        logger.write('training marginal density network...\n')
        learn_density(self.model_m, ys, monitor_every=monitor_every, logger=logger, rng=rng)
        logger.write('training done\n')
        
    def calc_ent(self, k=1, reuse_samples=True, method='umtkl', rng=np.random):

        if reuse_samples:
            samples = self.samples
        else:
            samples = self.sim_model.sim(self.n_samples)
        
        u_j = self.model_j.calc_random_numbers(samples)
        samples1 = samples[:,:self.sim_model.dim_x]
        samples2 = samples[:,self.sim_model.dim_x:]
        samples2 = samples2[rng.permutation(self.n_samples),:]
        samples_m = np.concatenate((samples1,samples2), axis=1)
        u_m = self.model_m.calc_random_numbers(samples_m)
        
        if method == 'umtkl':
            u_j = u_j[np.all(np.abs(u_j)<stats.norm.ppf(1.0-1e-6), axis=1)]
            z_j = stats.norm.cdf(u_j)
            u_m = u_m[np.all(np.abs(u_m)<stats.norm.ppf(1.0-1e-6), axis=1)]
            z_m = stats.norm.cdf(u_m)
            h_j = tkl(z_j, k=k) - np.mean(np.log(np.prod(stats.norm.pdf(u_j), axis=1)))
            h_m = tkl(z_m, k=k) - np.mean(np.log(np.prod(stats.norm.pdf(u_m), axis=1)))
            
        elif method == 'umtksg':
            u_j = u_j[np.all(np.abs(u_j)<stats.norm.ppf(1.0-1e-6), axis=1)]
            z_j = stats.norm.cdf(u_j)
            u_m = u_m[np.all(np.abs(u_m)<stats.norm.ppf(1.0-1e-6), axis=1)]
            z_m = stats.norm.cdf(u_m)
            h_j = tksg(z_j, k=k) - np.mean(np.log(np.prod(stats.norm.pdf(u_j), axis=1)))
            h_m = tksg(z_m, k=k) - np.mean(np.log(np.prod(stats.norm.pdf(u_m), axis=1)))  
            
        correction_j = -np.mean(self.model_j.logdet_jacobi_u(samples))
        correction_m = -np.mean(self.model_m.logdet_jacobi_u(samples_m))

        return h_m+correction_m-(h_j+correction_j)
    
    def ksg_ent(self, k=1, reuse_samples=True, method='kl'):
        
        if reuse_samples:
            samples = self.samples
        else:
            samples = self.sim_model.sim(self.n_samples)
        
        if method == 'kl':
            return mi_kl(samples, self.sim_model.dim_x, k=k)
        elif method == 'ksg':
            return mi_ksg(samples, self.sim_model.dim_x, k=k)
            
            
    
    
    