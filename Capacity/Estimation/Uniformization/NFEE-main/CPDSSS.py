import numpy as np
from scipy import stats

import util.io
import os

from ent_est import entropy
from ent_est.entropy import tkl

from simulators.complex import mvn
from simulators.CPDSSS_models import CPDSSS
from datetime import date
from datetime import timedelta

import time
import gc

"""
Functions for generating the model and entropy
"""
def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

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

def calc_entropy(sim_model,base_samples=None,n_samples=100):
    H=-1
    val_tol = 0.1
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




"""
Parameters for CPDSSS
"""
N=4
L=2
M=int(N/L)
P=N-int(N/L)
max_T=5
T_range = range(2,N+max_T)
T_range = range(8,9)
T_range = range(4,8)

"""
Number of iterations
"""
n_trials = 100 #iterations to average
knn_samples = 200000 #samples to generate per entropy calc
n_train_samples = 100
completed_iter=0


"""
Initialize arrays
"""
MI_tKL = np.empty(len(T_range))
MI_means = np.empty(len(T_range))
MI_cum = np.empty((n_trials,len(T_range)))*np.nan
H_gxc_cum=np.empty((n_trials,len(T_range)))*np.nan
H_xxc_cum=np.empty((n_trials,len(T_range)))*np.nan
H_joint_cum=np.empty((n_trials,len(T_range)))*np.nan
H_cond_cum=np.empty((n_trials,len(T_range)))*np.nan
H_x=np.empty((n_trials,len(T_range)))*np.nan
H_g=np.empty((n_trials,len(T_range)))*np.nan
H_xg=np.empty((n_trials,len(T_range)))*np.nan

        
"""
File names
"""

today=date.today().strftime("%b_%d")
filename="CPDSSS_data_dump(0_iter)({0}k_samples)({1})".format(int(knn_samples/1000),today)
path = 'temp_data/CPDSSS_data/50k_N4_L2'
path = 'temp_data/CPDSSS_data/NlogN_10k_K=3'
path = 'temp_data/CPDSSS_data/NlogN_10k_K=3,T=8,samp=40k'
path = "temp_data/CPDSSS_data/N4_L2/Nscaling_knn={}k_T=8".format(int(knn_samples/1000))
path = "temp_data/CPDSSS_data/N4_L2/Nscaling_knn={}k_T=2-7".format(int(knn_samples/1000))

# path = "temp_data/CPDSSS_data/Ignore"
# filename=os.path.join(path, filename)

#fix filename if file already exists
filename = update_filename(path,'',knn_samples,today,completed_iter,rename=False)    
#create initial file
util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,0), os.path.join(path,filename)) 
# filename = update_filename(path,filename,n_samples,today,1) 


"""
Generate data
"""
for i in range(n_trials):        
            
    for k, T in enumerate(T_range):
        sim_model = CPDSSS(T,N,L)

        n_sims = knn_samples

        X,X_T,X_cond,G = sim_model.get_base_X_G(n_sims)
        gxc=np.concatenate((X_cond,G),axis=1)
        joint=np.concatenate((X,G),axis=1)

        if T==1:
            sim_model.set_use_G_flag(g_flag=False)
            print_border("Calculate H(x), T: 1, iter: {}".format(i+1))
            H_x[i,k] = calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=X_T)

            sim_model.set_use_G_flag(g_flag=True)
            sim_model.set_sim_G_only(sim_g=True)
            print_border("Calculate H(g), T: 1, iter: {}".format(i+1))
            H_g[i,k] = calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=G)

            sim_model.set_sim_G_only(sim_g=False)
            sim_model.set_use_G_flag(g_flag=True)            
            print_border("Calculate H(x,g), T: 1, iter: {}".format(i+1))
            H_xg[i,k] = calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=joint)

            MI_cum[i,k] = H_x[i,k] + H_g[i,k] - H_xg[i,k]
            if k== np.size(T_range)-1:
                completed_iter = completed_iter + 1
                filename = update_filename(path,filename,n_sims,today,completed_iter)    

            util.io.save((T_range, MI_cum,H_x,H_g,H_xg,i), os.path.join(path,filename)) 

        else:
            first_tx_model = CPDSSS(T-1,N,L)

            first_tx_model.set_use_G_flag(g_flag=True)
            print_border("calculating H_gxc, T: {0}, iter: {1}".format(T,i+1))        
            H_gxc = calc_entropy(sim_model=first_tx_model,n_samples=n_train_samples,base_samples=gxc)

            first_tx_model.set_use_G_flag(g_flag=False)
            print_border("calculating H_cond, T: {0}, iter: {1}".format(T,i+1))
            H_cond = calc_entropy(sim_model=first_tx_model,n_samples=n_train_samples,base_samples=X_cond)

            sim_model.set_use_G_flag(False)
            print_border("calculating H_xxc, T: {0}, iter: {1}".format(T,i+1))
            H_xxc = calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=X)

            sim_model.set_use_G_flag(True)
            print_border("calculating H_joint, T: {0}, iter: {1}".format(T,i+1))
            H_joint = calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=joint)

            H_gxc_cum[i,k]=H_gxc
            H_xxc_cum[i,k]=H_xxc
            H_joint_cum[i,k]=H_joint
            H_cond_cum[i,k]=H_cond
            MI_cum[i,k] = H_gxc + H_xxc - H_joint - H_cond
            if k== np.size(T_range)-1:
                completed_iter = completed_iter + 1
                filename = update_filename(path,filename,n_sims,today,completed_iter)    
            
            ''' try using i as completed iter. matrix row is not complete, but this saves at least some data'''
            util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 
