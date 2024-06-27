import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc

from datetime import date



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
T_range = range(2,8)

"""
Number of iterations
"""
n_trials = 100 #iterations to average
knn_samples = 500000 #samples to generate per entropy calc
n_train_samples = 30000
completed_iter=0
GQ_gaussian = False


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
H_h=np.empty((n_trials,len(T_range)))*np.nan
H_xh=np.empty((n_trials,len(T_range)))*np.nan

        
"""
File names
"""

today=date.today().strftime("%b_%d")
# filename="CPDSSS_data_dump(0_iter)({0}k_samples)({1})".format(int(knn_samples/1000),today)

base_path = "temp_data/CPDSSS_data/MI(h,X)/N4_L2/"

path = 'temp_data/CPDSSS_data/50k_N4_L2'
path = 'temp_data/CPDSSS_data/NlogN_10k_K=3'
path = 'temp_data/CPDSSS_data/NlogN_10k_K=3,T=8,samp=40k'
path = "temp_data/CPDSSS_data/N4_L2/Nscaling_knn={}k_T=8".format(int(knn_samples/1000))
path = "temp_data/CPDSSS_data/N4_L2/Nscaling_knn={}k_T=2-7,learnTol=0.05".format(int(knn_samples/1000))

path = base_path + "tol=0.01,T=2-7/".format(int(knn_samples/1000))
filename = "CPDSSS_data_dump({})".format(today)

# path = "temp_data/CPDSSS_data/Ignore"
# filename=os.path.join(path, filename)

#fix filename if file already exists
filename = misc.update_filename(path,filename,completed_iter,rename=False)    
#create initial file
# util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,0), os.path.join(path,filename)) 
# filename = update_filename(path,filename,n_samples,today,1) 


"""
Generate data
"""
for i in range(n_trials):        
            
    for k, T in enumerate(T_range):
        sim_model = CPDSSS(T,N,L,use_gaussian_approx=GQ_gaussian)

        n_sims = knn_samples

        X,X_T,X_cond,h = sim_model.get_base_X_h(n_sims)
        hxc=np.concatenate((X_cond,h),axis=1)
        joint=np.concatenate((X,h),axis=1)

        if T==1:
            sim_model.set_use_h_flag(h_flag=False)
            misc.print_border("Calculate H(x), T: 1, iter: {}".format(i+1))
            H_x[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=X_T)

            sim_model.set_use_h_flag(h_flag=True)
            sim_model.set_sim_h_only(sim_h=True)
            misc.print_border("Calculate H(h), T: 1, iter: {}".format(i+1))
            H_h[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=h)

            sim_model.set_sim_h_only(sim_h=False)
            sim_model.set_use_h_flag(h_flag=True)            
            misc.print_border("Calculate H(x,h), T: 1, iter: {}".format(i+1))
            H_xh[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=joint)

            MI_cum[i,k] = H_x[i,k] + H_h[i,k] - H_xh[i,k]
            if k== np.size(T_range)-1:
                completed_iter = completed_iter + 1
                filename = misc.update_filename(path,filename,completed_iter)    

            util.io.save((T_range, MI_cum,H_x,H_h,H_xh,i), os.path.join(path,filename)) 

        else:
            first_tx_model = CPDSSS(T-1,N,L,use_gaussian_approx=GQ_gaussian)

            first_tx_model.set_use_h_flag(h_flag=True)
            misc.print_border("1/4 calculating H(h,x_old), T: {0}, iter: {1}".format(T,i+1))        
            H_gxc = ent.calc_entropy(sim_model=first_tx_model,n_samples=n_train_samples,base_samples=hxc,val_tol=0.02)

            first_tx_model.set_use_h_flag(h_flag=False)
            misc.print_border("2/4 calculating H(x_old), T: {0}, iter: {1}".format(T,i+1))
            H_cond = ent.calc_entropy(sim_model=first_tx_model,n_samples=n_train_samples,base_samples=X_cond,val_tol=0.02)

            sim_model.set_use_h_flag(False)
            misc.print_border("3/4 calculating H(x_T, x_old), T: {0}, iter: {1}".format(T,i+1))
            H_xxc = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=X,val_tol=0.02)

            sim_model.set_use_h_flag(True)
            misc.print_border("4/4 calculating H_(h,x_T,x_old), T: {0}, iter: {1}".format(T,i+1))
            H_joint = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=joint,val_tol=0.02)

            H_gxc_cum[i,k]=H_gxc
            H_xxc_cum[i,k]=H_xxc
            H_joint_cum[i,k]=H_joint
            H_cond_cum[i,k]=H_cond
            MI_cum[i,k] = H_gxc + H_xxc - H_joint - H_cond
            if k== np.size(T_range)-1:
                completed_iter = completed_iter + 1
                filename = misc.update_filename(path,filename,completed_iter)    
            
            ''' try using i as completed iter. matrix row is not complete, but this saves at least some data'''
            util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 
