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

save_best_model = True

"""
Number of iterations
"""
n_trials = 100 #iterations to average
min_knn_samples = 1000000 #samples to generate per entropy calc
n_train_samples = 100000
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

best_trn_loss = np.empty((T_range[-1],4))*np.Inf

H_hxc_thread = None
H_cond_thread = None
H_xxc_thread = None
H_joint_thread = None


        
"""
File names
"""
today=date.today().strftime("%b_%d")
base_path = "temp_data/CPDSSS_data/MI(h,X)/N4_L2/"

path = base_path + "coarse-fine_75k_x_dims"
filename = "CPDSSS_data({})".format(today)



#fix filename if file already exists
filename = misc.update_filename(path,filename,-1,rename=False)     


"""
Generate data
"""
for i in range(n_trials):        
            
    for k, T in enumerate(T_range):
        sim_model = CPDSSS(T,N,L,use_gaussian_approx=GQ_gaussian)
        #generate base samples based on max dimension
        sim_model.use_chan_in_sim()
        knn_samples = int(min(min_knn_samples, 0.75*n_train_samples * sim_model.x_dim))
        X,X_T,X_cond,h = sim_model.get_base_X_h(knn_samples)
        hxc=np.concatenate((X_cond,h),axis=1)
        joint=np.concatenate((X,h),axis=1)

        # if T==1:
        #     sim_model.set_use_h_flag(h_flag=False)
        #     misc.print_border("Calculate H(x), T: 1, iter: {}".format(i+1))
        #     estimator = ent.learn_model(sim_model,n_train_samples)
        #     estimator.samples = X_T

        #     H_x[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=X_T)

        #     sim_model.set_use_h_flag(h_flag=True)
        #     sim_model.set_sim_h_only(sim_h=True)
        #     misc.print_border("Calculate H(h), T: 1, iter: {}".format(i+1))
        #     H_h[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=h)

        #     sim_model.set_sim_h_only(sim_h=False)
        #     sim_model.set_use_h_flag(h_flag=True)            
        #     misc.print_border("Calculate H(x,h), T: 1, iter: {}".format(i+1))
        #     H_xh[i,k] = ent.calc_entropy(sim_model=sim_model,n_samples=n_train_samples,base_samples=joint)

        #     MI_cum[i,k] = H_x[i,k] + H_h[i,k] - H_xh[i,k]
        #     if k== np.size(T_range)-1:
        #         completed_iter = completed_iter + 1
        #         filename = misc.update_filename(path,filename,completed_iter)    

        #     util.io.save((T_range, MI_cum,H_x,H_h,H_xh,i), os.path.join(path,filename)) 

        # else:

        """Calculate entropies needed for mutual information. Evaluate knn entropy (CPU) while training new model (GPU)
            General flow: 
                Train model (main thread)
                evaluate uniform points and run knn (background thread)
                wait for previoius knn thread to finish (main thread)
                combine knn entropy with jacobian correction term (main thread)
                Start new model while current knn is running (main thread)
         """

        prev_tx_model = CPDSSS(T-1,N,L,use_gaussian_approx=GQ_gaussian)
        
        #Train H(h,x_cond)
        misc.print_border("1/4 calculating H(h,x_old), T: {0}, iter: {1}".format(T,i+1))        
        prev_tx_model.use_chan_in_sim()
        H_hxc_thread,H_hxc_correction,hxc_model = ent.calc_entropy_thread(prev_tx_model,n_train_samples,hxc)
        if save_best_model:
            best_trn_loss[T,0] = ent.update_best_model(hxc_model,hxc,best_trn_loss[T,0],f'CPDSSS_hxc_{T}T')
        if H_joint_thread is not None: #don't run if first iteration
            #get knn H(joint) from previous iteration
            knn = H_joint_thread.get_result()
            H_joint_cum[prev_idx] = knn + H_joint_correction 
            #Combine entropies for mutual information
            MI_cum[prev_idx] = H_gxc_cum[prev_idx] + H_xxc_cum[prev_idx] - H_joint_cum[prev_idx] - H_cond_cum[prev_idx]
        # H_hxc = ent.calc_entropy(sim_model=first_tx_model,n_samples=n_train_samples,base_samples=hxc,val_tol=0.02)
        
        filename = misc.update_filename(path,filename,i)            
        util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 

        #Train H(x_cond)
        misc.print_border("2/4 calculating H(x_old), T: {0}, iter: {1}".format(T,i+1))
        prev_tx_model.use_chan_in_sim(False)
        H_cond_thread,H_cond_correction,cond_model = ent.calc_entropy_thread(prev_tx_model,n_train_samples,X_cond)        
        if save_best_model:
            best_trn_loss[T,1] = ent.update_best_model(cond_model,X_cond,best_trn_loss[T,1],f'CPDSSS_Xcond_{T}T')
        #wait for knn H(h,x_cond)
        knn = H_hxc_thread.get_result()        
        H_gxc_cum[i,k] = knn + H_hxc_correction #knn + correction
        
        util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 

        #Train H(x_T,x_cond)
        misc.print_border("3/4 calculating H(x_T, x_old), T: {0}, iter: {1}".format(T,i+1))
        sim_model.use_chan_in_sim(False)
        H_xxc_thread, H_xxc_correction,xxc_model = ent.calc_entropy_thread(sim_model,n_train_samples,X)
        if save_best_model:
            best_trn_loss[T,2] = ent.update_best_model(xxc_model,X,best_trn_loss[T,2],f'CPDSSS_xxc_{T}T')
        #wait for knn H(x_cond)
        knn = H_cond_thread.get_result()
        H_cond_cum[i,k] = knn + H_cond_correction

        util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 
        
        #Train H(h,x_T,x_cond)
        misc.print_border("4/4 calculating H_(h,x_T,x_old), T: {0}, iter: {1}".format(T,i+1))
        sim_model.use_chan_in_sim()
        H_joint_thread, H_joint_correction,joint_model = ent.calc_entropy_thread(sim_model,n_train_samples,joint)        
        if save_best_model:
            best_trn_loss[T,3] = ent.update_best_model(hxc_model,joint,best_trn_loss[T,3],f'CPDSSS_joint_{T}T')
        #wait for knn H(x_T,x_cond)
        knn = H_xxc_thread.get_result()
        H_xxc_cum[i,k] = knn + H_xxc_correction

        util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,i), os.path.join(path,filename)) 

        #Save this index set for next iteration
        prev_idx = (i,k)


        # if k== np.size(T_range)-1:
        #     completed_iter = completed_iter + 1
    filename = misc.update_filename(path,filename,i+1)            
   

#get knn H(joint) from final iteraiton
knn = H_joint_thread.get_result()
H_joint_cum[prev_idx] = knn + H_joint_correction 
#Combine entropies for mutual information
MI_cum[prev_idx] = H_gxc_cum[prev_idx] + H_xxc_cum[prev_idx] - H_joint_cum[prev_idx] - H_cond_cum[prev_idx]
