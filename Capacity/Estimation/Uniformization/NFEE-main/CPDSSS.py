import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc

from datetime import date

import configparser
config = configparser.ConfigParser()
config.read('CPDSSS.ini')
KNN_THREADING = config['GLOBAL'].getboolean('knn_GPU',False)



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

SAVE_MODEL = True

"""
Number of iterations
"""
n_trials = 100 #iterations to average
min_knn_samples = 2000000 #samples to generate per entropy calc
n_train_samples = 100000
GQ_gaussian = False
use_pretrained = True
fine_tune = False


"""
Initialize arrays
"""
MI_tKL = np.empty(len(T_range))
MI_means = np.empty(len(T_range))
MI = np.empty((n_trials,len(T_range)))*np.nan
H_hxc=np.empty((n_trials,len(T_range)))*np.nan
H_xxc=np.empty((n_trials,len(T_range)))*np.nan
H_joint=np.empty((n_trials,len(T_range)))*np.nan
H_cond=np.empty((n_trials,len(T_range)))*np.nan
H_x=np.empty((n_trials,len(T_range)))*np.nan
H_h=np.empty((n_trials,len(T_range)))*np.nan
H_xh=np.empty((n_trials,len(T_range)))*np.nan

best_trn_loss = np.ones((T_range[-1],4))*1e5

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
path = base_path + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path = f'temp_data/saved_models/{N}N'
X_path = os.path.join(model_path,'X')
XH_path = os.path.join(model_path,'XH')


#fix filename if file already exists
filename = misc.update_filename(path,filename,-1,rename=False)     
# model = ent.load_model(8,'CPDSSS_hxc_2T','temp_data/saved_models/2T')

"""
Generate data
"""
prev_idx=(0,0)
for i in range(n_trials):        
            
    for k, T in enumerate(T_range):
        index = (i,k)

        sim_model = CPDSSS(T,N,L,use_gaussian_approx=GQ_gaussian)
        #generate base samples based on max dimension
        sim_model.set_dim_joint()
        knn_samples = int(min(min_knn_samples, 0.75*n_train_samples * sim_model.x_dim))
        X,X_T,X_cond,h = sim_model.get_base_X_h(knn_samples)
        hxc=np.concatenate((X_cond,h),axis=1)
        joint=np.concatenate((X,h),axis=1)

        """Calculate entropies needed for mutual information. Evaluate knn entropy (CPU) while training new model (GPU)
            General flow: 
                Train model (main thread)
                evaluate uniform points and run knn (background thread)
                wait for previoius knn thread to finish (main thread)
                combine knn entropy with jacobian correction term (main thread)
                Start new model while current knn is running (main thread)
         """

        prev_tx_model = CPDSSS(T-1,N,L,use_gaussian_approx=GQ_gaussian)
        
        '''Train H(h,x_cond)'''
        misc.print_border("1/4 calculating H(h,x_old), T: {0}, iter: {1}".format(T,i+1))        
        sim_model.set_dim_hxc()
        name = f'{T-1}T'        
        model = ent.load_model(name=name,path = XH_path)
        if KNN_THREADING:            
            H_hxc_thread,H_hxc_correction,estimator = ent.calc_entropy_thread(sim_model,n_train_samples,hxc,model = model)
            if H_joint_thread is not None: #don't run if first iteration            
                knn = H_joint_thread.get_result()
                H_joint[prev_idx] = knn + H_joint_correction         
                MI[prev_idx] = H_hxc[prev_idx] + H_xxc[prev_idx] - H_joint[prev_idx] - H_cond[prev_idx]
        else:
            H_hxc[index],estimator = ent.calc_entropy(sim_model,n_train_samples,hxc,model = model)
        if SAVE_MODEL:
            best_trn_loss[T-1,0] = ent.update_best_model(estimator.model,hxc,best_trn_loss[T-1,0],name,XH_path)
        
        filename = misc.update_filename(path,filename,i)            
        util.io.save((T_range, MI,H_hxc,H_xxc,H_joint,H_cond,i), os.path.join(path,filename)) 

        '''Train H(x_cond)'''
        misc.print_border("2/4 calculating H(x_old), T: {0}, iter: {1}".format(T,i+1))
        
        sim_model.set_dim_cond()
        name=f'{T-1}T'
        model = ent.load_model(name=name,path = X_path)
        if KNN_THREADING:
            H_cond_thread,H_cond_correction,estimator = ent.calc_entropy_thread(sim_model,n_train_samples,X_cond,model=model)        
            #wait for knn H(h,x_cond)
            knn = H_hxc_thread.get_result()        
            H_hxc[index] = knn + H_hxc_correction #knn + correction
        else:
            H_cond[index],estimator = ent.calc_entropy(sim_model,n_train_samples,X_cond,model=model)        
        if SAVE_MODEL:
            best_trn_loss[T-1,1] = ent.update_best_model(estimator.model,X_cond,best_trn_loss[T-1,1],name,X_path)
        
        
        util.io.save((T_range, MI,H_hxc,H_xxc,H_joint,H_cond,i), os.path.join(path,filename)) 

        '''Train H(x_T,x_cond)'''
        misc.print_border("3/4 calculating H(x_T, x_old), T: {0}, iter: {1}".format(T,i+1))        
        sim_model.set_dim_xxc()
        name=f'{T}T'
        model = ent.load_model(name=name,path = X_path)
        if KNN_THREADING:
            H_xxc_thread, H_xxc_correction,estimator = ent.calc_entropy_thread(sim_model,n_train_samples,X,model=model)
            knn = H_cond_thread.get_result() #previous result
            H_cond[index] = knn + H_cond_correction
        else:
            H_xxc[index], estimator = ent.calc_entropy(sim_model,n_train_samples,X,model=model)
        if SAVE_MODEL:
            best_trn_loss[T-1,2] = ent.update_best_model(estimator.model,X,best_trn_loss[T-1,2],name,X_path)
                
        util.io.save((T_range, MI,H_hxc,H_xxc,H_joint,H_cond,i), os.path.join(path,filename)) 
        
        '''Train H(h,x_T,x_cond)'''
        misc.print_border("4/4 calculating H_(h,x_T,x_old), T: {0}, iter: {1}".format(T,i+1))
        sim_model.set_dim_joint()
        name=f'{T}T'
        model = ent.load_model(name=name,path = XH_path)
        if KNN_THREADING:
            H_joint_thread, H_joint_correction,estimator = ent.calc_entropy_thread(sim_model,n_train_samples,joint,model=model)        
            knn = H_xxc_thread.get_result()
            H_xxc[index] = knn + H_xxc_correction
        else:
            H_joint[index], estimator = ent.calc_entropy(sim_model,n_train_samples,joint,model=model)        
            MI[index] = H_hxc[index] + H_xxc[index] - H_joint[index] - H_cond[index]
        if SAVE_MODEL:
            best_trn_loss[T-1,3] = ent.update_best_model(estimator.model,joint,best_trn_loss[T-1,3],name,XH_path)
        #wait for knn H(x_T,x_cond)
        

        util.io.save((T_range, MI,H_hxc,H_xxc,H_joint,H_cond,i), os.path.join(path,filename)) 

        #Save this index set for next iteration
        prev_idx = index


        # if k== np.size(T_range)-1:
        #     completed_iter = completed_iter + 1
    filename = misc.update_filename(path,filename,i+1)            
   

#get knn H(joint) from final iteraiton
knn = H_joint_thread.get_result()
H_joint[prev_idx] = knn + H_joint_correction 
#Combine entropies for mutual information
MI[prev_idx] = H_hxc[prev_idx] + H_xxc[prev_idx] - H_joint[prev_idx] - H_cond[prev_idx]
