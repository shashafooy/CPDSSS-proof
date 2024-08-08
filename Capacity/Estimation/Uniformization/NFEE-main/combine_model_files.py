import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc


def compare_models(samples,name1,base_folder, name2, new_model_folder):
    model = ent.create_model(samples.shape[1])    
    print(f"checking training loss for {name1} and {name2}")
    old_loss = ent.load_model(model,name1,base_folder).eval_trnloss(samples)
    new_loss = ent.load_model(model,name2,new_model_folder).eval_trnloss(samples)
    print(f"{name1} loss: {old_loss:.3f}, new loss: {new_loss:.3f}")
    if new_loss<old_loss:
        ent.save_model(model,name1,base_folder)
    else:
        model = ent.load_model(model,name1,base_folder)
        ent.save_model(model,name2,new_model_folder)


"""
Parameters for CPDSSS
"""
N=4
L=2
M=int(N/L)
P=N-int(N/L)

min_samples = 2000000 #samples to generate per entropy calc
n_train_samples = 100000
GQ_gaussian = False

"""
Generate data
"""

model_path_old = 'temp_data/saved_models'
model_path_new = 'temp_data/saved_models/HPC'

sub_folders = ['2T','3T','4T','5T','6T','7T']



for folder in sub_folders:
    base_folder = os.path.join(model_path_old,folder)
    hpc_folder = os.path.join(model_path_new,folder)
    
    T = int(folder[:-1])
    print(f"\nComparing models for {T}T")
    
    sim_model = CPDSSS(T,N,L,use_gaussian_approx=GQ_gaussian)
    #generate base samples based on max dimension
    sim_model.set_dim_joint()
    knn_samples = int(min(min_samples, 0.75*n_train_samples * sim_model.x_dim))
    X,X_T,X_cond,h = sim_model.get_base_X_h(knn_samples)
    hxc=np.concatenate((X_cond,h),axis=1)
    joint=np.concatenate((X,h),axis=1)

    #Compare with new model files
    name = f'CPDSSS_hxc_{folder}'
    compare_models(hxc,name,base_folder,name,hpc_folder)

    name = f'CPDSSS_xxc_{folder}'
    compare_models(X,name,base_folder,name,hpc_folder)

    name = f'CPDSSS_Xcond_{folder}'
    compare_models(X_cond,name,base_folder,name,hpc_folder)

    name = f'CPDSSS_joint_{folder}'
    compare_models(joint,name,base_folder,name,hpc_folder)

    #Compare with models from previous T, but same distribution
    if folder != sub_folders[0]:
        print(f"\nComparing models {T}T with previous {T-1}T")
        sim_model.set_dim_hxc()
        compare_models(hxc,f'CPDSSS_hxc_{T}T',base_folder,f'CPDSSS_joint_{T-1}T',prev_folder)

        sim_model.set_dim_cond()
        compare_models(X_cond,f'CPDSSS_Xcond_{T}T',base_folder,f'CPDSSS_xxc_{T-1}T',prev_folder)

    prev_folder = base_folder
