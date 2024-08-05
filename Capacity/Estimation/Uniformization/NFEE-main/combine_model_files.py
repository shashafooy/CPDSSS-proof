import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc


def compare_models(samples,name,base_folder,new_model_folder):
    model = ent.create_model(samples.shape[1])
    name = f'CPDSSS_{name}_{folder}'
    print(f"checking training loss for {name}")
    old_loss = ent.load_model(model,name,base_folder).eval_trnloss(samples)
    new_loss = ent.load_model(model,name,new_model_folder).eval_trnloss(samples)
    print(f"old loss: {old_loss:.3f}, new loss: {new_loss:.3f}")
    if new_loss<old_loss:
        ent.save_model(model,name,base_folder)


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
    
    sim_model = CPDSSS(T,N,L,use_gaussian_approx=GQ_gaussian)
    #generate base samples based on max dimension
    sim_model.set_dim_joint()
    knn_samples = int(min(min_samples, 0.75*n_train_samples * sim_model.x_dim))
    X,X_T,X_cond,h = sim_model.get_base_X_h(knn_samples)
    hxc=np.concatenate((X_cond,h),axis=1)
    joint=np.concatenate((X,h),axis=1)


    sim_model.set_dim_hxc()
    compare_models(hxc,'hxc',base_folder,hpc_folder)

    sim_model.set_dim_xxc()
    compare_models(X,'xxc',base_folder,hpc_folder)

    sim_model.set_dim_cond()
    compare_models(X_cond,'Xcond',base_folder,hpc_folder)

    sim_model.set_dim_joint()
    compare_models(joint,'joint',base_folder,hpc_folder)


