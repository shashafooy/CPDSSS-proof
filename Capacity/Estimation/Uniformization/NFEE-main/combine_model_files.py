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
N=6
L=2
M=int(N/L)
P=N-int(N/L)

min_samples = 2000000 #samples to generate per entropy calc
n_train_samples = 100000
GQ_gaussian = False
max_T=7

"""
Generate data
"""

model_path_old = f'temp_data/saved_models/{N}N'
model_path_new = f'temp_data/saved_models/{N}N_old'

sub_folders = ['X','XH']



# for folder in sub_folders:
#     base_folder = os.path.join(model_path_old,folder)
#     new_folder = os.path.join(model_path_new,folder)
base_folder_X = os.path.join(model_path_old,'X')
base_folder_XH = os.path.join(model_path_old,'XH')
old_folder_X = os.path.join(model_path_new,'X')
old_folder_XH = os.path.join(model_path_new,'XH')

sim_model = CPDSSS(max_T,N,L)
#generate base samples based on max dimension
sim_model.set_dim_joint()
knn_samples = int(min(min_samples, 0.75*n_train_samples * sim_model.x_dim))
X,_,_,h = sim_model.get_base_X_h(knn_samples)

for file in os.listdir(base_folder_X):
    T=int(file[0])
    X_samp = X[:,:T*N]
    XH_samp=np.concatenate((X_samp,h),axis=1)

    name = f'{T}T'

    #Compare with new model files
    compare_models(X_samp,name,base_folder_X,name,old_folder_X)
    compare_models(XH_samp,name,base_folder_XH,name,old_folder_XH)
    

