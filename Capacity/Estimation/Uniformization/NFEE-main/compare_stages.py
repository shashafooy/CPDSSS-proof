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
T=4

n_stages=range(1,15)
"""
Number of iterations
"""
min_samp = 2e4
scale_samp = 1e3
n_samples = int(min(min_samp,scale_samp*(N+1)*T)) #samples to generate per entropy calc

"""
Initialize arrays
"""
H_hxc=np.empty((len(n_stages)))*np.nan
H_xxc=np.empty((len(n_stages)))*np.nan
H_joint=np.empty((len(n_stages)))*np.nan
H_cond=np.empty((len(n_stages)))*np.nan

        
"""
File names
"""

model_path = f'temp_data/saved_models/{N}N'
X_path = os.path.join(model_path,'X')
XH_path = os.path.join(model_path,'XH')


"""
Generate data
"""
sim_model = CPDSSS(T,N,L)
#generate base samples based on max dimension
sim_model.set_dim_joint()
X,X_T,X_cond,h = sim_model.get_base_X_h(n_samples)
hxc=np.concatenate((X_cond,h),axis=1)
joint=np.concatenate((X,h),axis=1)

prev_idx=(0,0)

model = ent.load_model(name = f'{T}T',path=X_path)

for i in n_stages:        
    H_xxc[i] = model.eval_stageloss(X,i)

print(H_xxc)

import matplotlib.pyplot as plt

plt.plot(n_stages,H_xxc)
plt.xlabel("Num stages")
plt.ylabel("Loss")
plt.title("Loss after N stages")