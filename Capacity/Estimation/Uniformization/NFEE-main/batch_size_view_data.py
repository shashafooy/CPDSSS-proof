from datetime import timedelta
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np
import re

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

plt.rcParams['text.usetex']=True




"""
Load and combine all datasets
"""

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = 'temp_data/batch_size/20N_100k_train'

for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    if re.search(r"\(-1_iter\)",filename): continue #skip empty file
    _batch_size,N,_error,_duration,_H_sim,_H_reuse = util.io.load(os.path.join(filepath, filename))
    

    # Initialize arrays
    if 'error' not in locals():
        batch_size = _batch_size
        error = np.empty((0,len(_batch_size)))
        duration = np.empty((0,len(_batch_size)))
        H_sim = np.empty((0,len(_batch_size)))
        H_reuse = np.empty((0,len(_batch_size)))
        
       

    error,_ = viewData.align_and_concatenate(error,_error,(batch_size),(_batch_size))
    duration,_ = viewData.align_and_concatenate(duration,_duration,(batch_size),(_batch_size))
    H_sim,_ = viewData.align_and_concatenate(H_sim,_H_sim,(batch_size),(_batch_size))
    H_reuse,(batch_size) = viewData.align_and_concatenate(H_reuse,_H_reuse,(batch_size),(_batch_size))
    
viewData.clean_data(error)
viewData.clean_data(duration)
viewData.clean_data(H_sim)
viewData.clean_data(H_reuse)

# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    error = viewData.remove_outlier(error)
    duration = viewData.remove_outlier(duration)
    H_sim = viewData.remove_outlier(H_sim)
    H_reuse = viewData.remove_outlier(H_reuse)


#remove 4096
batch_size = batch_size[:-1]
error = error[:,:-1]
duration = duration[:,:-1]
H_sim = H_sim[:,:-1]
H_reuse = H_reuse[:,:-1]


H_true = Laplace(mu=0,b=2,N=N).entropy()
H_sim_err = np.abs(H_sim-H_true)
H_reuse_err = np.abs(H_reuse-H_true)



mean_err = np.nanmean(error,axis=0)
mean_H_sim_err = np.nanmean(H_sim_err,axis=0)
mean_H_reuse_err = np.nanmean(H_reuse_err,axis=0)
mean_duration = np.round(np.nanmean(duration,axis=0))

var_err = np.nanvar(error,axis=0)
var_duration = np.nanvar(duration,axis=0)
var_H_sim_err = np.nanvar(H_sim_err,axis=0)
var_H_reuse_err = np.nanvar(H_reuse_err,axis=0)

MSE_train = np.nanmean(error**2,axis=0)
MSE_H_sim = np.nanmean(H_sim_err**2,axis=0)
MSE_H_reuse = np.nanmean(H_reuse_err**2,axis=0)



# PLOTS


for i in range(len(batch_size)):
    min_sec = str(timedelta(seconds = mean_duration[i]))
    print(f'Batch: {batch_size[i]}, Mean training duration: {min_sec}')


plt.figure(1)
# plt.axhline(y=H_true,linestyle='--')
plt.plot(batch_size, mean_err, marker='o', label='training')
plt.plot(batch_size, mean_H_sim_err, marker='o', label='uniform entropy')
# plt.plot(batch_size, mean_H_reuse_err, marker='o', linestyle='None', label='mean_H_reuse_err')
plt.xscale('log')
plt.xticks(batch_size,labels=[str(a) for a in batch_size])
plt.title("validation error per batch size")
plt.xlabel("batch size")
plt.ylabel("H(x) error")
plt.legend()
# plt.legend(["MAF error","knn new data err","knn training data err"])
plt.tight_layout()

plt.figure(2)
# plt.axhline(y=H_true,linestyle='--')
plt.plot(batch_size, MSE_train, marker='o', label='training')
plt.plot(batch_size, MSE_H_sim, marker='o', label='uniform entropy')
# plt.plot(batch_size, MSE_H_reuse, marker='o', linestyle='None', label='reuse data')
plt.yscale('log')
plt.xscale('log')
plt.xticks(batch_size,labels=[str(a) for a in batch_size])
plt.title("MSE per batch size")
plt.xlabel("batch size")
plt.ylabel("H(x) MSE")
plt.legend()
plt.tight_layout()


plt.show()