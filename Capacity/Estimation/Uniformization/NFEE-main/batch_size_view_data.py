from datetime import timedelta
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

plt.rcParams['text.usetex']=True




"""
Load and combine all datasets
"""

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = 'temp_data/batch_size/15N_100k_train'
N=15

for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    _batch_size,N,_error,_duration = util.io.load(os.path.join(filepath, filename))
    

    # Initialize arrays
    if 'error' not in locals():
        batch_size = _batch_size
        error = np.empty((0,len(_batch_size)))
        duration = np.empty((0,len(_batch_size)))
        
       

    error,_ = viewData.align_and_concatenate(error,_error,(batch_size),(_batch_size))
    duration,_ = viewData.align_and_concatenate(duration,_duration,(batch_size),(_batch_size))
    
viewData.clean_data(error)
viewData.clean_data(duration)

# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    error = viewData.remove_outlier(error)
    duration = viewData.remove_outlier(duration)


mean_err = np.nanmean(error,axis=0)
mean_duration = np.round(np.nanmean(duration,axis=0))

var_err = np.nanvar(error,axis=0)
var_duration = np.nanvar(duration,axis=0)

H_true = Laplace(mu=0,b=2,N=N).entropy()

# PLOTS


for i in range(num_trainings):
    min_sec = str(timedelta(seconds = mean_duration[i])))
    print(f'Batch: {batch_size[i]}, Mean training duration: {min_sec}')


plt.figure(1)
# plt.axhline(y=H_true,linestyle='--')
plt.plot(batch_size,mean_err)
plt.title("validation error per batch size")
plt.xlabel("batch size")
plt.ylabel("H(x) error")


plt.show()