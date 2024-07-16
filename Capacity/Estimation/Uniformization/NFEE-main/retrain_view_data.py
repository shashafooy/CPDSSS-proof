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

filepath = 'temp_data/retrain/15N_1M_train'
N=15

for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    N,_error,_duration = util.io.load(os.path.join(filepath, filename))
    _trainings = list(range(1,_error.shape[1]+1))

    # Initialize arrays
    if 'error' not in locals():
        error = np.empty((0,len(_trainings)))
        duration = np.empty((0,len(_trainings)))
        trainings = _trainings
       

    error,_ = viewData.align_and_concatenate(error,_error,(trainings),(_trainings))
    duration,_ = viewData.align_and_concatenate(duration,_duration,(trainings),(_trainings))
    
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

num_trainings = error.shape[1]
trainings = range(1,num_trainings+1)

# PLOTS

min_sec = []
for i in range(num_trainings):
    min_sec.append(str(timedelta(seconds = mean_duration[i])))
    print(f'Mean training {i+1} duration: {min_sec[i]}')
tot_sec = np.cumsum(mean_duration)[-1].astype(np.int32).tolist()
print(f'Total train time: {str(timedelta(seconds = tot_sec))}')


plt.figure(1)
# plt.axhline(y=H_true,linestyle='--')
plt.plot(trainings,mean_err)
plt.title("validation error after retraining")
plt.xlabel("times model trained")
plt.ylabel("H(x) error")


plt.show()