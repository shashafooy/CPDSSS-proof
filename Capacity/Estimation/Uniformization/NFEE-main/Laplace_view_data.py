import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace

plt.rcParams['text.usetex']=True


def append_data(old_data, old_T_range, new_data, new_T_range):
    """
    Adjust both datasets to match their T_range before appending
    """

    #Insert min_diff columns of NaN at beginning of matrix to align
    min_diff = min(old_T_range) - min(new_T_range)
    if(min_diff < 0): #Pad new data     
        new_data = np.insert(new_data,[0]*abs(min_diff),np.nan,axis=1)
    elif(min_diff > 0): #Pad old data
        old_data = np.insert(old_data,[0]*min_diff,np.nan,axis=1)

    #Insert max_diff columns of NaN at end of matrix to align
    max_diff = max(old_T_range) - max(new_T_range)
    if(max_diff > 0): #Pad new data
        new_data = np.insert(new_data,[new_data.shape[1]]*max_diff,np.nan,axis=1)
    elif(max_diff < 0): #Pad old data
        old_data = np.insert(old_data,[old_data.shape[1]]*abs(max_diff),np.nan,axis=1)
    
    #Update total data and range
    tot_data = np.append(old_data,new_data,axis=0)
    tot_T_range = range(min(old_T_range[0],new_T_range[0]), max(old_T_range[-1],new_T_range[-1])+1)

    return tot_data, tot_T_range

def remove_outlier(data, num_std):
    num_outlier = 1
    while num_outlier > 0:
        std = np.nanstd(data,axis=0)
        mean = np.nanmean(data,axis=0)
        #check if any value outside of 3*std
        idx = (data > mean +num_std*std) |  (data < mean - num_std*std)
        num_outlier = np.count_nonzero(idx)
        data[idx]=np.NaN
    return data

"""
Load and combine all datasets
"""
max_T=0
min_T=0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = 'temp_data/laplace_test'


N=2
L=2
# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    _N_range,_H_unif_laplace,_H_KL_laplace,_MSE_uniform,_MSE_KL,_iter = util.io.load(os.path.join(filepath, filename))
    iter = range(0,_iter+1)

    # Initialize arrays
    if 'H_unif_laplace' not in locals():
        H_unif_laplace = np.empty((0,_H_unif_laplace.shape[1]))
        H_KL_laplace = np.empty((0,_H_KL_laplace.shape[1]))        
        N_range = _N_range


    H_unif_laplace,_ = append_data(H_unif_laplace,N_range,_H_unif_laplace[iter,:],_N_range)
    H_KL_laplace,N_range = append_data(H_KL_laplace,N_range,_H_KL_laplace[iter,:],_N_range)
    

H_unif_laplace[np.isinf]=np.nan
H_KL_laplace[np.isinf]=np.nan
# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    H_unif_laplace = remove_outlier(data=H_unif_laplace,num_std=3)
    H_KL_laplace = remove_outlier(data=H_KL_laplace,num_std=3)


H_unif_mean = np.nanmean(H_unif_laplace,axis=0)
H_KL_mean = np.nanmean(H_KL_laplace,axis=0)

MSE_uniform = np.empty(len(N_range))
MSE_KL = np.empty(len(N_range))
H_true = np.empty(len(N_range))

for i,N in enumerate(N_range):
    H_true[i] = Laplace(mu=0,b=2,N=N).entropy()

MSE_uniform = np.nanmean((H_unif_laplace - H_true)**2,axis=0)
MSE_KL = np.nanmean((H_KL_laplace - H_true)**2,axis=0)

plt.figure(0)
plt.plot(N_range,H_true,'-o',N_range,H_unif_mean,'--^',N_range,H_KL_mean,'--x')
plt.yscale("log")
plt.title("Entropy of laplace distribution")
plt.legend(["True H(x)","Uniform H(x)","KL H(x)"])
plt.xlabel("N dimensions")
plt.ylabel("H(x)")

plt.figure(1)
plt.plot(N_range,np.abs(H_true - H_unif_mean),'--^',N_range,abs(H_true - H_KL_mean),'--x')
plt.yscale("log")
plt.title("Entropy Error")
plt.legend(["Uniform error","KL error"])
plt.xlabel("N dimensions")
plt.ylabel("log error")

plt.show()