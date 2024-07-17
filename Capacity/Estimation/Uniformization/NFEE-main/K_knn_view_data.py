import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

# plt.rcParams['text.usetex']=True




"""
Load and combine all datasets
"""
max_T=0
min_T=0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = 'temp_data/k_knn_test/1M_knn_no_tol'
N=15

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    _k_list,_H_unif_KL,_H_unif_KSG,_H_KL,_H_KSG = util.io.load(os.path.join(filepath, filename))
    _k_list = np.array(_k_list)
    # Initialize arrays
    if 'k_list' not in locals():
        k_list = _k_list
        H_unif_KL = np.empty((0,len(k_list)))
        H_unif_KSG = np.empty((0,len(k_list)))
        H_KL = np.empty((0,len(k_list)))        
        H_KSG = np.empty((0,len(k_list)))        
       

    H_unif_KL,_ = viewData.align_and_concatenate(H_unif_KL,_H_unif_KL,k_list,_k_list)
    H_unif_KSG,_ = viewData.align_and_concatenate(H_unif_KSG,_H_unif_KSG,k_list,_k_list)
    H_KL,_ = viewData.align_and_concatenate(H_KL,_H_KL,k_list,_k_list)
    H_KSG,k_list = viewData.align_and_concatenate(H_KSG,_H_KSG,k_list,_k_list)

# k_list=k_list[0]    
viewData.clean_data(H_unif_KL)
viewData.clean_data(H_unif_KSG)
viewData.clean_data(H_KL)
viewData.clean_data(H_KSG)

# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    H_unif_KL = viewData.remove_outlier(H_unif_KL)
    H_unif_KSG = viewData.remove_outlier(H_unif_KSG)
    H_KL = viewData.remove_outlier(H_KL)
    H_KSG = viewData.remove_outlier(H_KSG)


mean_unif_KL = np.nanmean(H_unif_KL,axis=0)
mean_unif_KSG = np.nanmean(H_unif_KSG,axis=0)
mean_KL = np.nanmean(H_KL,axis=0)
mean_KSG = np.nanmean(H_KSG,axis=0)

std_unif_KL = np.nanstd(H_unif_KL,axis=0)
std_unif_KSG = np.nanstd(H_unif_KSG,axis=0)
std_KL = np.nanstd(H_KL,axis=0)
std_KSG = np.nanstd(H_KSG,axis=0)

H_true = Laplace(mu=0,b=2,N=N).entropy()




MSE_unif_KL = np.nanmean((H_unif_KL - H_true)**2,axis=0)
MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true)**2,axis=0)
MSE_KL = np.nanmean((H_KL - H_true)**2,axis=0)
MSE_KSG = np.nanmean((H_KSG - H_true)**2,axis=0)

RMSE_unif_KL = np.sqrt(MSE_unif_KL)
RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
RMSE_KL = np.sqrt(MSE_KL)
RMSE_KSG = np.sqrt(MSE_KSG)

err_unif_KL = np.abs(H_unif_KL - H_true)
err_unif_KSG = np.abs(H_unif_KSG - H_true)
err_KL = np.abs(H_KL - H_true)
err_KSG = np.abs(H_KSG - H_true)


# PLOTS

#entropy

plt.figure(1)
plt.axhline(y=H_true,linestyle='--')
plt.plot(k_list,mean_unif_KL,'o')
plt.plot(k_list,mean_unif_KSG,'s')
plt.plot(k_list,mean_KL,'o')
plt.plot(k_list,mean_KSG,'s')
plt.yscale("log")    
plt.title("H(x) for different K knn values")
plt.legend(["True H(x)","unif KL H(x)","unif KSG H(x)","KL H(x)","KSG H(x)"])
plt.xlabel("K value for knn")
plt.ylabel("H(x)")

plt.figure(2)
plt.plot(k_list,RMSE_unif_KL,'o',)
plt.plot(k_list,RMSE_unif_KSG,'s',)
plt.plot(k_list,RMSE_KL,'o',)
plt.plot(k_list,RMSE_KSG,'s',)
plt.yscale("log")    
plt.title("RMSE for different K knn values")
plt.legend(["unif KL H(x)","unif KSG H(x)","KL H(x)","KSG H(x)"])
plt.xlabel("K value for knn")
plt.ylabel("RMSE H(x)")





plt.show()