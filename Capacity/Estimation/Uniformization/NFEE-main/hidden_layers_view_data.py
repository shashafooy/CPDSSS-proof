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
max_T=0
min_T=0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = 'temp_data/laplace_test/hidden_layers'
N=8

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    _layers,_nodes,_H_unif_KL,_H_unif_KSG,_H_KL,_H_KSG = util.io.load(os.path.join(filepath, filename))

    # Initialize arrays
    if 'layers' not in locals():
        layers = _layers
        N_layers = len(layers)
        nodes = _nodes
        N_nodes = len(nodes)
        H_unif_KL = np.empty((0,N_layers,N_nodes))
        H_unif_KSG = np.empty((0,N_layers,N_nodes))
        H_KL = np.empty((0))        
        H_KSG = np.empty((0))        
       

    H_unif_KL,_ = viewData.align_and_concatenate(H_unif_KL,_H_unif_KL,(layers,nodes),(_layers,_nodes))
    H_unif_KL,_ = viewData.align_and_concatenate(H_unif_KSG,_H_unif_KSG,(layers,nodes),(_layers,_nodes))
    H_KL,_ = viewData.align_and_concatenate(H_KL,_H_KL,(layers,nodes),(_layers,_nodes))
    H_KSG,(layers,nodes) = viewData.align_and_concatenate(H_KSG,_H_KSG,(layers,nodes),(_layers,_nodes))
    
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


H_unif_KL_mean = np.nanmean(H_unif_KL,axis=0)
H_unif_KSG_mean = np.nanmean(H_unif_KSG,axis=0)
H_KL_mean = np.nanmean(H_KL,axis=0)
H_KSG_mean = np.nanmean(H_KSG,axis=0)

H_unif_KL_std = np.nanstd(H_unif_KL,axis=0)
H_unif_KSG_std = np.nanstd(H_unif_KSG,axis=0)
H_KL_std = np.nanstd(H_KL,axis=0)
H_KSG_std = np.nanstd(H_KSG,axis=0)

H_true = Laplace(mu=0,b=2,N=N).entropy()




MSE_unif_KL = np.nanmean((H_unif_KL - H_true)**2,axis=0)
MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true)**2,axis=0)
MSE_KL = np.nanmean((H_KL - H_true)**2,axis=0)
MSE_KSG = np.nanmean((H_KSG - H_true)**2,axis=0)

RMSE_unif_KL = np.sqrt(MSE_unif_KL)
RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
RMSE_KL = np.sqrt(MSE_KL)
RMSE_KSG = np.sqrt(MSE_KSG)


# PLOTS

#entropy
plt.figure(0)
plt.plot(N_range,H_true,'-',
         N_range,H_unif_KL_mean,'--^',
         N_range,H_unif_KSG_mean,'--v',
         N_range,H_KL_mean,'--x',
         N_range,H_KSG_mean,'--o')
plt.yscale("log")
plt.title("Entropy of laplace distribution")
plt.legend(["True H(x)","Uniform KL H(x)","Uniform KSG H(x)","KL H(x)","KSG H(x)"])
plt.xlabel("N dimensions")
plt.ylabel("H(x)")

#Absolute error
plt.figure(1)
plt.plot(N_range,np.abs(H_true - H_unif_KL_mean),'--^',
         N_range,np.abs(H_true - H_unif_KSG_mean),'--v',
         N_range,np.abs(H_true - H_KL_mean),'--x',
         N_range,np.abs(H_true - H_KSG_mean),'--o')
plt.yscale("log")
plt.title("Entropy Error")
plt.legend(["Uniform KL","Uniform KSG","KL","KSG"])
plt.xlabel("N dimensions")
plt.ylabel("log error")

#MSE
plt.figure(2)
plt.plot(N_range,MSE_unif_KL,'--^',
         N_range,MSE_unif_KSG,'--v',
         N_range,MSE_KL,'--x',
         N_range,MSE_KSG,'--o')
plt.yscale("log")
plt.title("Entropy MSE Error")
plt.legend(["Uniform KL","Uniform KSG","KL","KSG"])
plt.xlabel("N dimensions")
plt.ylabel("log MSE")

#RMSE
plt.figure(3)
plt.plot(N_range,RMSE_unif_KL,'--^',
         N_range,RMSE_unif_KSG,'--v',
         N_range,RMSE_KL,'--x',
         N_range,RMSE_KSG,'--o')
plt.yscale("log")
plt.title("Entropy RMSE Error")
plt.legend(["Uniform KL","Uniform KSG","KL","KSG"])
plt.xlabel("N dimensions")
plt.ylabel("log RMSE")

plt.figure(4)
plt.plot(N_range,H_unif_KL_std,'--^',
         N_range,H_unif_KSG_std,'--v',
         N_range,H_KL_std,'--x',
         N_range,H_KSG_std,'--o')
plt.yscale("log")
plt.title("Entropy std")
plt.legend(["Uniform KL","Uniform KSG","KL","KSG"])
plt.xlabel("N dimensions")
plt.ylabel("log std")

plt.show()