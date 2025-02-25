from _utils import set_sys_path

set_sys_path()
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

plt.rcParams["text.usetex"] = True


"""
Load and combine all datasets
"""
max_T = 0
min_T = 0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = "temp_data/num_hidden_nodes/15N_1M_knn"

# load previously saved KNN for N=15 1M samples
N, H_KL, H_KSG = util.io.load("temp_data/KL_KSG/N15_1M/kl_ksg")
# N=15

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename = os.path.splitext(filename)[0]  # remove extention
    _hidden_nodes, _H_unif_KL, _H_unif_KSG, _, _ = util.io.load(os.path.join(filepath, filename))

    # Initialize arrays
    if "H_unif_KL" not in locals():
        hidden_nodes = _hidden_nodes
        tot_hidden = len(hidden_nodes)
        H_unif_KL = np.empty((0, tot_hidden))
        H_unif_KSG = np.empty((0, tot_hidden))

    H_unif_KL, _ = viewData.align_and_concatenate(
        H_unif_KL, _H_unif_KL, (hidden_nodes), (_hidden_nodes)
    )
    H_unif_KSG, (hidden_nodes) = viewData.align_and_concatenate(
        H_unif_KSG, _H_unif_KSG, (hidden_nodes), (_hidden_nodes)
    )


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


mean_unif_KL = np.nanmean(H_unif_KL, axis=0)
mean_unif_KSG = np.nanmean(H_unif_KSG, axis=0)
mean_KL = np.nanmean(H_KL, axis=0)
mean_KSG = np.nanmean(H_KSG, axis=0)

std_unif_KL = np.nanvar(H_unif_KL, axis=0)
std_unif_KSG = np.nanvar(H_unif_KSG, axis=0)
std_KL = np.nanvar(H_KL, axis=0)
std_KSG = np.nanvar(H_KSG, axis=0)

H_true = Laplace(mu=0, b=2, N=N).entropy()


MSE_unif_KL = np.nanmean((H_unif_KL - H_true) ** 2, axis=0)
MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true) ** 2, axis=0)
MSE_KL = np.nanmean((H_KL - H_true) ** 2, axis=0)
MSE_KSG = np.nanmean((H_KSG - H_true) ** 2, axis=0)

RMSE_unif_KL = np.sqrt(MSE_unif_KL)
RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
RMSE_KL = np.sqrt(MSE_KL)
RMSE_KSG = np.sqrt(MSE_KSG)

err_unif_KL = np.nanmean(np.abs(H_unif_KL - H_true), axis=0)
err_unif_KSG = np.nanmean(np.abs(H_unif_KSG - H_true), axis=0)
err_KL = np.nanmean(np.abs(H_KL - H_true), axis=0)
err_KSG = np.nanmean(np.abs(H_KSG - H_true), axis=0)


# PLOTS

# entropy

plt.figure(1)
plt.axhline(y=H_true, linestyle="--")
plt.plot(hidden_nodes, mean_unif_KL, "bo")
plt.plot(hidden_nodes, mean_unif_KSG, "rs")
plt.axhline(mean_KL, color="b")
plt.axhline(mean_KSG, color="r")
plt.yscale("log")
plt.title("H(x) for different number of hidden nodes")
plt.legend(["True H(x)", "unif KL H(x)", "unif KSG H(x)", "KL H(x)", "KSG H(x)"])
plt.xlabel("hidden nodes per layer")
plt.ylabel("H(x)")

plt.figure(2)
plt.plot(
    hidden_nodes,
    RMSE_unif_KL,
    "bo",
)
plt.plot(
    hidden_nodes,
    RMSE_unif_KSG,
    "rs",
)
plt.axhline(RMSE_KL, color="b")
plt.axhline(RMSE_KSG, color="r")
plt.yscale("log")
plt.title("RMSE for different number of hidden nodes")
plt.legend(["unif KL H(x)", "unif KSG H(x)", "KL H(x)", "KSG H(x)"])
plt.xlabel("hidden nodes per layer")
plt.ylabel("RMSE H(x)")

plt.figure(3)
plt.plot(
    hidden_nodes,
    err_unif_KL,
    "bo",
)
plt.plot(
    hidden_nodes,
    err_unif_KSG,
    "rs",
)
plt.axhline(err_KL, color="b")
plt.axhline(err_KSG, color="r")
plt.yscale("log")
plt.title("error for different number of hidden nodes")
plt.legend(["unif KL H(x)", "unif KSG H(x)", "KL H(x)", "KSG H(x)"])
plt.xlabel("hidden nodes per layer")
plt.ylabel("error H(x)")


plt.show()
