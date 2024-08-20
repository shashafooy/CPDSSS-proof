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
max_T = 0
min_T = 0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = "temp_data/laplace_test/hidden_layers/15N_1M_knn"
N = 15

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    if not os.path.isfile(os.path.join(filepath, filename)):
        continue
    filename = os.path.splitext(filename)[0]  # remove extention
    _layers, _nodes, _H_unif_KL, _H_unif_KSG, _H_KL, _H_KSG = util.io.load(
        os.path.join(filepath, filename)
    )

    # Initialize arrays
    if "layers" not in locals():
        layers = _layers
        N_layers = len(layers)
        nodes = _nodes
        N_nodes = len(nodes)
        H_unif_KL = np.empty((0, N_layers, N_nodes))
        H_unif_KSG = np.empty((0, N_layers, N_nodes))
        H_KL = np.empty((0))
        H_KSG = np.empty((0))

    H_unif_KL, _ = viewData.align_and_concatenate(
        H_unif_KL, _H_unif_KL, (layers, nodes), (_layers, _nodes)
    )
    H_unif_KSG, (layers, nodes) = viewData.align_and_concatenate(
        H_unif_KSG, _H_unif_KSG, (layers, nodes), (_layers, _nodes)
    )
    H_KL, _ = viewData.align_and_concatenate(H_KL, _H_KL, (layers, nodes), (_layers, _nodes))
    H_KSG, _ = viewData.align_and_concatenate(H_KSG, _H_KSG, (layers, nodes), (_layers, _nodes))

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

std_unif_KL = np.nanstd(H_unif_KL, axis=0)
std_unif_KSG = np.nanstd(H_unif_KSG, axis=0)
std_KL = np.nanstd(H_KL, axis=0)
std_KSG = np.nanstd(H_KSG, axis=0)

H_true = Laplace(mu=0, b=2, N=N).entropy()


MSE_unif_KL = np.nanmean((H_unif_KL - H_true) ** 2, axis=0)
MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true) ** 2, axis=0)
MSE_KL = np.nanmean((H_KL - H_true) ** 2, axis=0)
MSE_KSG = np.nanmean((H_KSG - H_true) ** 2, axis=0)

RMSE_unif_KL = np.sqrt(MSE_unif_KL)
RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
RMSE_KL = np.sqrt(MSE_KL)
RMSE_KSG = np.sqrt(MSE_KSG)

err_unif_KL = np.abs(H_unif_KL - H_true)
err_unif_KSG = np.abs(H_unif_KSG - H_true)
err_KL = np.abs(H_KL - H_true)
err_KSG = np.abs(H_KSG - H_true)


# PLOTS

# entropy

fig, ax = plt.subplots(1, len(layers))
fig.suptitle("Entropy for different hidden layers, N={}".format(N))
y_lim = np.zeros(len(layers))
y_lim = np.zeros((len(layers), 2))
for i in range(len(layers)):

    ax[i].axhline(y=H_true, color="r", linestyle="-")
    ax[i].axhline(y=mean_KL, color="g", linestyle="--")
    ax[i].axhline(y=mean_KSG, color="b", linestyle="--")

    ax[i].plot(
        nodes,
        mean_unif_KL[i, :],
        "--^",
        nodes,
        mean_unif_KSG[i, :],
        "--v",
    )
    # plt.yscale("log")
    ax[i].set_title("num hidden layers = {}".format(layers[i]))
    ax[i].legend(["True H(x)", "KL H(x)", "KSG H(x)", "Uniform KL H(x)", "Uniform KSG H(x)"])
    ax[i].set_xlabel("Nodes per Layer")
    ax[i].set_ylabel("H(x)")

    y_lim[i] = ax[i].get_ylim()

for axx in ax:
    axx.set_ylim([np.min(y_lim), np.max(y_lim)])


# Absolute error
# plt.figure(1)
# plt.plot(N_range,np.abs(H_true - H_unif_KL_mean),'--^',
#          N_range,np.abs(H_true - H_unif_KSG_mean),'--v',
#          N_range,np.abs(H_true - H_KL_mean),'--x',
#          N_range,np.abs(H_true - H_KSG_mean),'--o')
# plt.yscale("log")
# plt.title("Entropy Error")
# plt.legend(["Uniform KL","Uniform KSG","KL","KSG"])
# plt.xlabel("N dimensions")
# plt.ylabel("log error")

# MSE
fig, ax = plt.subplots(1, len(layers))
fig.suptitle("MSE for different hidden layers, N={}".format(N))


for i in range(len(layers)):
    ax[i].axhline(y=MSE_KL, color="g", linestyle="--")
    ax[i].axhline(y=MSE_KSG, color="b", linestyle="--")

    ax[i].plot(
        nodes,
        MSE_unif_KL[i, :],
        "--^",
        nodes,
        MSE_unif_KSG[i, :],
        "--v",
    )
    # plt.yscale("log")
    ax[i].set_title("num hidden layers = {}".format(layers[i]))
    ax[i].legend(["KL", "KSG", "Uniform KL", "Uniform KSG"])
    ax[i].set_xlabel("Nodes per Layer")
    ax[i].set_ylabel("H(x) MSE")
    ax[i].set_yscale("log")

    y_lim[i] = ax[i].get_ylim()


for axx in ax:
    axx.set_ylim([np.min(y_lim), np.max(y_lim)])


fig, ax = plt.subplots(1, len(layers))
fig.suptitle("RMSE for different hidden layers, N={}".format(N))


for i in range(len(layers)):
    ax[i].axhline(y=RMSE_KL, color="g", linestyle="--")
    ax[i].axhline(y=RMSE_KSG, color="b", linestyle="--")
    # ax[i].scatter(np.tile(120,(H_unif_KL.shape[0],1)),np.abs(H_KL - H_true))

    nodes_matrix = np.tile(nodes, (H_unif_KL.shape[0], 1))
    ax[i].scatter(nodes_matrix, err_unif_KL[:, i], marker="x")
    ax[i].scatter(nodes_matrix, err_unif_KSG[:, i])

    # plt.yscale("log")
    ax[i].set_title("num hidden layers = {}".format(layers[i]))
    ax[i].legend(["KL", "KSG", "Uniform KL", "Uniform KSG"])
    ax[i].set_xlabel("Nodes per Layer")
    ax[i].set_ylabel("H(x) RMSE")
    ax[i].set_yscale("log")

    y_lim[i] = ax[i].get_ylim()


for axx in ax:
    axx.set_ylim([np.min(y_lim), np.max(y_lim)])

fig.tight_layout()

plt.show()
