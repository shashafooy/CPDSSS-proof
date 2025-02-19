from _utils import set_sys_path

set_sys_path()
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 14})


"""
Load and combine all datasets
"""
max_T = 0
min_T = 0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = "temp_data/laplace_test/high_epoch"


N = 2
L = 2
# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
N_range, data = viewData.read_data(filepath, REMOVE_OUTLIERS)

H_unif_KL, H_unif_KSG, H_KL, H_KSG = [viewData.Data(data[i] / N_range) for i in range(0, 4)]


H_true = Laplace(0, 2, 1).entropy()
# for i, N in enumerate(N_range):
#     H_true[i] = Laplace(mu=0, b=2, N=N).entropy()
# H_true = H_true / N_range

# MSE_unif_KL = np.nanmean((H_unif_KL - H_true) ** 2, axis=0)
# MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true) ** 2, axis=0)
# MSE_KL = np.nanmean((H_KL - H_true) ** 2, axis=0)
# MSE_KSG = np.nanmean((H_KSG - H_true) ** 2, axis=0)

# RMSE_unif_KL = np.sqrt(MSE_unif_KL)
# RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
# RMSE_KL = np.sqrt(MSE_KL)
# RMSE_KSG = np.sqrt(MSE_KSG)

err_unif_KL = np.abs(H_true - H_unif_KL.mean)
err_unif_KSG = np.abs(H_true - H_unif_KSG.mean)
err_KL = np.abs(H_true - H_KL.mean)
err_KSG = np.abs(H_true - H_KSG.mean)

# PLOTS

# entropy
plt.figure(0)
plt.plot(N_range, H_true, "-")
plt.plot(N_range, H_unif_KL_mean, "--^")
plt.plot(N_range, H_unif_KSG_mean, "--v")
plt.plot(N_range, H_KL_mean, "--x")
plt.plot(N_range, H_KSG_mean, "--o")
# plt.title("Normalized Entropy of the Laplace distribution")
plt.legend(["True", "KL + uniformization", "KSG + uniformization", "KL", "KSG"])
plt.xlabel("d")
plt.ylabel("Normalized Entropy")
plt.xticks(N_range[::2])
plt.tight_layout()

# Absolute error
plt.figure(1)
plt.plot(N_range, err_unif_KL, "--^")
plt.plot(N_range, err_unif_KSG, "--v")
plt.plot(N_range, err_KL, "--x")
plt.plot(N_range, err_KSG, "--o")
plt.yscale("log")
plt.title("Entropy Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log error")
plt.xticks(N_range[::2])
plt.tight_layout()

# MSE
plt.figure(2)
plt.plot(N_range, MSE_unif_KL, "--^")
plt.plot(N_range, MSE_unif_KSG, "--v")
plt.plot(N_range, MSE_KL, "--x")
plt.plot(N_range, MSE_KSG, "--o")
plt.yscale("log")
plt.title("Entropy MSE Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log MSE")
plt.xticks(N_range[::2])
plt.tight_layout()

# RMSE
plt.figure(3)
plt.plot(N_range, RMSE_unif_KL, "--^")
plt.plot(N_range, RMSE_unif_KSG, "--v")
plt.plot(N_range, RMSE_KL, "--x")
plt.plot(N_range, RMSE_KSG, "--o")
plt.yscale("log")
plt.title("Entropy RMSE Error")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log RMSE")
plt.xticks(N_range[::2])
plt.tight_layout()

# STD
plt.figure(4)
plt.plot(N_range, H_unif_KL_std, "--^")
plt.plot(N_range, H_unif_KSG_std, "--v")
plt.plot(N_range, H_KL_std, "--x")
plt.plot(N_range, H_KSG_std, "--o")
plt.yscale("log")
plt.title("Entropy std")
plt.legend(["Uniform KL", "Uniform KSG", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("log std")
plt.xticks(N_range[::2])
plt.tight_layout()


plt.show()
