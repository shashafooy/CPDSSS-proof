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

filepath = "temp_data/laplace_test"


N = 2
L = 2
# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    if not os.path.isfile(os.path.join(filepath, filename)):
        continue
    filename = os.path.splitext(filename)[0]  # remove extention
    _N_range, _H_unif_KL, _H_unif_KSG, _H_KL, _H_KSG, _iter = util.io.load(
        os.path.join(filepath, filename)
    )
    iter = range(0, _iter + 1)

    # Initialize arrays
    if "N_range" not in locals():
        N_range = _N_range
        N_size = len(N_range)
        H_unif_KL = np.empty((0, N_size))
        H_unif_KSG = np.empty((0, N_size))
        H_KL = np.empty((0, N_size))
        H_KSG = np.empty((0, N_size))

    H_unif_KL, _ = viewData.append_data(H_unif_KL, N_range, _H_unif_KL, _N_range)
    H_unif_KSG, _ = viewData.append_data(H_unif_KSG, N_range, _H_unif_KSG, _N_range)
    H_KL, _ = viewData.append_data(H_KL, N_range, _H_KL, _N_range)
    H_KSG, N_range = viewData.append_data(H_KSG, N_range, _H_KSG, _N_range)

N_range = np.asarray(N_range[1:])
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

H_unif_KL = H_unif_KL[:, 1:] / N_range
H_unif_KSG = H_unif_KSG[:, 1:] / N_range
H_KL = H_KL[:, 1:] / N_range
H_KSG = H_KSG[:, 1:] / N_range


H_unif_KL_mean = np.nanmean(H_unif_KL, axis=0)
H_unif_KSG_mean = np.nanmean(H_unif_KSG, axis=0)
H_KL_mean = np.nanmean(H_KL, axis=0)
H_KSG_mean = np.nanmean(H_KSG, axis=0)

H_unif_KL_std = np.nanstd(H_unif_KL, axis=0)
H_unif_KSG_std = np.nanstd(H_unif_KSG, axis=0)
H_KL_std = np.nanstd(H_KL, axis=0)
H_KSG_std = np.nanstd(H_KSG, axis=0)

H_true = np.empty(len(N_range))
for i, N in enumerate(N_range):
    H_true[i] = Laplace(mu=0, b=2, N=N).entropy()
H_true = H_true / N_range

MSE_unif_KL = np.nanmean((H_unif_KL - H_true) ** 2, axis=0)
MSE_unif_KSG = np.nanmean((H_unif_KSG - H_true) ** 2, axis=0)
MSE_KL = np.nanmean((H_KL - H_true) ** 2, axis=0)
MSE_KSG = np.nanmean((H_KSG - H_true) ** 2, axis=0)

RMSE_unif_KL = np.sqrt(MSE_unif_KL)
RMSE_unif_KSG = np.sqrt(MSE_unif_KSG)
RMSE_KL = np.sqrt(MSE_KL)
RMSE_KSG = np.sqrt(MSE_KSG)

err_unif_KL = np.abs(H_true - H_unif_KL_mean)
err_unif_KSG = np.abs(H_true - H_unif_KSG_mean)
err_KL = np.abs(H_true - H_KL_mean)
err_KSG = np.abs(H_true - H_KSG_mean)

# PLOTS

# entropy
plt.figure(0)
plt.plot(N_range, H_true, "-")
plt.plot(N_range, H_unif_KL_mean, "--^")
plt.plot(N_range, H_unif_KSG_mean, "--v")
plt.plot(N_range, H_KL_mean, "--x")
plt.plot(N_range, H_KSG_mean, "--o")
plt.title("Normalized Entropy of the Laplace distribution")
plt.legend(["True", "KL + uniformization", "KSG + uniformization", "KL", "KSG"])
plt.xlabel("d dimensions")
plt.ylabel("H(x)/d")
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
