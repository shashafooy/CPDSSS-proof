from _utils import set_sys_path

set_sys_path()
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from misc_CPDSSS import viewData

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 14})

"""
Load and combine all datasets
"""
max_T = 8
min_T = 0

# N_range = [2, 4, 6]
# L = 2
N = 6

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True
USE_KSG = True
NORMALIZE_MI = False


# for i, N in enumerate(N_range):

base_path = f"temp_data/conditional_test/random_A"
# filepath = base_path + "pretrained_model"

# (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
_T_range, data = viewData.read_data(base_path, REMOVE_OUTLIERS)
T_range = _T_range
H_y_given_x_true = viewData.Data(data[0])
H_xy_MAF = viewData.Data(data[1])
H_xy_kl_ksg = viewData.Data(data[2])
H_x_MAF = viewData.Data(data[3])
H_x_kl_ksg = viewData.Data(data[4])
H_cond_MAF = viewData.Data(data[5])

# temp placeholder values. Model loss values
H_cond_MAF.mean[7] = 109.9792
H_cond_MAF.mean[7:, 1] = np.array([110.21987, 123.41146, 135.82276])

H_xy_MAF.mean[7:, 1] = np.array([178.98333, 201.179542, 228.37499])
H_x_MAF.mean[7:, 1] = np.array([68.10809, 76.621849, 85.13507])

H_y_given_x_MAF = H_xy_MAF.mean - H_x_MAF.mean
H_y_given_x_knn = H_xy_kl_ksg.mean - H_x_kl_ksg.mean


# entropy values
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_knn, "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1], "--o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1], "--x")
ax1.legend(["H(y|x)", "KL", "KSG", "MAF", "cond MAF"])

fig.tight_layout()

# Normalizedentropy values
fig, ax1 = plt.subplots()
ax1.axhline(y=1, linestyle="dashed", color="black")
# ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_knn / H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1] / H_y_given_x_true.mean, "-o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1] / H_y_given_x_true.mean, "--x")
ax1.legend(["norm entropy", "KL", "KSG", "MAF", "cond MAF"])

# error
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_knn - H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1] - H_y_given_x_true.mean, "--o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1] - H_y_given_x_true.mean, "--x")
ax1.legend(["KL", "KSG", "MAF", "cond MAF"])
plt.show()
