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

H_y_given_x_MAF = H_xy_MAF.mean - H_x_MAF.mean
H_y_given_x_knn = H_xy_kl_ksg.mean - H_x_kl_ksg.mean


# entropy values
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_MAF, "--")
ax1.plot(T_range, H_y_given_x_knn, "--*")
ax1.plot(T_range, H_cond_MAF.mean, "--x")
ax1.legend(["H(y|x)", "MAF KL", "MAF KSG", "KL", "KSG", "cond MAF KL", "cond MAF KSG"])


# entropy values
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_MAF, "--")
ax1.plot(T_range, H_y_given_x_knn, "--*")
ax1.plot(T_range, H_cond_MAF.mean, "--x")
ax1.legend(["H(y|x)", "MAF KL", "MAF KSG", "KL", "KSG", "cond MAF KL", "cond MAF KSG"])
fig.tight_layout()

# Normalizedentropy values
fig, ax1 = plt.subplots()
ax1.hlines(1, 1, 10)
# ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_MAF / H_y_given_x_true.mean[:, np.newaxis], "--")
ax1.plot(T_range, H_y_given_x_knn / H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_cond_MAF.mean / H_y_given_x_true.mean[:, np.newaxis], "--x")
ax1.legend(["norm entropy", "MAF KL", "MAF KSG", "KL", "KSG", "cond MAF KL", "cond MAF KSG"])

# error
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_MAF - H_y_given_x_true.mean[:, np.newaxis], "--")
ax1.plot(T_range, H_y_given_x_knn - H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_cond_MAF.mean - H_y_given_x_true.mean[:, np.newaxis], "--x")
ax1.legend(["MAF KL", "MAF KSG", "KL", "KSG", "cond MAF KL", "cond MAF KSG"])
plt.show()
