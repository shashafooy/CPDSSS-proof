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


ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_MAF, "--")
ax1.plot(T_range, H_y_given_x_knn, "--*")
ax1.plot(T_range, H_cond_MAF.mean, "--x")


for data_HX, data_X, (d0, d1), t_range in zip(H_HX, H_X, d0d1, T_range):
    fig, ax = plt.subplots(1, 2)
    x = np.full(data_HX.data.shape, t_range)
    ax[0].scatter(x, data_HX.data), ax[0].set_title("H(X|h,Xold)")
    ax[1].scatter(x, data_X.data), ax[1].set_title("H(X,Xold)")
    fig.suptitle(f"Cond scatter for d_0={d0}, d_1={d1}")


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# fig4, ax4 = plt.subplots(2, 1)

# for i, N in enumerate(N_range):
for i, (d0, d1) in enumerate(d0d1):
    """Plot individual and cumulative Mutual Information"""

    _MI_mean = MI[i].mean / H_h[i] if NORMALIZE_MI else MI[i].mean
    # fig1.suptitle("N={}, L={}".format(N, L))
    # temp_range = range(1, max(T_range[i]) + 1)
    temp_range = np.insert(T_range[i], 0, 1)
    _MI_mean = np.insert(_MI_mean, 0, 0)  # start at 0 MI

    ax1.plot(temp_range, _MI_mean, label=rf"$d_0={d0},d_1={d1}$")
    # ax1.set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")

    (line,) = ax2.plot(temp_range, np.cumsum(_MI_mean), label=rf"$d_0={d0},d_1={d1}$")

    # ax2.set_title(r"Total $I(\mathbf{h},\mathbf{X})$")

ax1.set_xlabel(r"$T$")
ax1.set_ylabel(r"Mutual Information")
ax2.set_xlabel(r"$T$")
ax2.set_ylabel(r"Mutual Information")


# ax2.axhline(y=H_h[i], linestyle="dashed", color="black")
ax2.text(
    x=1,
    y=H_h[i] + 0.02,
    s=rf"$H(\mathbf{{h}})$",
    fontsize=14,
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] + 0.2))


ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

fig1.tight_layout()
fig2.tight_layout()
plt.show()
