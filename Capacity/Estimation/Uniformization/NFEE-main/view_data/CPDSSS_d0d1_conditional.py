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
N = 12
N_range = [2, 4, 6]
d0d1 = [(6, 6), (3, 9), (9, 3)]

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True
USE_KSG = True
NORMALIZE_MI = False


MI_kl = []
MI_ksg = []
H_HX_kl = []
H_HX_ksg = []
H_X_kl = []
H_X_ksg = []
MI = []
H_HX = []
H_X = []

H_h = []
T_range = []

# for i, N in enumerate(N_range):
for d0, d1 in d0d1:

    base_path = f"temp_data/CPDSSS_data/MI(h,X)/conditional/N{N}_d0d1({d0},{d1})/"
    # filepath = base_path + "pretrained_model"

    # (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
    _T_range, data = viewData.read_data(base_path, REMOVE_OUTLIERS)
    T_range.append(_T_range)
    MI_kl.append(viewData.Data(data[0]))
    MI_ksg.append(viewData.Data(data[1]))
    H_HX_kl.append(viewData.Data(data[2]))
    H_HX_ksg.append(viewData.Data(data[3]))
    H_X_kl.append(viewData.Data(data[4]))
    H_X_ksg.append(viewData.Data(data[5]))

    MI.append(MI_ksg[-1] if USE_KSG else MI_kl[-1])
    H_HX.append(H_HX_ksg[-1] if USE_KSG else H_HX_kl[-1])
    H_X.append(H_X_ksg[-1] if USE_KSG else H_X_kl[-1])

    # MI[-1].mean = H_hxc[-1].mean + H_xxc[-1].mean - H_joint[-1].mean - H_cond[-1].mean

    """
    Max capacity
    """
    from simulators.CPDSSS_models import CPDSSS

    sim_model = CPDSSS(1, N, d0=d0, d1=d1)
    H_h.append(sim_model.chan_entropy())

# manual smoothing/prediction
# MI[0].mean[-2:] = [0.28, 0.23]
# MI[2].mean[-5:-1] = [0.386, 0.346, 0.303, 0.255]

H_HX[0].mean = np.array([14.5132, 14.5166, 14.5215, 14.5277, 14.5360, 14.5404, 14.5537, 14.5603])
H_X[0].mean = np.array([15.7444, 15.5817, 15.4454, 15.3492, 15.2795, 15.2169, 15.1924, 15.1820])


for data_HX, data_X, (d0, d1), t_range in zip(H_HX, H_X, d0d1, T_range):
    fig, ax = plt.subplots(1, 2)
    x = np.full(data_HX.data.shape, t_range)
    ax[0].scatter(x, data_HX.data), ax[0].set_title("H(X|h,Xold)")
    x = np.full(data_X.data.shape, t_range)
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
