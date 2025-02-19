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
max_T = 0
min_T = 0

# N_range = [2, 4, 6]
# L = 2
N = 6
L = 3
d0 = 3
d1 = 3
N_range = [2, 6]

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True
USE_KSG = True


MI = []
H_HX = []
H_X = []

MI_kl = []
MI_ksg = []
H_HX_kl = []
H_HX_ksg = []
H_X_kl = []
H_X_ksg = []

T_range = []
H_h = []

# for i, N in enumerate(N_range):
for i, N in enumerate(N_range):
    d0 = int(N / 2)
    d1 = int(N / 2)

    # util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename))
    base_path = f"temp_data/CPDSSS_data/MI(h,X)/conditional/N{N}_d0d1({d0},{d1})/"

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

    """
    Max capacity
    """
    from simulators.CPDSSS_models import CPDSSS_Cond

    sim_model = CPDSSS_Cond(1, N, d0=d0, d1=d1)
    H_h.append(sim_model.chan_entropy())

for HX, X, N, t_range in zip(H_HX, H_X, N_range, T_range):
    fig, ax = plt.subplots(1, 2)
    x = np.full(HX.data.shape, t_range)
    ax[0].scatter(x, HX.data)
    ax[1].scatter(x, X.data)
    fig.suptitle(f"Cond scatter for N={N}")


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# fig4, ax4 = plt.subplots(2, 1)

# for i, N in enumerate(N_range):
for i, N in enumerate(N_range):
    """Plot individual and cumulative Mutual Information"""

    _MI_mean = MI[i].mean
    # fig1.suptitle("N={}, L={}".format(N, L))
    # temp_range = range(1, max(T_range[i]) + 1)
    temp_range = np.insert(T_range[i], 0, 1)
    _MI_mean = np.insert(_MI_mean, 0, 0)  # start at 0 MI

    ax1.plot(temp_range, _MI_mean, label=rf"$N={N}$")

    (line,) = ax2.plot(temp_range, np.cumsum(_MI_mean), label=rf"$N={N}$")


ax1.set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")
ax1.set_xlabel(r"$T$ Transmissions")
ax1.set_ylabel(r"Entropy")

ax2.axhline(y=H_h[i], linestyle="dashed", color=line.get_color())
ax2.set_title(r"Total $I(\mathbf{h},\mathbf{X})$")
ax2.set_xlabel(r"$T$ Transmissions")
ax2.set_ylabel(r"Entropy")
ax2.text(
    x=1,
    y=H_h[i] + 0.02,
    s=rf"$H(\mathbf{{h}})$",
    fontsize=14,
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] + 0.2))


# """Plot Mutual Information, but include bars showing variance"""

# # fig4.suptitle("N={}, L={}".format(N, L))
# yerr = np.insert(MI_std[i], 0, 0)
# ax4[0].errorbar(temp_range, _MI_mean, yerr=yerr, label=rf"$N={N}$")
# ax4[0].set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")
# ax4[0].set_xlabel(r"$T$")
# line, caps, bars = ax4[1].errorbar(
#     temp_range, np.cumsum(_MI_mean), yerr=np.cumsum(yerr), label=rf"$N={N}$"
# )
# ax4[1].set_title(r"Total $I(\mathbf{h},\mathbf{X})$")
# ax4[1].set_xlabel(r"T")
# ax4[1].axhline(
#     y=H_h[i], linestyle="dashed", color=line.get_color()
# )  # , label=rf"$H(\mathbf{{h}}),N={N}$")
# # Add text to the dashed line
# ax4[1].text(
#     x=1,
#     y=H_h[i] + 0.02,
#     s=rf"$H(\mathbf{{h}})$",
#     fontsize=10,
#     verticalalignment="bottom",
#     horizontalalignment="left",
# )


ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

fig1.tight_layout()
fig2.tight_layout()
plt.show()
