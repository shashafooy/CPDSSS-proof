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
L = 2
N_range = [2, 4, 6]

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True


MI = []
H_hxc = []
H_xxc = []
H_joint = []
H_cond = []

H_h = []
T_range = []

# for i, N in enumerate(N_range):
for i, N in enumerate(N_range):
    d0 = int(N / 2)
    d1 = d0
    # util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename))
    base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_d0d1({d0},{d1})/"
    filepath = base_path + "pretrained_model"

    _T_range, data = viewData.read_data(filepath, REMOVE_OUTLIERS)
    T_range.append(_T_range)
    MI.append(viewData.Data(data[0]))
    H_hxc.append(viewData.Data(data[1]))
    H_xxc.append(viewData.Data(data[2]))
    H_joint.append(viewData.Data(data[3]))
    H_cond.append(viewData.Data(data[4]))

    MI[-1].mean = H_hxc[-1].mean + H_xxc[-1].mean - H_joint[-1].mean - H_cond[-1].mean

    """
    Max capacity
    """
    from simulators.CPDSSS_models import CPDSSS

    sim_model = CPDSSS(1, N, L)
    H_h.append(sim_model.chan_entropy())

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
    ax1.set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")
    ax1.set_xlabel(r"$T$ Transmissions")
    ax1.set_ylabel(r"Entropy")

    (line,) = ax2.plot(temp_range, np.cumsum(_MI_mean), label=rf"$N={N}$")
    ax2.axhline(y=H_h[i], linestyle="dashed", color=line.get_color())
    ax2.set_title(r"Total $I(\mathbf{h},\mathbf{X})$")
    ax2.set_xlabel(r"$T$ Transmissions")
    ax2.set_ylabel(r"Entropy")
    ax2.text(
        x=1,
        y=H_h[i] + 0.02,
        s=rf"$H(\mathbf{{h}})$",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

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
