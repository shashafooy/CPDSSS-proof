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
N_range = [2, 4, 6]

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True


MI_mean = []
H_hxc_mean = []
H_xxc_mean = []
H_joint_mean = []
H_cond_mean = []

MI_std = []
H_hxc_std = []
H_xxc_std = []
H_joint_std = []
H_cond_std = []

H_h = []
T_range = []

# for i, N in enumerate(N_range):
for i, N in enumerate(N_range):

    # util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename))
    base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_L{L}/"
    filepath = base_path + "pretrained_model"

    # filepath=filepaths[1]
    # idx=0
    # for idx,filepath in enumerate(filepaths):
    for filename in os.listdir(filepath):
        filename = os.path.splitext(filename)[0]  # remove extention
        _T_range, MI_cum, H_gxc_cum, H_xxc_cum, H_joint_cum, H_cond_cum, completed_iter = (
            util.io.load(os.path.join(filepath, filename))
        )
        iter = range(0, completed_iter + 1)

        if "MI_tot" not in locals():
            MI_tot = np.empty((0, np.size(_T_range)))
            H_gxc_tot = np.empty((0, np.size(_T_range)))
            H_xxc_tot = np.empty((0, np.size(_T_range)))
            H_joint_tot = np.empty((0, np.size(_T_range)))
            H_cond_tot = np.empty((0, np.size(_T_range)))
            old_range = _T_range

        MI_tot, _ = viewData.align_and_concatenate(MI_tot, MI_cum, old_range, _T_range)
        H_gxc_tot, _ = viewData.align_and_concatenate(H_gxc_tot, H_gxc_cum, old_range, _T_range)
        H_xxc_tot, _ = viewData.align_and_concatenate(H_xxc_tot, H_xxc_cum, old_range, _T_range)
        H_joint_tot, _ = viewData.align_and_concatenate(
            H_joint_tot, H_joint_cum, old_range, _T_range
        )
        H_cond_tot, old_range = viewData.align_and_concatenate(
            H_cond_tot, H_cond_cum, old_range, _T_range
        )

    """
    Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
    """
    if REMOVE_OUTLIERS:
        MI_tot = viewData.remove_outlier(data=MI_tot, num_std=3)
        H_gxc_tot = viewData.remove_outlier(data=H_gxc_tot, num_std=3)
        H_xxc_tot = viewData.remove_outlier(data=H_xxc_tot, num_std=3)
        H_joint_tot = viewData.remove_outlier(data=H_joint_tot, num_std=3)
        H_cond_tot = viewData.remove_outlier(data=H_cond_tot, num_std=3)

    # MI_mean.append(np.nanmean(MI_tot, axis=0))
    H_hxc_mean.append(np.nanmean(H_gxc_tot, axis=0))
    H_xxc_mean.append(np.nanmean(H_xxc_tot, axis=0))
    H_joint_mean.append(np.nanmean(H_joint_tot, axis=0))
    H_cond_mean.append(np.nanmean(H_cond_tot, axis=0))

    MI_std.append(np.nanstd(MI_tot, axis=0))
    H_hxc_std.append(np.nanstd(H_gxc_tot, axis=0))
    H_xxc_std.append(np.nanstd(H_xxc_tot, axis=0))
    H_joint_std.append(np.nanstd(H_joint_tot, axis=0))
    H_cond_std.append(np.nanstd(H_cond_tot, axis=0))

    # Potentially more accurate taking into account each mean value
    MI_mean_sum = H_hxc_mean[-1] + H_xxc_mean[-1] - H_joint_mean[-1] - H_cond_mean[-1]
    MI_mean.append(MI_mean_sum)

    T_range.append(old_range)

    del MI_tot

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

    _MI_mean = MI_mean[i]
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
