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

N = 6
# L = 2
# d0=N/L
# d1=N-d0

d0 = 3
d1 = N - d0

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = False

# util.io.save((T_range, MI_cum,H_hxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename))
base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_d0d1({d0},{d1})/"

filepath = base_path + "pretrained_model"
# filepath = base_path + "high_epoch"
# filepath = base_path + "Smooth_MI"
# filepath = base_path + "N9_coarse-fine_experiment"

# filepath= 'temp_data/CPDSSS_data/N2_L2/50k_tol_0.1_patience_10'
# filepath= 'temp_data/CPDSSS_data/N2_L2/50k_samples'
# filepath = base_path + 'NlogN_10k_scaling'
# filepaths = [base_path + 'NlogN_10k_scaling', base_path + 'Nscaling_knn=200k_T=8']

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):

T_range, data_list = viewData.read_data(filepath, remove_outliers=REMOVE_OUTLIERS)


MI = viewData.Data(data_list[0])
H_hxc = viewData.Data(data_list[1])
H_xxc = viewData.Data(data_list[2])
H_joint = viewData.Data(data_list[3])
H_cond = viewData.Data(data_list[4])

# Potentially more accurate taking into accound each mean value
MI.mean = H_hxc.mean + H_xxc.mean - H_joint.mean - H_cond.mean


"""
Experiment combining data but with offset of 1 (like reusing old data). 
The target entropy should remain the same and this gives better accuracy
"""
if COMBINE_ENTROPIES:
    # H_hxc.mean = np.nanmean(np.append(H_hxc.data[:,:-1],H_joint.data[:,1:],axis=1),axis=0)
    # temp = np.empty(H_hxc.data.shape)*np.nan
    temp = np.insert(
        H_joint.data[:, :-1], 0, np.nan, axis=1
    )  # insert column of nan to align matrices
    H_hxc = viewData.Data(np.append(H_hxc.data, temp, axis=0))

    temp = np.insert(H_hxc.data[:, 1:], H_joint.data.shape[1] - 1, np.nan, axis=1)
    H_joint = viewData.Data(np.append(temp, H_joint.data, axis=0))

    temp = np.insert(
        H_cond.data[:, 1:], H_joint.data.shape[1] - 1, np.nan, axis=1
    )  # insert column of nan to align matrices
    H_xxc_long = viewData.Data(np.append(H_xxc.data, temp, axis=0))

    temp = np.insert(H_xxc.data[:, :-1], 0, np.nan, axis=1)
    H_cond = viewData.Data(np.append(temp, H_cond.data, axis=0))

    len_n = min(H_hxc.data_long.shape[0], H_xxc.data_long.shape[0])
    _MI_tot_long = (
        H_hxc.data[:len_n, :]
        + H_xxc.data[:len_n, :]
        - H_joint.data[:len_n, :]
        - H_cond.data[:len_n, :]
    )
    MI_long = viewData.Data(_MI_tot_long)
    # Due to inserting NaN to align matrices, first and last columns will always be NaN
    # Use original MI for first and last columns
    MI_long.data[: MI.data.shape[0], [0, MI_long.data.shape[1] - 1]] = MI.data[
        :, [0, MI_long.data.shape[1] - 1]
    ]
    MI_long.refresh_stats()


"""
Max capacity
"""
from simulators.CPDSSS_models import CPDSSS

sim_model = CPDSSS(1, N, d0=d0, d1=d1)
H_h = sim_model.chan_entropy()


"""View how each entropy changes as T increases"""
fig2, ax2 = plt.subplots(2, 2)
fig2.suptitle("Entropy increase per added transmission, N={}".format(N))

diff = H_hxc.mean[1:] - H_hxc.mean[:-1]
yerr = H_hxc.var[1:] + H_hxc.var[:-1]
ax2[0, 0].cla(), ax2[0, 0].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[0, 0].set_title("H1(g,x_cond)"), ax2[0, 0].set_ylabel("delta H()"), ax2[0, 0].set_xlabel("T")

diff = H_joint.mean[1:] - H_joint.mean[:-1]
yerr = H_joint.var[1:] - H_joint.var[:-1]
ax2[1, 0].cla(), ax2[1, 0].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[1, 0].set_title("H1(g,x,x_cond)"), ax2[1, 0].set_ylabel("delta H()"), ax2[1, 0].set_xlabel("T")

diff = H_cond.mean[1:] - H_cond.mean[:-1]
yerr = H_cond.var[1:] - H_cond.var[:-1]
ax2[0, 1].cla(), ax2[0, 1].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[0, 1].set_title("H1(x_cond)"), ax2[0, 1].set_ylabel("delta H()"), ax2[0, 1].set_xlabel("T")

diff = H_xxc.mean[1:] - H_xxc.mean[:-1]
yerr = H_xxc.var[1:] - H_xxc.var[:-1]
ax2[1, 1].cla(), ax2[1, 1].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[1, 1].set_title("H1(x,x_cond)"), ax2[1, 1].set_ylabel("delta H()"), ax2[1, 1].set_xlabel("T")


"""Entropy Scatter plots"""
fig2, ax2 = plt.subplots(2, 2)
fig2.suptitle("Entropy increase per added transmission, N={}".format(N))


def plt_H_diff_scatter(ax, data):
    diff = []
    for i in range(data.shape[1] - 1):
        trim1, trim2 = (data[:, i + 1], data[:, i])
        trim1 = trim1[~np.isnan(trim1)]
        trim2 = trim2[~np.isnan(trim2)]
        min_dim = min(trim1.shape[0], trim2.shape[0])
        diff.append(trim1[:min_dim] - trim2[:min_dim])
    default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    for i, y in enumerate(diff):
        x = np.full(y.shape, T_range[i])
        ax.scatter(x, y, color=default_color)


ax2[0, 0].cla(), ax2[0, 0].set_title("H(g,x_cond)")
plt_H_diff_scatter(ax2[0, 0], H_hxc.data)

ax2[1, 0].cla(), ax2[1, 0].set_title("H(g,x,x_cond)")
plt_H_diff_scatter(ax2[1, 0], H_joint.data)

ax2[0, 1].cla(), ax2[0, 1].set_title("H(x_cond)")
plt_H_diff_scatter(ax2[0, 1], H_cond.data)

ax2[1, 1].cla(), ax2[1, 1].set_title("H(x,x_cond)")
plt_H_diff_scatter(ax2[1, 1], H_xxc.data)


fig2.tight_layout()

"""Plot individual and cumulative Mutual Information"""
fig3, ax3 = plt.subplots(1, 1)
fig3.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
temp_range = np.insert(T_range, 0, 1)
MI.mean = np.insert(MI.mean, 0, 0)  # start at 0 MI
ax3.cla()
ax3.plot(temp_range, MI.mean)
ax3.set_title(r"$I(\mathbf{g},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")
ax3.set_xlabel(r"$T$")
fig3, ax3 = plt.subplots(1, 1)
ax3.cla()
ax3.plot(temp_range, np.cumsum(MI.mean), label=r"$I(\mathbf{g},\mathbf{X})$")
ax3.axhline(y=H_h, linestyle="dashed", label=r"$H(\mathbf{g})$")
ax3.set_title(r"$I(\mathbf{g},\mathbf{X})$")
ax3.set_xlabel(r"T")
ax3.legend()

fig3.tight_layout()

"""Plot Mutual Information, but include bars showing variance"""
fig4, ax4 = plt.subplots(2, 1)
fig4.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
yerr = np.insert(MI.std, 0, 0)
ax4[0].cla(), ax4[0].errorbar(temp_range, MI.mean, yerr=yerr)
ax4[0].set_title(r"$I(\mathbf{g},\mathbf{x}_T | \mathbf{x}_{1:T-1})$"), ax4[0].set_xlabel(r"$T$")
ax4[1].cla(), ax4[1].errorbar(
    temp_range, np.cumsum(MI.mean), yerr=np.cumsum(yerr), label=r"$I(\mathbf{g},\mathbf{X})$"
), ax4[1].set_title("total MI"), ax4[1].set_xlabel("T")
ax4[1].axhline(y=H_h, linestyle="dashed", label="H(G)"), ax4[1].legend()
ax4[1].set_title(r"$I(\mathbf{g},\mathbf{X})$"), ax4[1].set_xlabel(r"T")
fig4.tight_layout()

"""Scatter plot of the Mutual Information"""
if len(T_range) == 1:
    # plot just T=8
    fig5, ax5 = plt.subplots(1, 1)
    T_matrix = np.tile([T_range[0]], (MI.data.shape[0], 1))
    ax5.cla(), ax5.scatter(T_matrix, MI.data), ax5.set_title(
        "Low number of samples 10k*N, std = {0:.3f}".format(np.nanstd(MI.data))
    ), ax5.set_xlabel("T")
else:
    fig5, ax5 = plt.subplots(1, 2)
    fig5.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
    T_matrix = np.tile(np.array(T_range), (MI.data.shape[0], 1))
    ax5[0].cla(), ax5[0].scatter(T_matrix, MI.data), ax5[0].set_title("MI increase per T"), ax5[
        0
    ].set_xlabel("T")
    ax5[1].cla(), ax5[1].scatter(T_matrix, np.cumsum(MI.data, axis=1)), ax5[1].set_title(
        "total MI"
    ), ax5[1].set_xlabel("T")


fig5.tight_layout()

if COMBINE_ENTROPIES:
    """scatter plot with combining similar entropies"""
    fig6, ax6 = plt.subplots(1, 2)
    fig6.suptitle(rf"Combined Entropies, $N=${N}, $d_0=${d0}, $d_1=${d1}")
    T_matrix = np.tile(np.array(T_range), (MI_long.data.shape[0], 1))
    ax6[0].cla(), ax6[0].scatter(T_matrix, MI_long.data), ax6[0].set_title(
        "MI increase per T"
    ), ax6[0].set_xlabel("T")
    ax6[1].cla(), ax6[1].scatter(T_matrix, np.nancumsum(MI_long.data, axis=1)), ax6[1].set_title(
        "total MI"
    ), ax6[1].set_xlabel("T")
    fig5.tight_layout()

    """Error bars with combining similar entropies"""
    fig4, ax4 = plt.subplots(1, 2)
    fig4.suptitle(rf"Combined Entropies, $N=${N}, $d_0=${d0}, $d_1=${d1}")
    yerr = np.insert(MI_long.var, 0, 0)
    MI_long.mean = np.insert(MI_long.mean, 0, 0)
    ax4[0].cla(), ax4[0].errorbar(temp_range, MI_long.mean, yerr=yerr), ax4[0].set_title(
        "MI increase per T, error bars"
    ), ax4[0].set_xlabel("T")
    ax4[1].cla(), ax4[1].errorbar(temp_range, np.cumsum(MI_long.mean), yerr=np.cumsum(yerr)), ax4[
        1
    ].set_title("total MI"), ax4[1].set_xlabel("T")
    fig4.tight_layout()

plt.show()
