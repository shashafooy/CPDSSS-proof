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

REMOVE_OUTLIERS = False
COMBINE_ENTROPIES = False

# util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename))
base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_d0d1({d0},{d1})/"
# filepaths = [base_path + "50k_high_epoch", base_path + "50k_samples"]
# filepath = base_path + "50k_tol_0.1_patience_10"
# filepath = base_path + "50k_N4_L2"
# filepath = base_path + "50k_N4_L2"
# filepath = base_path + 'NlogN_10k_scaling'
# filepath = base_path + 'NlogN_10k_K=3,T=8'
# filepath = base_path + 'knn=200k_T=2-7'
# filepath = base_path + 'coarse-fine_75k_x_dims'
filepath = base_path + "pretrained_model"
# filepath = base_path + "N9_coarse-fine_experiment"

# filepath= 'temp_data/CPDSSS_data/N2_L2/50k_tol_0.1_patience_10'
# filepath= 'temp_data/CPDSSS_data/N2_L2/50k_samples'
# filepath = base_path + 'NlogN_10k_scaling'
# filepaths = [base_path + 'NlogN_10k_scaling', base_path + 'Nscaling_knn=200k_T=8']

# filepath=filepaths[1]
# idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename = os.path.splitext(filename)[0]  # remove extention
    T_range, MI_cum, H_gxc_cum, H_xxc_cum, H_joint_cum, H_cond_cum, completed_iter = util.io.load(
        os.path.join(filepath, filename)
    )
    iter = range(0, completed_iter + 1)

    if "MI_tot" not in locals():
        MI_tot = np.empty((0, np.size(T_range)))
        H_gxc_tot = np.empty((0, np.size(T_range)))
        H_xxc_tot = np.empty((0, np.size(T_range)))
        H_joint_tot = np.empty((0, np.size(T_range)))
        H_cond_tot = np.empty((0, np.size(T_range)))
        old_range = T_range

    MI_tot, _ = viewData.align_and_concatenate(MI_tot, MI_cum, old_range, T_range)
    H_gxc_tot, _ = viewData.align_and_concatenate(H_gxc_tot, H_gxc_cum, old_range, T_range)
    H_xxc_tot, _ = viewData.align_and_concatenate(H_xxc_tot, H_xxc_cum, old_range, T_range)
    H_joint_tot, _ = viewData.align_and_concatenate(H_joint_tot, H_joint_cum, old_range, T_range)
    H_cond_tot, old_range = viewData.align_and_concatenate(
        H_cond_tot, H_cond_cum, old_range, T_range
    )

T_range = old_range
"""
Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
"""
if REMOVE_OUTLIERS:
    MI_tot = viewData.remove_outlier(data=MI_tot, num_std=3)
    H_gxc_tot = viewData.remove_outlier(data=H_gxc_tot, num_std=3)
    H_xxc_tot = viewData.remove_outlier(data=H_xxc_tot, num_std=3)
    H_joint_tot = viewData.remove_outlier(data=H_joint_tot, num_std=3)
    H_cond_tot = viewData.remove_outlier(data=H_cond_tot, num_std=3)


MI_mean = np.nanmean(MI_tot, axis=0)
H_gxc_mean = np.nanmean(H_gxc_tot, axis=0)
H_xxc_mean = np.nanmean(H_xxc_tot, axis=0)
H_joint_mean = np.nanmean(H_joint_tot, axis=0)
H_cond_mean = np.nanmean(H_cond_tot, axis=0)

MI_var = np.nanvar(MI_tot, axis=0)
H_gxc_var = np.nanvar(H_gxc_tot, axis=0)
H_xxc_var = np.nanvar(H_xxc_tot, axis=0)
H_joint_var = np.nanvar(H_joint_tot, axis=0)
H_cond_var = np.nanvar(H_cond_tot, axis=0)


T_range = old_range
# Potentially more accurate taking into accound each mean value
MI_mean_sum = H_gxc_mean + H_xxc_mean - H_joint_mean - H_cond_mean


"""
Experiment combining data but with offset of 1 (like reusing old data). 
The target entropy should remain the same and this gives better accuracy
"""
if COMBINE_ENTROPIES:
    # H_gxc_mean = np.nanmean(np.append(H_gxc_tot[:,:-1],H_joint_tot[:,1:],axis=1),axis=0)
    # temp = np.empty(H_gxc_tot.shape)*np.nan
    temp = np.insert(
        H_joint_tot[:, :-1], 0, np.nan, axis=1
    )  # insert column of nan to align matrices
    H_gxc_tot_long = np.append(H_gxc_tot, temp, axis=0)
    H_gxc_mean = np.nanmean(H_gxc_tot_long, axis=0)

    # temp = temp = np.empty(H_joint_tot.shape)*np.nan
    temp = np.insert(H_gxc_tot[:, 1:], H_joint_tot.shape[1] - 1, np.nan, axis=1)
    H_joint_tot_long = np.append(temp, H_joint_tot, axis=0)
    H_joint_mean = np.nanmean(H_joint_tot_long, axis=0)

    temp = np.insert(
        H_cond_tot[:, 1:], H_joint_tot.shape[1] - 1, np.nan, axis=1
    )  # insert column of nan to align matrices
    H_xxc_tot_long = np.append(H_xxc_tot, temp, axis=0)
    H_xxc_mean = np.nanmean(H_xxc_tot_long, axis=0)

    temp = np.insert(H_xxc_tot[:, :-1], 0, np.nan, axis=1)
    H_cond_tot_long = np.append(temp, H_cond_tot, axis=0)
    H_cond_mean = np.nanmean(H_cond_tot_long, axis=0)

    len_n = min(H_gxc_tot_long.shape[0], H_xxc_tot_long.shape[0])
    MI_mean = H_gxc_mean + H_xxc_mean - H_joint_mean - H_cond_mean
    MI_tot_long = (
        H_gxc_tot_long[:len_n, :]
        + H_xxc_tot_long[:len_n, :]
        - H_joint_tot_long[:len_n, :]
        - H_cond_tot_long[:len_n, :]
    )
    # Due to inserting NaN to align matrices, first and last columns will always be NaN
    # Use original MI for first and last columns
    MI_tot_long[: MI_tot.shape[0], [0, MI_tot_long.shape[1] - 1]] = MI_tot[
        :, [0, MI_tot_long.shape[1] - 1]
    ]
    # MI_tot_long[:,[0, MI_tot_long.shape[1]-1]] = MI_tot[:,[0,MI_tot_long.shape[1]-1]]
    MI_mean_long = np.nanmean(MI_tot_long, axis=0)
    # MI_mean = MI_mean_long


"""
Max capacity
"""
from simulators.CPDSSS_models import CPDSSS

sim_model = CPDSSS(1, N, d0=d0, d1=d1)
H_h = sim_model.chan_entropy()


"""View how each entropy changes as T increases"""
fig2, ax2 = plt.subplots(2, 2)
fig2.suptitle("Entropy increase per added transmission, N={}".format(N))

diff = H_gxc_mean[1:] - H_gxc_mean[0:-1]
yerr = np.nanvar(H_gxc_tot[:, 1:], axis=0) + np.nanvar(H_gxc_tot[:, :-1], axis=0)
ax2[0, 0].cla(), ax2[0, 0].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[0, 0].set_title("H1(g,x_cond)"), ax2[0, 0].set_ylabel("delta H()"), ax2[0, 0].set_xlabel("T")

diff = H_joint_mean[1:] - H_joint_mean[0:-1]
yerr = np.nanvar(H_joint_tot[:, 1:], axis=0) + np.nanvar(H_joint_tot[:, :-1], axis=0)
ax2[1, 0].cla(), ax2[1, 0].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[1, 0].set_title("H1(g,x,x_cond)"), ax2[1, 0].set_ylabel("delta H()"), ax2[1, 0].set_xlabel("T")

diff = H_cond_mean[1:] - H_cond_mean[0:-1]
yerr = np.nanvar(H_cond_tot[:, 1:], axis=0) + np.nanvar(H_cond_tot[:, :-1], axis=0)
ax2[0, 1].cla(), ax2[0, 1].errorbar(T_range[:-1], diff, yerr=yerr)
ax2[0, 1].set_title("H1(x_cond)"), ax2[0, 1].set_ylabel("delta H()"), ax2[0, 1].set_xlabel("T")

diff = H_xxc_mean[1:] - H_xxc_mean[0:-1]
yerr = np.nanvar(H_xxc_tot[:, 1:], axis=0) + np.nanvar(H_xxc_tot[:, :-1], axis=0)
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
plt_H_diff_scatter(ax2[0, 0], H_gxc_tot)

ax2[1, 0].cla(), ax2[1, 0].set_title("H(g,x,x_cond)")
plt_H_diff_scatter(ax2[1, 0], H_joint_tot)

ax2[0, 1].cla(), ax2[0, 1].set_title("H(x_cond)")
plt_H_diff_scatter(ax2[0, 1], H_cond_tot)

ax2[1, 1].cla(), ax2[1, 1].set_title("H(x,x_cond)")
plt_H_diff_scatter(ax2[1, 1], H_xxc_tot)


fig2.tight_layout()

"""Plot individual and cumulative Mutual Information"""
fig3, ax3 = plt.subplots(2, 1)
fig3.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
temp_range = range(1, max(T_range) + 1)
temp_range = np.insert(T_range, 0, 1)
MI_mean = np.insert(MI_mean, 0, 0)  # start at 0 MI
ax3[0].cla(), ax3[0].plot(temp_range, MI_mean)
ax3[0].set_title(r"$I(\mathbf{g},\mathbf{x}_T | \mathbf{x}_{1:T-1})$"), ax3[0].set_xlabel(r"$T$")
ax3[1].cla(), ax3[1].plot(temp_range, np.cumsum(MI_mean), label=r"$I(\mathbf{g},\mathbf{X})$")
ax3[1].axhline(y=H_h, linestyle="dashed", label=r"$H(\mathbf{g})$")
ax3[1].set_title(r"$I(\mathbf{g},\mathbf{X})$"), ax3[1].set_xlabel(r"T")
ax3[1].legend()

fig3.tight_layout()

"""Plot Mutual Information, but include bars showing variance"""
fig4, ax4 = plt.subplots(2, 1)
fig4.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
yerr = np.insert(np.nanstd(MI_tot, axis=0), 0, 0)
ax4[0].cla(), ax4[0].errorbar(temp_range, MI_mean, yerr=yerr)
ax4[0].set_title(r"$I(\mathbf{g},\mathbf{x}_T | \mathbf{x}_{1:T-1})$"), ax4[0].set_xlabel(r"$T$")
ax4[1].cla(), ax4[1].errorbar(
    temp_range, np.cumsum(MI_mean), yerr=np.cumsum(yerr), label=r"$I(\mathbf{g},\mathbf{X})$"
), ax4[1].set_title("total MI"), ax4[1].set_xlabel("T")
ax4[1].axhline(y=H_h, linestyle="dashed", label="H(G)"), ax4[1].legend()
ax4[1].set_title(r"$I(\mathbf{g},\mathbf{X})$"), ax4[1].set_xlabel(r"T")
fig4.tight_layout()

"""Scatter plot of the Mutual Information"""
if len(T_range) == 1:
    # plot just T=8
    fig5, ax5 = plt.subplots(1, 1)
    T_matrix = np.tile([T_range[0]], (MI_tot.shape[0], 1))
    ax5.cla(), ax5.scatter(T_matrix, MI_tot), ax5.set_title(
        "Low number of samples 10k*N, std = {0:.3f}".format(np.nanstd(MI_tot))
    ), ax5.set_xlabel("T")
else:
    fig5, ax5 = plt.subplots(1, 2)
    fig5.suptitle(rf"$N=${N}, $d_0=${d0}, $d_1=${d1}")
    T_matrix = np.tile(np.array(T_range), (MI_tot.shape[0], 1))
    ax5[0].cla(), ax5[0].scatter(T_matrix, MI_tot), ax5[0].set_title("MI increase per T"), ax5[
        0
    ].set_xlabel("T")
    ax5[1].cla(), ax5[1].scatter(T_matrix, np.cumsum(MI_tot, axis=1)), ax5[1].set_title(
        "total MI"
    ), ax5[1].set_xlabel("T")


fig5.tight_layout()

if COMBINE_ENTROPIES:
    """scatter plot with combining similar entropies"""
    fig6, ax6 = plt.subplots(1, 2)
    fig6.suptitle(rf"Combined Entropies, $N=${N}, $d_0=${d0}, $d_1=${d1}")
    T_matrix = np.tile(np.array(T_range), (MI_tot_long.shape[0], 1))
    ax6[0].cla(), ax6[0].scatter(T_matrix, MI_tot_long), ax6[0].set_title("MI increase per T"), ax6[
        0
    ].set_xlabel("T")
    ax6[1].cla(), ax6[1].scatter(T_matrix, np.nancumsum(MI_tot_long, axis=1)), ax6[1].set_title(
        "total MI"
    ), ax6[1].set_xlabel("T")
    fig5.tight_layout()

    """Error bars with combining similar entropies"""
    fig4, ax4 = plt.subplots(1, 2)
    fig4.suptitle(rf"Combined Entropies, $N=${N}, $d_0=${d0}, $d_1=${d1}")
    yerr = np.insert(np.nanvar(MI_tot_long, axis=0), 0, 0)
    MI_mean_long = np.insert(MI_mean_long, 0, 0)
    ax4[0].cla(), ax4[0].errorbar(temp_range, MI_mean_long, yerr=yerr), ax4[0].set_title(
        "MI increase per T, error bars"
    ), ax4[0].set_xlabel("T")
    ax4[1].cla(), ax4[1].errorbar(temp_range, np.cumsum(MI_mean_long), yerr=np.cumsum(yerr)), ax4[
        1
    ].set_title("total MI"), ax4[1].set_xlabel("T")
    fig4.tight_layout()

plt.show()
