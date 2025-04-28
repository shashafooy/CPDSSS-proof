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

    sim_model = CPDSSS(1, N, d0=d0, d1=d1, use_fading=False)
    H_h.append(sim_model.chan_entropy())

# manual smoothing/prediction
"""12N using fading"""
# MI[0].mean[-2:] = [0.28, 0.23]
# MI[2].mean[-5:-1] = [0.386, 0.346, 0.303, 0.255]

# H_HX[0].mean = np.array([14.5132, 14.5166, 14.5215, 14.5277, 14.5360, 14.5404, 14.5537, 14.5603])
# H_X[0].mean = np.array([15.7444, 15.5817, 15.4454, 15.3492, 15.2795, 15.2169, 15.1924, 15.1820])

"""12N without fading"""
# model loss values
# d0d1=6,6
H_X[0].mean[:] = [13.1207, 12.6009, 12.2494, 11.9966, 11.8077, 11.6757, 11.5875, 11.5429, 11.6195]
H_HX[0].mean[:] = [10.7586, 10.7599, 10.7533, 10.7724, 10.7602, 10.7563, 10.7592, 10.7637, 10.8048]
MI[0].mean = H_X[0].mean - H_HX[0].mean

# d0d1=3,9
H_HX[1].mean = np.asarray([13.636, 13.634, 13.634, 13.632, 13.635, 13.641, 13.651, 13.658, 13.657])
H_X[1].mean = np.asarray([15.488, 15.447, 15.425, 15.388, 15.374, 15.358, 15.352, 15.351, 15.475])
MI[1].mean = H_X[1].mean - H_HX[1].mean

# d0d1=9,3
H_HX[2].mean = np.asarray([8.424, 8.445, 8.372, 8.295, 8.309, 8.279, 8.263, 8.610, 8.643])
H_X[2].mean = np.asarray([10.737, 10.391, 10.157, 10.017, 9.889, 9.824, 9.729, 9.634, 9.815])
MI[2].mean = H_X[2].mean - H_HX[2].mean

H_h = CPDSSS(1, N, d0=6, d1=6, use_fading=False).chan_entropy()

"""3N without fading"""
# d0d1=1,2
H_X[0].mean[:] = [4.124, 4.100, 4.098, 4.089, 4.079, 4.073, 4.067, 4.057, 4.053]
H_HX[0].mean[:] = [3.902, 3.900, 3.904, 3.903, 3.906, 3.902, 3.903, 3.904, 3.905]
MI[0].mean = H_X[0].mean - H_HX[0].mean

# d0d1=2,1
H_X[1].mean = np.asarray([4.113, 4.058, 4.016, 3.987, 3.965, 3.945, 3.928, 3.916, 3.904])
H_HX[1].mean = np.asarray([3.706, 3.711, 3.713, 3.713, 3.714, 3.714, 3.713, 3.714, 3.714])
MI[1].mean = H_X[1].mean - H_HX[1].mean

H_h = CPDSSS(1, 3, d0=1, d1=2, use_fading=False).chan_entropy()

"""6N without fading"""
# d0d1=3,3
H_X[0].mean = np.asarray([7.261, 7.121, 7.012, 6.925, 6.859, 6.808, 6.771, 6.739, 6.710])
H_HX[0].mean = np.asarray([6.414, 6.412, 6.418, 6.413, 6.422, 6.416, 6.419, 6.416, 6.422])
MI[0].mean = H_X[0].mean - H_HX[0].mean

# d0d1=2,4
H_X[1].mean = np.asarray([7.816, 7.780, 7.751, 7.715, 7.683, 7.659, 7.628, 7.595, 7.581])
H_HX[1].mean = np.asarray([7.131, 7.130, 7.133, 7.129, 7.125, 7.126, 7.128, 7.127, 7.130])
MI[1].mean = H_X[1].mean - H_HX[1].mean
H_h = CPDSSS(1, 6, d0=3, d1=3, use_fading=False).chan_entropy()

"""6N using fading"""
# d0d1=3,3
H_HX[0].mean = np.asarray([7.462, 7.463, 7.463, 7.470, 7.467, 7.471, 7.469, 7.477, 7.471])
H_HX[1].mean = 7.4318  # H(X|h) = N/2 * np.log(2*pi*e) + 1/2 * mean(log(det(G*G' + Q*Q')))
H_X[0].mean = np.asarray([8.034, 7.959, 7.913, 7.867, 7.828, 7.790, 7.764, 7.755, 7.727])
MI[0].mean = H_X[0].mean - H_HX[0].mean
MI[0].mean = np.concatenate([[0.6226], MI[0].mean])  # Add in T=1 starting MI from H(h|x)

# d0d1=2,4
H_HX[1].mean = np.asarray([7.808, 7.814, 7.811, 7.814, 7.814, 7.817, 7.813, 7.815, 7.809])
H_HX[1].mean = 7.781  # H(X|h) = N/2 * np.log(2*pi*e) + 1/2 * mean(log(det(G*G' + Q*Q')))
H_X[1].mean = np.asarray([8.253, 8.225, 8.203, 8.183, 8.168, 8.156, 8.148, 8.135, 8.127])
MI[1].mean = H_X[1].mean - H_HX[1].mean
MI[1].mean = np.concatenate([[0.4406], MI[1].mean])  # Add in T=1 starting MI from H(h|x)

# d0d1=4,2
H_HX[2].mean = np.asarray([7.367, 7.375, 7.373, 7.311, 7.311, 7.319, 7.328, 7.333, 7.329])
H_HX[1].mean = 7.2405  # H(X|h) = N/2 * np.log(2*pi*e) + 1/2 * mean(log(det(G*G' + Q*Q')))
H_X[2].mean = np.asarray([8.084, 7.974, 7.907, 7.815, 7.782, 7.743, 7.720, 7.706, 7.688])
MI[2].mean = H_X[2].mean - H_HX[2].mean
MI[2].mean = np.concatenate([[0.8116], MI[2].mean])  # Add in T=1 starting MI from H(h|x)
H_h = CPDSSS(1, 6, d0=3, d1=3, use_fading=True).chan_entropy()

T_range[:] = [range(1, 11)] * 3

d0d1 = [(3, 3), (2, 4), (4, 2)]

"""6N using fading direct H(h|X)"""
H_hx = []
# d0d1=3,3
# H_hx.append(
#     np.array(
#         [
#             5.358,
#             4.854,
#             4.421,
#             4.023,
#             3.701,
#             3.415,
#             3.151,
#             2.927,
#             2.738,
#             2.536,
#             2.404,
#             2.311,
#             2.173,
#             2.090,
#             1.916,
#         ]
#     )
# )
H_hx.append(
    np.array(
        [
            5.354,
            4.856,
            4.421,
            4.017,
            3.695,
            3.398,
            3.146,
            2.921,
            2.727,
            2.531,
            2.359,
            2.226,
            2.067,
            1.967,
            1.885,
        ]
    )
)
# MI[0].mean = H_h - H_hx[0]
# d0d1=2,4
H_hx.append(
    np.array(
        [
            5.573,
            5.199,
            4.844,
            4.540,
            4.224,
            3.949,
            3.702,
            3.443,
            3.211,
            3.081,
            2.885,
            2.675,
            2.520,
            2.326,
            2.266,
        ]
    )
)
# MI[1].mean = H_h - H_hx[1]
# d0d1=4,2
# H_hx.append(
#     np.array(
#         [
#             5.098,
#             4.512,
#             4.092,
#             3.712,
#             3.380,
#             3.113,
#             2.845,
#             2.605,
#             2.487,
#             2.249,
#             2.085,
#             1.956,
#             1.738,
#             1.595,
#             1.765,
#         ]
#     )
# )
H_hx.append(
    np.array(
        [
            5.202,
            4.664,
            4.239,
            3.823,
            3.507,
            3.130,
            2.938,
            2.669,
            2.554,
            2.385,
            2.257,
            2.023,
            1.840,
            1.684,
            1.638,
        ]
    )
)
# MI[2].mean = H_h - H_hx[2]

fig, ax = plt.subplots(1, 2)
for i, (d0, d1) in enumerate(d0d1):
    mi = H_h - H_hx[i]
    NN = mi.shape[0]
    ax[0].plot(range(1, NN), mi[1:] - mi[:-1], label=rf"$d_0={d0},d_1={d1}$")
    ax[1].plot(range(1, NN + 1), mi, label=rf"$d_0={d0},d_1={d1}$")

ax[0].legend(), ax[1].legend()
ax[1].axhline(y=H_h, linestyle="dashed", color="black")
ax[1].text(
    x=1,
    y=H_h + 0.02,
    s=rf"$H(\mathbf{{h}})$",
    fontsize=14,
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax[1].set_ylim((0, ax[1].get_ylim()[1] + 0.2))
ax[0].set_title(r"$\Delta I(h|x_{1:T})$")
ax[1].set_title(r"$I(h|x_{1:T})$")

# Plot only total mutual information
fig, ax = plt.subplots()
for i, (d0, d1) in enumerate(d0d1):
    mi = H_h - H_hx[i]
    NN = mi.shape[0]
    ax.plot(range(1, NN + 1), mi, label=rf"$d_0={d0},d_1={d1}$")

ax.legend()
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"Mutual Information")


ax.axhline(y=H_h, linestyle="dashed", color="black")
ax.text(
    x=1,
    y=H_h + 0.02,
    s=rf"$H(\mathbf{{h}})$",
    fontsize=14,
    verticalalignment="bottom",
    horizontalalignment="left",
)
ax.set_ylim((0, ax.get_ylim()[1] + 0.2))

fig.tight_layout(pad=0.2)


"""3N using orthogonal G,Q gram schmidt"""
# d0d1=3,0
# H(X|Xold) = H(X|h,Xold)

# for data_HX, data_X, (d0, d1), t_range in zip(H_HX, H_X, d0d1, T_range):
#     fig, ax = plt.subplots(1, 2)
#     x = np.full(data_HX.data.shape, t_range)
#     ax[0].scatter(x, data_HX.data), ax[0].set_title("H(X|h,Xold)")
#     x = np.full(data_X.data.shape, t_range)
#     ax[1].scatter(x, data_X.data), ax[1].set_title("H(X,Xold)")
#     fig.suptitle(f"Cond scatter for d_0={d0}, d_1={d1}")


# T_range[:] = [range(2, 11)] * 3
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# fig4, ax4 = plt.subplots(2, 1)

# for i, N in enumerate(N_range):
for i, (d0, d1) in enumerate(d0d1):
    """Plot individual and cumulative Mutual Information"""

    _MI_mean = MI[i].mean / H_h if NORMALIZE_MI else MI[i].mean
    # fig1.suptitle("N={}, L={}".format(N, L))
    # temp_range = range(1, max(T_range[i]) + 1)
    if T_range[i][0] > 1:
        temp_range = np.insert(T_range[i], 0, 1)
        _MI_mean = np.insert(_MI_mean, 0, 0)  # start at 0 MI
    else:
        temp_range = T_range[i]

    ax1.plot(temp_range, _MI_mean, label=rf"$d_0={d0},d_1={d1}$")
    # ax1.set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")

    (line,) = ax2.plot(temp_range, np.cumsum(_MI_mean), label=rf"$d_0={d0},d_1={d1}$")

    # ax2.set_title(r"Total $I(\mathbf{h},\mathbf{X})$")

ax1.set_xlabel(r"$T$")
ax1.set_ylabel(r"Mutual Information")
ax2.set_xlabel(r"$T$")
ax2.set_ylabel(r"Mutual Information")


ax2.axhline(y=H_h, linestyle="dashed", color="black")
ax2.text(
    x=1,
    y=H_h + 0.02,
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
