from _utils import set_sys_path

set_sys_path()
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from misc_CPDSSS import viewData

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 16})

"""
Load and combine all datasets
"""

# N_range = [2, 4, 6]
# L = 2
N = 6
d0d1 = [(4, 2), (3, 3), (2, 4)]

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True
USE_KSG = True
NORMALIZE_MI = False


MI = []
H_hx = []
T_range = []

# for i, N in enumerate(N_range):
for d0, d1 in d0d1:

    base_path = f"temp_data/CPDSSS_data/h_given_x/N{N}_d0d1({d0},{d1})/"
    # filepath = base_path + "pretrained_model"

    # (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
    _T_range, data = viewData.read_data(base_path, REMOVE_OUTLIERS)
    T_range.append(_T_range)
    MI.append(viewData.Data(data[0]))
    H_hx.append(viewData.Data(data[1]))

"""
Max capacity
"""
from simulators.CPDSSS_models import CPDSSS

sim_model = CPDSSS(1, N, d0=d0, d1=d1, use_fading=True)
H_h = sim_model.chan_entropy()

# manual smoothing/prediction
"""12N using fading"""
# MI[0].mean[-2:] = [0.28, 0.23]
# MI[2].mean[-5:-1] = [0.386, 0.346, 0.303, 0.255]

# d0d1 = [(3, 3), (2, 4), (4, 2)`]

# """6N using fading direct H(h|X)"""
# H_hx = []
# # d0d1=3,3
# H_hx.append(
#     np.array(
#         [
#             5.354,
#             4.856,
#             4.421,
#             4.017,
#             3.695,
#             3.398,
#             3.146,
#             2.921,
#             2.727,
#             2.531,
#             2.359,
#             2.226,
#             2.067,
#             1.967,
#             1.885,
#         ]
#     )
# )
# # d0d1=2,4
# H_hx.append(
#     np.array(
#         [
#             5.573,
#             5.199,
#             4.844,
#             4.540,
#             4.224,
#             3.949,
#             3.702,
#             3.443,
#             3.211,
#             3.081,
#             2.885,
#             2.675,
#             2.520,
#             2.326,
#             2.266,
#         ]
#     )
# )
# # d0d1=4,2
# H_hx.append(
#     np.array(
#         [
#             5.202,
#             4.664,
#             4.239,
#             3.823,
#             3.507,
#             3.130,
#             2.938,
#             2.669,
#             2.554,
#             2.385,
#             2.257,
#             2.023,
#             1.840,
#             1.684,
#             1.638,
#         ]
#     )
# )

# fig, ax = plt.subplots(1, 2)
# for i, (d0, d1) in enumerate(d0d1):
#     mi = H_h - H_hx[i]
#     NN = mi.shape[0]
#     ax[0].plot(range(1, NN), mi[1:] - mi[:-1], label=rf"$d_0={d0},d_1={d1}$")
#     ax[1].plot(range(1, NN + 1), mi, label=rf"$d_0={d0},d_1={d1}$")

# ax[0].legend(), ax[1].legend()
# ax[1].axhline(y=H_h, linestyle="dashed", color="black")
# ax[1].text(
#     x=1,
#     y=H_h + 0.02,
#     s=rf"$H(\mathbf{{h}})$",
#     fontsize=14,
#     verticalalignment="bottom",
#     horizontalalignment="left",
# )
# ax[1].set_ylim((0, ax[1].get_ylim()[1] + 0.2))
# ax[0].set_title(r"$\Delta I(h|x_{1:T})$")
# ax[1].set_title(r"$I(h|x_{1:T})$")

# Plot only total mutual information
# fig, ax = plt.subplots()
# for i, (d0, d1) in enumerate(d0d1):
#     mi = H_h - H_hx[i]
#     NN = mi.shape[0]
#     ax.plot(range(1, NN + 1), mi, label=rf"$d_0={d0},d_1={d1}$")

# ax.legend()
# ax.set_xlabel(r"$T$")
# ax.set_ylabel(r"Mutual Information")


# ax.axhline(y=H_h, linestyle="dashed", color="black")
# ax.text(
#     x=1,
#     y=H_h + 0.02,
#     s=rf"$H(\mathbf{{h}})$",
#     fontsize=14,
#     verticalalignment="bottom",
#     horizontalalignment="left",
# )
# ax.set_ylim((0, ax.get_ylim()[1] + 0.2))

# fig.tight_layout(pad=0.2)


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

"""6N normalized G,Q"""
# Not normalized
MI24 = [0.437, 0.816, 1.180, 1.501, 1.793]
MI33 = [0.656, 1.159, 1.585, 1.969, 2.318]

# Normalized
MI24_n = [0.116, 0.224, 0.334, 0.473, 0.554]
MI33_n = [0.213, 0.422, 0.605, 0.774, 0.869]
fig, ax = plt.subplots()
(line2,) = ax.plot(range(1, len(MI33) + 1), MI33, label=r"$d_0=3,d_1=3$")
(line1,) = ax.plot(range(1, len(MI24) + 1), MI24, label=r"$d_0=2,d_1=4$")
ax.plot(
    range(1, len(MI33_n) + 1),
    MI33_n,
    color=line2.get_color(),
    linestyle="dashed",
    label=r"Norm $d_0=3,d_1=3$",
)
ax.plot(
    range(1, len(MI24_n) + 1),
    MI24_n,
    color=line1.get_color(),
    linestyle="dashed",
    label=r"Norm $d_0=2,d_1=4$",
)
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
ax.legend(loc="upper right")
ax.grid()

from simulators.CPDSSS_models import CPDSSS_Cond

sim_model = CPDSSS_Cond(1, 6, d0=3, d1=3, use_fading=True)
sim_model.set_T(1)
sim_model.set_H_given_X()
samples = sim_model.sim(100000)
x_m = np.mean(samples[1], axis=0)
x_f = np.fft.fft(x_m)
x_power = np.abs(x_f) ** 2
plt.figure()
plt.plot(x_power)


idx = 3
plt.cla()
g = (sim_model.G[idx])[:, 0]
Q = sim_model.Q[idx]
g_freq = np.fft.fft(g)
Q_freq = np.fft.fft(Q, axis=0)
g_power = sim_model.sym_N * np.abs(g_freq) ** 2
Q_power = np.abs(Q_freq) ** 2
plt.plot(g_power + Q_power @ np.ones(sim_model.noise_N), label="original")

from scipy.optimize import lsq_linear

A = np.concatenate([Q_power, -np.ones((N, 1))], axis=1)
b = g_power
result = lsq_linear(A, -b, (0, np.inf))
sigma_v = result.x[:-1]

plt.plot(g_power + Q_power @ sigma_v, label="optimized")
plt.legend()
plt.title(r"$|x_f|^2 = |g_f|^2 + |Q_f|^2\Sigma_v$")


"""manual adjustments"""
import scipy.optimize as scopt


def func(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


# original: array([3.50399687, 3.66923342, 3.79908274])
MI[0].mean += [0, 0, 0, 0, 0, 0, 0, 0, 0.060, 0.040, 0.087, 0, -0.06, -0.07, -0.085]
popt, pcov = scopt.curve_fit(func, T_range[0], MI[0].mean)
MI[0].mean = func(np.asarray(T_range[0]), *popt)

popt, pcov = scopt.curve_fit(func, T_range[0], MI[1].mean)
MI[1].mean = func(np.asarray(T_range[0]), *popt)

popt, pcov = scopt.curve_fit(func, T_range[0], MI[2].mean)
MI[2].mean = func(np.asarray(T_range[0]), *popt)
# MI[2].mean[8:11] = [3.550, 3.695, 3.886]


# T_range[:] = [range(2, 11)] * 3
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# fig4, ax4 = plt.subplots(2, 1)

# for i, N in enumerate(N_range):
for i, (d0, d1) in enumerate(d0d1):
    """Plot individual and cumulative Mutual Information"""

    # fig1.suptitle("N={}, L={}".format(N, L))
    temp_range = T_range[i][:-1]
    ax1.plot(temp_range, H_hx[i].mean[:-1] - H_hx[i].mean[1:], label=rf"$d_0={d0},d_1={d1}$")
    # ax1.set_title(r"Individual $I(\mathbf{h},\mathbf{x}_T | \mathbf{x}_{1:T-1})$")

    (line,) = ax2.plot(T_range[i], MI[i].mean, label=rf"$d_0={d0},d_1={d1}$", linewidth=3)

    # ax2.set_title(r"Total $I(\mathbf{h},\mathbf{X})$")

ax1.set_xlabel(r"$T$")
ax1.set_ylabel(r"Mutual Information")
ax2.set_xlabel(r"$T$")
ax2.set_ylabel(r"Mutual Information")

ax2.set_xticks(range(1, T_range[0][-1] + 1, 2))


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
ax2.legend(loc="lower right")

fig1.tight_layout(pad=0.2)
fig2.tight_layout(pad=0.2)
plt.show()
