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


H_cond = [14.070, 28.203, 42.321, 56.372, 70.429, 84.476, 98.670, 112.757, 126.926, 140.834]
H_MAF = [13.696, 27.997, 41.591, 56.362, 70.803, 84.844, 99.077, 113.219, 128.235, 142.265]
H_knn = [14.428, 28.364, 41.922, 56.580, 70.826, 84.854, 99.014, 113.772, 127.922, 141.530]
H_true = [13.808, 27.620, 41.460, 55.219, 69.001, 82.802, 96.555, 110.411, 124.410, 138.123]

H_cond = [13.935, 27.875, 41.802, 55.729, 69.636, 83.593, 97.489]
H_MAF = [13.612, 27.604, 41.862, 55.558, 69.649, 83.312, 97.982]
H_knn = [14.294, 28.541, 41.878, 56.773, 70.451, 84.418, 99.112]
H_true = [13.835, 27.652, 41.448, 55.250, 69.411, 82.845, 96.762]


###Some inserted data while experimenting###
# H_cond_MAF.mean[:, 1] = np.asarray(
#     [
#         14.486671658729943,
#         28.70834253000461,
#         42.68600229455808,
#         56.46467613836232,
#         70.06056066005162,
#         83.46207887091992,
#         96.81139383630642,
#         110.33187362132283,
#         123.76809966408065,
#         136.08115791907372,
#     ]
# )

# H_xy_MAF.mean[:, 1] = np.asarray(
#     [
#         23.002646998104904,
#         45.74013424384931,
#         68.29048398908634,
#         90.67414461903387,
#         113.00676944398734,
#         135.17360414946148,
#         157.20579286855903,
#         179.24316181949726,
#         202.90165337042362,
#         236.55020229982432,
#     ]
# )

# H_x_MAF.mean[:, 1] = np.asarray(
#     [
#         8.515698186973975,
#         17.020996837509028,
#         25.53769044359545,
#         34.05097993880985,
#         42.55809054019807,
#         51.07560955536743,
#         59.5886852969201,
#         68.10676048384481,
#         76.62171099417729,
#         85.12592969432993,
#     ]
# )

# temp placeholder values. Model loss values
# H_cond_MAF.mean[7] = 109.9792
# H_cond_MAF.mean[7:, 1] = np.array([110.21987, 123.41146, 135.82276])

# H_xy_MAF.mean[7:, 1] = np.array([178.98333, 201.179542, 228.37499])
# H_x_MAF.mean[7:, 1] = np.array([68.10809, 76.621849, 85.13507])

H_y_given_x_MAF = H_xy_MAF.mean - H_x_MAF.mean
H_y_given_x_knn = H_xy_kl_ksg.mean - H_x_kl_ksg.mean


# entropy values
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_knn, "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1], "--o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1], "--x")
ax1.legend(["H(y|x)", "KL", "KSG", "MAF", "cond MAF"])

fig.tight_layout()

# Normalizedentropy values
fig, ax1 = plt.subplots()
ax1.axhline(y=1, linestyle="dashed", color="black")
# ax1.plot(T_range, H_y_given_x_true.mean, "-")
ax1.plot(T_range, H_y_given_x_knn / H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1] / H_y_given_x_true.mean, "-o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1] / H_y_given_x_true.mean, "--x")
ax1.legend(["norm entropy", "KL", "KSG", "MAF", "cond MAF"])

# error
fig, ax1 = plt.subplots()
ax1.plot(T_range, H_y_given_x_knn - H_y_given_x_true.mean[:, np.newaxis], "--*")
ax1.plot(T_range, H_y_given_x_MAF[:, 1] - H_y_given_x_true.mean, "--o")
ax1.plot(T_range, H_cond_MAF.mean[:, 1] - H_y_given_x_true.mean, "--x")
ax1.legend(["KL", "KSG", "unif", "cond unif"])
ax1.set_ylabel("Entropy Error")
ax1.set_xlabel("T")
ax1.set_xticks(list(T_range))
fig.tight_layout(pad=0.2)

# log error
fig, ax1 = plt.subplots()
ax1.cla()
ax1.plot(T_range, np.log(np.abs(H_y_given_x_knn - H_y_given_x_true.mean[:, np.newaxis])), "--*")
ax1.plot(T_range, np.log(np.abs(H_y_given_x_MAF[:, 1] - H_y_given_x_true.mean)), "--o")
ax1.plot(T_range, np.log(np.abs(H_cond_MAF.mean[:, 1] - H_y_given_x_true.mean)), "--x")
ax1.legend(["KL", "KSG", "unif", "cond unif"])
ax1.set_ylabel("Entropy Error")
ax1.set_xlabel("T")
ax1.set_xticks(list(T_range))
fig.tight_layout(pad=0.2)
plt.show()
