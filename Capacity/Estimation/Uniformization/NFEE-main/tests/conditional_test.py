import numpy as np
from datetime import date
import scipy.linalg as lin

import os
from _utils import set_sys_path

set_sys_path()

import simulators.CPDSSS_models as simMod
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io


SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True

SAVE_FILE = True

"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entros.pathy calc
n_train_samples = 100000

N = 6
inputs = 2
givens = N - inputs

model = None


mu = np.zeros((N))
row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = np.tile(row, (N, 1))
np.fill_diagonal(sigma, 1)
# sigma = np.array([np.roll(row, i) for i in range(N)])
for i in range(1, N):
    sigma[i, :i] = row[i]

sim_model = simMod.Gaussian(mu, sigma)
samples = sim_model.sim(n_train_samples * N)

samples = [
    samples[:, :inputs],
    samples[:, inputs:],
]  # first 2 dim conditioned on last N-2 dimensions
sim_model.input_dim = [inputs, givens]

H, estimator = ent.calc_entropy(sim_model, base_samples=samples)

joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
cond_H = joint_H - marginal_H

print("sigma always decreasing for more N")
print(f"estimated H: {H:.4f}")
print(f"true H: {cond_H:.4f}")

"""Toeplitz sigma"""

row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = lin.toeplitz(row)

sim_model = simMod.Gaussian(mu, sigma)
samples = sim_model.sim(n_train_samples * N)

samples = [
    samples[:, :inputs],
    samples[:, inputs:],
]  # first 2 dim conditioned on last N-2 dimensions
sim_model.input_dim = [inputs, givens]

H, estimator = ent.calc_entropy(sim_model, base_samples=samples)

joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
cond_H = joint_H - marginal_H

print("sigma toeplitz, further away values have less weight")
print(f"estimated H: {H:.4f}")
print(f"true H: {cond_H:.4f}")


# """
# File names
# """
# today = date.today().strftime("%b_%d")
# base_path = f"temp_data/CPDSSS_data/MI(h,X)/conditional/N{N}_d0d1({d0},{d1})/"
# path = base_path  # + "pretrained_model"
# filename = "CPDSSS_data({})".format(today)

# model_path = f"temp_data/saved_models/{N}N_d0d1({d0},{d1})"
# X_path = os.path.join(model_path, "X")
# XH_path = os.path.join(model_path, "XH")


# # fix filename if file already exists
# if SAVE_FILE:
#     filename = misc.update_filename(path, filename, -1, rename=False)

# for i in range(n_trials):
#     sim_model = CPDSSS_Cond(2, N, d0=d0, d1=d1)
#     for k, T in enumerate(T_range):
#         index = (i, k)
#         """
#         Generate samples
#         """
#         misc.print_border("Generating CPDSSS samples")
#         # sim_model = CPDSSS_Cond(T, N, d0=d0, d1=d1)
#         sim_model.set_T(T)
#         # generate base samples based on max dimension
#         sim_model.set_XHcond()
#         knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
#         samples = sim_model.sim(knn_samples, reuse_GQ=True)

#         """Train H(xT | h, x1:T-1)"""
#         misc.print_border(f"1/2 calculating H(xT | h, x1:T-1), T: {T}, iter: {i+1}")
#         name = f"{T-1}T"

#         H_XH[index], _ = ent.calc_entropy(sim_model, base_samples=samples)

#         if SAVE_FILE:
#             filename = misc.update_filename(path, filename, i)
#             util.io.save(
#                 (T_range, MI, H_XH, H_XX, i),
#                 os.path.join(path, filename),
#             )

#         """Train H(xT |x1:T-1)"""
#         misc.print_border(f"2/2 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
#         name = f"{T-1}T"

#         sim_model.set_Xcond()
#         samples = sim_model.sim(knn_samples, reuse_GQ=True)
#         H_XX[index], _ = ent.calc_entropy(sim_model, base_samples=samples)
#         MI[index] = H_XX[index] - H_XH[index]

#         if SAVE_FILE:
#             util.io.save(
#                 (T_range, MI, H_XH, H_XX, i),
#                 os.path.join(path, filename),
#             )
