import numpy as np
from datetime import date
import scipy.linalg as lin

import os
from _utils import set_sys_path

set_sys_path()

import simulators.CPDSSS_models as simMod
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
from ent_est.entropy import UMestimator
import util.io


SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True
LOAD_MODEL = False

SAVE_FILE = True

"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entros.pathy calc
n_train_samples = 100000

N = 8
T = 1
inputs = 2
givens = N - inputs

model = None
n_samples = N * n_train_samples

model_paths = f"temp_data/saved_models/conditional_tests/{N}N"


# for N in range(17, 20):
#    sim_model = simMod.Laplace(0, 2, N)
#    sim_model.input_dim = [inputs, N - inputs]
#    samples = sim_model.sim(n_samples)
#    H, estimator = ent.calc_entropy(
#        sim_model, base_samples=[samples[:, :inputs], samples[:, inputs:]], method="both"
#    )
#    H_true = sim_model.entropy() * inputs / N
#    print(f"\nLaplace N={N}")
#    print(f"estimated H: {H}")
#    print(f"true H: {H_true:.4f}")

# import sys

# sys.exit()


misc.print_border("A is random")
"""
y=Ax+n
A is random"""
name = f"random_A_{T}T"
mu = np.zeros((N))
row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = np.tile(row, (N, 1))
np.fill_diagonal(sigma, 1)
sigma_n = 2  # * np.eye(N)
sigma_A = 1

A = np.random.normal(0, np.sqrt(sigma_A), (n_samples, N, N))
x = np.random.normal(0, 1, (n_samples, N, T))
# n = np.random.multivariate_normal(mu, sigma_n, n_samples)
n = np.random.normal(0, np.sqrt(sigma_n), (n_samples, N, T))
y = np.matmul(A, x) + n
sim_model = simMod.Gaussian(mu, sigma)

sim_model.input_dim = [N * T, N * T]
samples = [y.reshape(n_samples, N * T, order="F"), x.reshape(n_samples, N * T, order="F")]

if LOAD_MODEL:
    model = ent.load_model(name=name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")


dets = np.sum(
    np.log(np.linalg.eigvalsh(sigma_n * np.eye(T) + sigma_A * np.matmul(x.transpose(0, 2, 1), x))),
    axis=1,
)
H_true = (N * T) / 2 * np.log(2 * np.pi * np.exp(1)) + N / 2 * np.mean(dets)
_ = ent.update_best_model(estimator.model, samples, name=name, path=model_paths)


print(f"\ny=Ax+n, A is random")
print(f"estimated H: {H}")
print(f"true H: {H_true:.4f}")

misc.print_border("A is constant")
import sys

sys.exit()
"""
y=Ax+n
A is constant"""
name = "constant_A"
mu = np.zeros((N))

sigma_n = 2 * np.eye(N)
# sigma_A = 1
A = np.random.normal(0, 1, (N, N))
x = np.random.normal(0, 1, (n_samples, N, 1))
n = np.random.multivariate_normal(mu, sigma_n, n_samples)
y = np.squeeze(np.matmul(A, x)) + n
sim_model = simMod.Gaussian(mu, sigma)
sim_model.input_dim = [N, N]
samples = [y, np.squeeze(x)]
if LOAD_MODEL:
    model = ent.load_model(name=name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")
H_true = N / 2 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(lin.det(sigma_n))
_ = ent.update_best_model(estimator.model, samples, name=name, path=model_paths)

print(f"y=Ax+n, A is constant")
print(f"estimated H: {H}")
print(f"true H: {H_true:.4f}")


# import sys

# sys.exit()

"""gaussian covariance, same diminishing covar for each var"""
misc.print_border("gaussian covar decreasing for each N")

name = "gaussian_cov_decreasing"
row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = np.tile(row, (N, 1))
np.fill_diagonal(sigma, 1)
for i in range(1, N):
    sigma[i, :i] = row[i]

sim_model = simMod.Gaussian(mu, sigma)
samples = sim_model.sim(n_train_samples * N)

samples = [
    samples[:, :inputs],
    samples[:, inputs:],
]  # first 2 dim conditioned on last N-2 dimensions
sim_model.input_dim = [inputs, givens]
if LOAD_MODEL:
    model = ent.load_model(name=name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")

joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
H_true = joint_H - marginal_H

_ = ent.update_best_model(estimator.model, samples, name=name, path=model_paths)

print(f"gaussian covar, sigma always decreasing for each additional N.")
print(f"estimated H: {H}")
print(f"true H: {H_true:.4f}")


"""Toeplitz sigma"""
misc.print_border("gaussian covar toeplitz. high correlation with nearby vars")

name = "gaussian_cov_distance_falloff"
row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = lin.toeplitz(row)

sim_model = simMod.Gaussian(mu, sigma)
samples = sim_model.sim(n_train_samples * N)

samples = [
    samples[:, :inputs],
    samples[:, inputs:],
]  # first 2 dim conditioned on last N-2 dimensions
sim_model.input_dim = [inputs, givens]

if LOAD_MODEL:
    model = ent.load_model(name=name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")
joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
H_true = joint_H - marginal_H
_ = ent.update_best_model(estimator.model, samples, name=name, path=model_paths)


print("sigma toeplitz, further away values have less weight")
print(f"estimated H: {H}")
print(f"true H: {H_true:.4f}")


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
