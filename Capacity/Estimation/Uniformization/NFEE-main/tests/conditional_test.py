import numpy as np
from datetime import date
import scipy.linalg as lin

import os
from _utils import set_sys_path

set_sys_path()

import simulators.CPDSSS_models as simMod
from misc_CPDSSS.entropy_util import Cond_MAF as entCondMAF
from misc_CPDSSS.entropy_util import MAF as entMAF
from misc_CPDSSS import util as misc
from ent_est.entropy import UMestimator
import util.io


SAVE_MODEL = True
TRAIN_ONLY = True
REUSE_MODEL = True
LOAD_MODEL = True

SAVE_FILE = True

"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entros.pathy calc
n_train_samples = 100000

N = 6
T = 2
T_range = range(10, 11)
inputs = 2
givens = N - inputs

model = None
n_samples = N * T * n_train_samples

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
# MI = []
# for d0 in range(0,N+1):

#     sim_model = simMod.CPDSSS_Cond(2, N, d0=d0, d1=N-d0)

#     print("starting H(X|h)")
#     sim_model.set_XHcond()
#     samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
#     #samples[1] = np.random.normal(0,1,samples[1].shape)
#     H_XH,estimator = ent.calc_entropy(sim_model, base_samples=samples)[0]
#     # loss = 3.3744
#     # knn = 3.3691
#     H_XH_kl_ksg = ent.knn_entropy(estimator, samples, method="kl_ksg")

#     print("starting H(X)")
#     sim_model.set_Xcond()
#     samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
#     H_X = ent.calc_entropy(sim_model, base_samples=samples)[0]
#     H_X_kl_ksg = ent.knn_entropy(estimator, samples, method="kl_ksg")


#     MI.append(H_X - H_XH)
#     print(f"CPDSSS with d0={d0},d1={N-d0}")
#     print(f"MI: {MI[-1]}")

# print(f"All MI starting from d0=0\n{MI}")

# import sys
# sys.exit()


# n_samples = n_train_samples * 2 * N
# x = np.random.normal(0, 1, (n_samples, N))
# h = np.random.normal(0, 1, (n_samples, N))
# sim_model.set_XHcond()
# samples = [x, h]
# H_XH = ent.calc_entropy(sim_model, base_samples=samples)[0]

# H_X = simMod.Gaussian(0, 1, N).entropy()
# MI = H_X - H_XH
# print("independent gaussian x,h")
# print(f"MI: {MI}")

# import sys

# sys.exit()


misc.print_border("A is random")
"""
y=Ax+n
A is random
Test cond MAF compared to MAF and knn.
Multiple-Antennas and Isotropically Random Unitary Inputs: The Received Signal Density in Closed Form
Babak Hassibi and Thomas L. Marzetta
"""

mu = np.zeros((N))
row = np.ones((N)) * np.exp(-np.arange(N) / 2)
sigma = np.tile(row, (N, 1))
np.fill_diagonal(sigma, 1)
sigma_n = 2  # * np.eye(N)
sigma_A = 1

X_model_path = os.path.join(model_paths, "MAF_X")
XY_model_path = os.path.join(model_paths, "MAF_XY")
cond_model_path = os.path.join(model_paths, "cond_MAF")

random_A_path = "temp_data/conditional_test/random_A/"
today = date.today().strftime("%b_%d")
filename = "cond_data({})".format(today)
if SAVE_FILE:
    filename = misc.update_filename(random_A_path, filename, -1, rename=False)


H_xy_MAF = np.empty((n_trials, len(T_range), 2)) * np.nan
H_x_MAF = np.empty((n_trials, len(T_range), 2)) * np.nan
H_xy_kl_ksg = np.empty((n_trials, len(T_range), 2)) * np.nan
H_x_kl_ksg = np.empty((n_trials, len(T_range), 2)) * np.nan
H_cond_MAF = np.empty((n_trials, len(T_range), 2)) * np.nan
H_xy_true = np.empty((n_trials, len(T_range))) * np.nan
H_y_given_x_true = np.empty((n_trials, len(T_range))) * np.nan

for iter in range(n_trials):
    for Ti, T in enumerate(T_range):

        index = (iter, Ti)
        model_name = f"random_A_{T}T"
        n_samples = 2 * N * T * n_train_samples
        A = np.random.normal(0, np.sqrt(sigma_A), (n_samples, N, N))
        x = np.random.normal(0, 1, (n_samples, N, T))
        n = np.random.normal(0, np.sqrt(sigma_n), (n_samples, N, T))
        y = np.matmul(A, x) + n
        sim_model = simMod.Gaussian(mu, sigma)

        samples = [y.reshape(n_samples, N * T, order="F"), x.reshape(n_samples, N * T, order="F")]
        # covar is symmetric so det is the product (log sum) of eigenvalues
        dets = np.sum(
            np.log(
                np.linalg.eigvalsh(
                    sigma_n * np.eye(T) + sigma_A * np.matmul(x.transpose(0, 2, 1), x)
                )
            ),
            axis=1,
        )
        # gaussian entropy 0.5*log(2 pi e ) + 0.5*log(det(sigma))
        # Matrix form of gaussian
        # Multiple-Antennas and Isotropically Random Unitary Inputs: The Received Signal Density in Closed Form
        H_y_given_x_true[index] = (N * T) / 2 * np.log(2 * np.pi * np.exp(1)) + N / 2 * np.mean(
            dets
        )
        H_x_true = simMod.Gaussian(0, 1, N * T).entropy()
        H_xy_true[index] = H_y_given_x_true[index] + H_x_true

        misc.print_border(f"evaluating joint H(x,y) MAF, T={T}, iter={iter}")
        test_samples = np.concatenate(samples, axis=1)
        model = entMAF.load_model(name=model_name, path=XY_model_path) if LOAD_MODEL else None
        sim_model.input_dim = N * T * 2
        if TRAIN_ONLY:
            n_hiddens = [max(4 * sim_model.x_dim, 200)] * 3
            estimator = entMAF.learn_model(
                sim_model, model, train_samples=test_samples, n_hiddens=n_hiddens
            )
        else:
            H_xy_MAF[index], estimator = entMAF.calc_entropy(
                sim_model, model=model, base_samples=test_samples, method="both"
            )
            H_xy_kl_ksg[index] = entMAF.knn_entropy(estimator, test_samples, method="kl_ksg")
        if SAVE_MODEL:
            _ = entMAF.update_best_model(
                estimator.model, test_samples, name=model_name, path=XY_model_path
            )
        if SAVE_FILE and not TRAIN_ONLY:
            filename = misc.update_filename(random_A_path, filename, iter)
            util.io.save(
                (T_range, H_y_given_x_true, H_xy_MAF, H_xy_kl_ksg, H_x_MAF, H_x_kl_ksg, H_cond_MAF),
                os.path.join(random_A_path, filename),
            )
        print(f"H_xy MAF:\n{H_xy_MAF[index]}")
        print(f"H_xy kl,ksg:\n{H_xy_kl_ksg[index]}")
        print(f"True H: {H_xy_true[index]}")

        misc.print_border(f"evaluating H(x) MAF, T={T}, iter={iter}")
        test_samples = samples[1]
        model = entMAF.load_model(name=model_name, path=X_model_path) if LOAD_MODEL else None
        sim_model.input_dim = N * T
        if TRAIN_ONLY:
            n_hiddens = [max(4 * sim_model.x_dim, 200)] * 3
            estimator = entMAF.learn_model(
                sim_model, model, train_samples=test_samples, n_hiddens=n_hiddens
            )
        else:
            H_x_MAF[index], estimator = entMAF.calc_entropy(
                sim_model, model=model, base_samples=test_samples, method="both"
            )
            H_x_kl_ksg[index] = entMAF.knn_entropy(estimator, test_samples, method="kl_ksg")
        if SAVE_MODEL:
            _ = entMAF.update_best_model(
                estimator.model, test_samples, name=model_name, path=X_model_path
            )
        if SAVE_FILE and not TRAIN_ONLY:
            util.io.save(
                (T_range, H_y_given_x_true, H_xy_MAF, H_xy_kl_ksg, H_x_MAF, H_x_kl_ksg, H_cond_MAF),
                os.path.join(random_A_path, filename),
            )
        print(f"H_x MAF:\n{H_x_MAF[index]}")
        print(f"H_x kl,ksg:\n{H_x_kl_ksg[index]}")
        print(f"True H: {H_x_true}")

        misc.print_border(f"evaluating cond H(y|x) condMAF, T={T}, iter={iter}")
        model = entCondMAF.load_model(name=model_name, path=cond_model_path) if LOAD_MODEL else None
        sim_model.input_dim = [N * T, N * T]
        if TRAIN_ONLY:
            n_hiddens = [max(4 * sim_model.x_dim, 200)] * 3
            estimator = entCondMAF.learn_model(
                sim_model, model, train_samples=samples, n_hiddens=n_hiddens
            )
        else:
            H_cond_MAF[index], estimator = entCondMAF.calc_entropy(
                sim_model, model=model, base_samples=samples, method="both"
            )
        if SAVE_MODEL:
            _ = entCondMAF.update_best_model(
                estimator.model, samples, name=model_name, path=cond_model_path
            )
        if SAVE_FILE and not TRAIN_ONLY:
            util.io.save(
                (T_range, H_y_given_x_true, H_xy_MAF, H_xy_kl_ksg, H_x_MAF, H_x_kl_ksg, H_cond_MAF),
                os.path.join(random_A_path, filename),
            )
        print(f"\ny=Ax+n, A is random T={T}")
        print(f"estimated cond MAF H: {H_cond_MAF[index]}")
        print(f"estimated MAF H_XY - H_X: {H_xy_MAF[index]-H_x_MAF[index]}")
        print(f"estimated knn H_XY - H_X: {H_xy_kl_ksg[index]-H_x_kl_ksg[index]}")
        print(f"true H: {H_y_given_x_true[index]}")

import sys

sys.exit()

misc.print_border("A is constant")

"""
y=Ax+n
A is constant"""
model_name = "constant_A"
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
    model = entCondMAF.load_model(name=model_name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = entCondMAF.calc_entropy(sim_model, base_samples=samples, method="both")
H_y_given_x_true = N / 2 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(lin.det(sigma_n))
if SAVE_MODEL:
    _ = entCondMAF.update_best_model(estimator.model, samples, name=model_name, path=model_paths)

print(f"y=Ax+n, A is constant")
print(f"estimated H: {H}")
print(f"true H: {H_y_given_x_true:.4f}")


# import sys

# sys.exit()

"""gaussian covariance, same diminishing covar for each var"""
misc.print_border("gaussian covar decreasing for each N")

model_name = "gaussian_cov_decreasing"
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
    model = entCondMAF.load_model(name=model_name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = entCondMAF.calc_entropy(sim_model, base_samples=samples, method="both")

joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
H_y_given_x_true = joint_H - marginal_H

if SAVE_MODEL:
    _ = entCondMAF.update_best_model(estimator.model, samples, name=model_name, path=model_paths)

print(f"gaussian covar, sigma always decreasing for each additional N.")
print(f"estimated H: {H}")
print(f"true H: {H_y_given_x_true:.4f}")


"""Toeplitz sigma"""
misc.print_border("gaussian covar toeplitz. high correlation with nearby vars")

model_name = "gaussian_cov_distance_falloff"
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
    model = entCondMAF.load_model(name=model_name, path=model_paths)
    estimator = UMestimator(sim_model, model, samples)
    H = estimator.calc_ent(samples=samples, method="both", SHOW_PDF_PLOTS=True)
else:
    H, estimator = entCondMAF.calc_entropy(sim_model, base_samples=samples, method="both")
joint_H = sim_model.entropy()
marginal_H = 0.5 * np.log(lin.det(sigma[inputs:, inputs:])) + givens / 2 * (1 + np.log(2 * np.pi))
H_y_given_x_true = joint_H - marginal_H
if SAVE_MODEL:
    _ = entCondMAF.update_best_model(estimator.model, samples, name=model_name, path=model_paths)


print("sigma toeplitz, further away values have less weight")
print(f"estimated H: {H}")
print(f"true H: {H_y_given_x_true:.4f}")


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
