import numpy as np
from scipy import stats

import util.io
import os

from ent_est import entropy
from ent_est.entropy import tkl

from simulators.complex import mvn
from simulators.CPDSSS_models import CPDSSS_XS
from datetime import date
from datetime import timedelta

import time
import gc
import math

"""
Functions for generating the model and entropy
"""


def UM_KL_Gaussian(x):
    std_x = np.std(x, axis=0)
    z = stats.norm.cdf(x)
    return tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x), axis=1)))


def create_model(n_inputs, rng):
    n_hiddens = [100, 100]
    act_fun = "tanh"
    n_mades = 10

    import ml.models.mafs as mafs

    return mafs.MaskedAutoregressiveFlow(
        n_inputs=n_inputs,
        n_hiddens=n_hiddens,
        act_fun=act_fun,
        n_mades=n_mades,
        mode="random",
        rng=rng,
    )


def calc_entropy(sim_model, base_samples=None, n_samples=100):
    H = -1
    val_tol = 0.05
    patience = 10
    # redo learning if calc_ent returns error
    while H == -1:
        net = create_model(sim_model.x_dim, rng=np.random)
        estimator = entropy.UMestimator(sim_model, net)
        start_time = time.time()
        # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
        estimator.learn_transformation(
            n_samples=int(n_samples * sim_model.x_dim), val_tol=val_tol, patience=patience
        )
        end_time = time.time()
        print("learning time: ", str(timedelta(seconds=int(end_time - start_time))))
        estimator.samples = estimator.samples if base_samples is None else base_samples
        reuse = False if base_samples is None else True
        start_time = time.time()
        H, _, _, _ = estimator.calc_ent(reuse_samples=reuse, method="umtkl", k=1)
        end_time = time.time()
        print("knn time: ", str(timedelta(seconds=int(end_time - start_time))))

        net.release_shared_data()
        for i in range(3):
            gc.collect()

    return H


def update_filename(path, old_name, n_samples, today, iter, rename=True):
    new_name = "CPDSSS_data_dump({0}_iter)({1}k_samples)({2})".format(
        iter, int(n_samples / 1000), today
    )
    unique_name = new_name
    # Check if name already exists, append number to end until we obtain new name
    i = 1
    while os.path.isfile(os.path.join(path, unique_name + ".pkl")):
        unique_name = new_name + "_" + str(i)
        i = i + 1
    new_name = unique_name
    if rename:
        os.rename(os.path.join(path, old_name + ".pkl"), os.path.join(path, new_name + ".pkl"))
    return new_name


def print_border(msg):
    print("-" * len(msg) + "\n" + msg + "\n" + "-" * len(msg))


"""
Parameters for CPDSSS
"""
N = 2
L = 2
M = int(N / L)
P = N - int(N / L)

"""
Number of iterations
"""
n_trials = 100  # iterations to average
knn_samples = 200000  # samples to generate per entropy calc
n_train_samples = 50000
completed_iter = 0

"""
Initialize arrays
"""
MI_xs = np.empty((n_trials)) * np.nan
H_x = np.empty((n_trials)) * np.nan
H_s = np.empty((n_trials)) * np.nan
H_xs = np.empty((n_trials)) * np.nan


"""
File names
"""

today = date.today().strftime("%b_%d")

base_path = "temp_data/CPDSSS_data/MI(S,X)/N2_L2"
path = base_path

# fix filename if file already exists
filename = update_filename(path, "", knn_samples, today, completed_iter, rename=False)
# create initial file
util.io.save((MI_xs, H_x, H_s, H_xs, 0), os.path.join(path, filename))


"""
Generate data
"""
for i in range(n_trials):

    sim_model = CPDSSS_XS(N, L)

    n_sims = knn_samples

    sim_model.sim_use_XS()
    XS = sim_model.sim(n_sims)
    X = XS[:, : sim_model.N]
    S = XS[:, -sim_model.sym_N :]

    sim_model.sim_use_X()
    print_border("Calculate H(x), iter: {}".format(i + 1))
    H_x[i] = calc_entropy(sim_model=sim_model, n_samples=n_train_samples, base_samples=X)

    H_s[i] = 0.5 * np.log(np.linalg.det(2 * math.pi * np.exp(1) * np.eye(sim_model.sym_N)))

    sim_model.sim_use_XS()
    print_border("Calculate H(x,s), iter: {}".format(i + 1))
    H_xs[i] = calc_entropy(sim_model=sim_model, n_samples=n_train_samples, base_samples=XS)

    MI_xs[i] = H_x[i] + H_s[i] - H_xs[i]

    completed_iter = completed_iter + 1
    filename = update_filename(path, filename, n_sims, today, completed_iter)

    util.io.save((MI_xs, H_x, H_s, H_xs, i), os.path.join(path, filename))
