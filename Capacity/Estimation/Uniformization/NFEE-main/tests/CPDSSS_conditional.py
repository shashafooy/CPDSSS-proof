import numpy as np
from datetime import date

import os
from _utils import set_sys_path

set_sys_path()

from simulators.CPDSSS_models import CPDSSS_Cond
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io


SAVE_MODEL = True
TRAIN_ONLY = False
KNN_ONLY = False
REUSE_MODEL = True

SAVE_FILE = True

"""
Parameters for CPDSSS
"""
N = 2
# L = 3
d0 = int(N / 2)
d1 = int(N / 2)
d0 = 1
d1 = int(N - d0)
T_range = range(1, 10)
# T_range = range(2, 6)
# T_range = range(5, 7)


"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 200000  # samples to generate per entropy calc
n_train_samples = 100000


"""
Initialize arrays
"""
MI = np.empty((n_trials, len(T_range))) * np.nan
H_h_given_x = np.empty((n_trials, len(T_range))) * np.nan


model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/h_given_x/N{N}_d0d1({d0},{d1})/"
path = base_path  # + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path_h_given_x = f"temp_data/saved_models/h_given_x/{N}N_d0d1({d0},{d1})"
model_path_x_given_h = f"temp_data/saved_models/x_given_h/{N}N_d0d1({d0},{d1})"
model_path_x = f"temp_data/saved_models/X/{N}N_d0d1({d0},{d1})"


# fix filename if file already exists
if SAVE_FILE:
    filename = misc.update_filename(path, filename, -1, rename=False)

misc.print_border(f"Evaluating N={N}, d0={d0}, d1={d1}")

for i in range(n_trials):
    sim_model = CPDSSS_Cond(2, N, d0=d0, d1=d1, use_fading=True)

    # sim_model.set_T(1)
    # sim_model.set_Xcond()
    # samples = sim_model.sim(n_train_samples * sim_model.x_dim)
    # misc.print_border(f"H(X)")
    # # H_X = 4.256
    # H_X, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")

    # misc.print_border(f"H(X,h)")
    # samples[0] = np.concatenate((samples[0], sim_model.h), axis=1)
    # sim_model.input_dim[0] = 2 * N
    # sim_model.update_x_dim()
    # H_XH, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")

    # misc.print_border(f"H(h)")
    # H_h = sim_model.chan_entropy()

    for k, T in enumerate(T_range):
        name = f"{T}T"
        index = (i, k)

        misc.print_border(f"Evaluating H(h|x) T={T}, iter: {i+1}, d0={d0}, d1={d1}")
        print(f"H(h) = {sim_model.chan_entropy()}")
        sim_model.set_T(T)
        sim_model.set_H_given_X()
        samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
        model = ent.load_model(name=name, path=model_path_h_given_x) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H_h_given_x[index], estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, KNN_only=KNN_ONLY
            )
            MI[index] = sim_model.chan_entropy() - H_h_given_x[index]
            MI_h = sim_model.chan_entropy() - H_h_given_x[index]
        if SAVE_MODEL:
            _ = ent.update_best_model(
                estimator.model, sim_model=sim_model, name=name, path=model_path_h_given_x
            )

        if SAVE_FILE:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI, H_h_given_x),
                os.path.join(path, filename),
            )

        misc.print_border(f"Evaluating H(x|h) T={T}, iter: {i+1}, d0={d0}, d1={d1}")
        sim_model.set_X_given_H()
        samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
        model = ent.load_model(name=name, path=model_path_x_given_h) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H_x_given_h, estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, KNN_only=KNN_ONLY
            )
        if SAVE_MODEL:
            _ = ent.update_best_model(
                estimator.model, sim_model=sim_model, name=name, path=model_path_x_given_h
            )

        misc.print_border(f"Evaluating H(x) T={T}, iter: {i+1}, d0={d0}, d1={d1}")
        sim_model.set_X_no_givens()
        samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
        model = ent.load_model(name=name, path=model_path_x) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H_x, estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, KNN_only=KNN_ONLY
            )
        if SAVE_MODEL:
            _ = ent.update_best_model(
                estimator.model, sim_model=sim_model, name=name, path=model_path_x
            )

        MI_x = H_x - H_x_given_h

        print(f"MI = H(h) - H(h|x) = {sim_model.chan_entropy()} - {H_h_given_x[index]} = {MI_h}")
        print(f"MI = H(x) - H(x|h) = {H_x} - {H_x_given_h} = {MI_x}")

        continue

        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        # sim_model = CPDSSS_Cond(T, N, d0=d0, d1=d1)
        sim_model.set_T(T)
        # generate base samples based on max dimension
        sim_model.set_XHcond()
        knn_samples = int(max(min_knn_samples, n_train_samples * sim_model.x_dim))
        samples = sim_model.sim(knn_samples, reuse_GQ=True)

        """Train H(xT | h, x1:T-1)"""
        misc.print_border(f"1/2 calculating H(xT | h, x1:T-1), T: {T}, iter: {i+1}")
        model_path_h_given_x = os.path.join(base_model_path, "XH")

        model = ent.load_model(name=name, path=model_path_h_given_x) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H, estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, method="both"
            )
            H_XH_KL[index], H_XH_KSG[index] = H[0], H[1]

        if SAVE_MODEL:
            _ = ent.update_best_model(
                estimator.model, samples, name=name, path=model_path_h_given_x
            )

        if SAVE_FILE:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
                os.path.join(path, filename),
            )
        del estimator

        """Train H(xT |x1:T-1)"""
        misc.print_border(f"2/2 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
        model_path_h_given_x = os.path.join(base_model_path, "X")

        sim_model.set_Xcond()
        samples = sim_model.sim(knn_samples, reuse_GQ=True)

        model = ent.load_model(name=name, path=model_path_h_given_x) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H, estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, method="both"
            )
            H_XX_KL[index], H_XX_KSG[index] = H[0], H[1]
        MI_KL[index] = H_XX_KL[index] - H_XH_KL[index]
        MI_KSG[index] = H_XX_KSG[index] - H_XH_KSG[index]

        if SAVE_MODEL:
            _ = ent.update_best_model(
                estimator.model, samples, name=name, path=model_path_h_given_x
            )

        if SAVE_FILE:
            util.io.save(
                (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
                os.path.join(path, filename),
            )
        del estimator
