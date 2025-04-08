import numpy as np
from datetime import date

import os
from _utils import set_sys_path

set_sys_path()

from simulators.CPDSSS_models import CPDSSS_Gram_Schmidt
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io


SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True

SAVE_FILE = False

"""
Parameters for CPDSSS
"""
N = 3
# L = 3
d0 = int(N / 2)
d1 = int(N / 2)
d0 = 2
d1 = int(N - d0)
T_range = range(2, 11)
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
MI_KL = np.empty((n_trials, len(T_range))) * np.nan
MI_KSG = np.empty((n_trials, len(T_range))) * np.nan
H_XH_KL = np.empty((n_trials, len(T_range))) * np.nan
H_XH_KSG = np.empty((n_trials, len(T_range))) * np.nan
H_XX_KL = np.empty((n_trials, len(T_range))) * np.nan
H_XX_KSG = np.empty((n_trials, len(T_range))) * np.nan


model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/MI(H,X)_orthogonal/conditional_noFading/N{N}_d0d1({d0},{d1})/"
path = base_path  # + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

base_model_path = f"temp_data/saved_models/conditional_noFading_orthogonal/{N}N_d0d1({d0},{d1})"


# fix filename if file already exists
if SAVE_FILE:
    filename = misc.update_filename(path, filename, -1, rename=False)

misc.print_border(f"Starting orthogonal CPDSSS simulation, N: {N}, d0: {d0}, d1: {d1}")

for i in range(n_trials):
    sim_model = CPDSSS_Gram_Schmidt(2, N, d0=d0, d1=d1)
    for k, T in enumerate(T_range):
        name = f"{T}T"
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        # sim_model = CPDSSS_Cond(T, N, d0=d0, d1=d1)
        sim_model.set_T(T)
        # generate base samples based on max dimension
        sim_model.set_XHcond()
        knn_samples = int(max(min_knn_samples, n_train_samples * sim_model.x_dim))
        samples = sim_model.sim(knn_samples)

        """Train H(xT | h, x1:T-1)"""
        misc.print_border(f"1/2 calculating H(xT | h, x1:T-1), T: {T}, iter: {i+1}")
        model_path = os.path.join(base_model_path, "XH")

        model = ent.load_model(name=name, path=model_path) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H, estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples, method="both"
            )
            H_XH_KL[index], H_XH_KSG[index] = H[0], H[1]

        if SAVE_MODEL:
            _ = ent.update_best_model(estimator.model, samples, name=name, path=model_path)

        if SAVE_FILE:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
                os.path.join(path, filename),
            )
        del estimator

        """Train H(xT |x1:T-1)"""
        misc.print_border(f"2/2 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
        model_path = os.path.join(base_model_path, "X")

        sim_model.set_Xcond()
        samples = sim_model.sim(knn_samples, reuse_GQ=True)

        model = ent.load_model(name=name, path=model_path) if REUSE_MODEL else None
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
            _ = ent.update_best_model(estimator.model, samples, name=name, path=model_path)

        if SAVE_FILE:
            util.io.save(
                (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
                os.path.join(path, filename),
            )
        del estimator
