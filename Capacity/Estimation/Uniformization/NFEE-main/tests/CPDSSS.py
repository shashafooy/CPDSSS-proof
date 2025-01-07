import numpy as np


import os
from _utils import set_sys_path

set_sys_path()

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS.entropy_util import MAF as ent
from misc_CPDSSS import util as misc
import util.io
from datetime import date


SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True


def run_CPDSSS(
    sim_model,
    base_samples,
    test_samples=None,
    model_name="",
    model_path="",
):
    model = ent.load_model(name=model_name, path=model_path) if REUSE_MODEL else None
    if TRAIN_ONLY:
        estimator = ent.learn_model(sim_model, model=model, train_samples=base_samples)
        H = 0
    else:
        H, estimator = ent.calc_entropy(sim_model, base_samples=base_samples, model=model)
    if SAVE_MODEL:
        samples = base_samples if test_samples is None else test_samples
        _ = ent.update_best_model(estimator.model, samples, name=model_name, path=model_path)

    return H


"""
Parameters for CPDSSS
"""
N = 6
# L = 3
d0 = 4
d1 = 2
T_range = range(2, 10)
# T_range = range(5, 7)


"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entropy calc
n_train_samples = 100000


"""
Initialize arrays
"""
MI = np.empty((n_trials, len(T_range))) * np.nan
H_hxc = np.empty((n_trials, len(T_range))) * np.nan
H_xxc = np.empty((n_trials, len(T_range))) * np.nan
H_joint = np.empty((n_trials, len(T_range))) * np.nan
H_cond = np.empty((n_trials, len(T_range))) * np.nan


model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_d0d1({d0},{d1})/"
path = base_path + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path = f"temp_data/saved_models/{N}N_d0d1({d0},{d1})"
X_path = os.path.join(model_path, "X")
XH_path = os.path.join(model_path, "XH")


# fix filename if file already exists
if not TRAIN_ONLY:
    filename = misc.update_filename(path, filename, -1, rename=False)

for i in range(n_trials):
    for k, T in enumerate(T_range):
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        sim_model = CPDSSS(T, N, d0=d0, d1=d1)
        # generate base samples based on max dimension
        sim_model.set_dim_joint()
        knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
        X, X_T, X_cond, h = sim_model.get_base_X_h(knn_samples)
        hxc = np.concatenate((X_cond, h), axis=1)
        joint = np.concatenate((X, h), axis=1)

        """Calculate entropies needed for mutual information. Evaluate knn entropy (CPU) while training new model (GPU)
            General flow: 
                Train model (main thread)
                evaluate uniform points and run knn (background thread)
                wait for previoius knn thread to finish (main thread)
                combine knn entropy with jacobian correction term (main thread)
                Start new model while current knn is running (main thread)
         """

        """Train H(h,x_cond)"""
        misc.print_border("1/4 calculating H(h,x_old), T: {0}, iter: {1}".format(T, i + 1))
        sim_model.set_dim_hxc()
        name = f"{T-1}T"

        H_hxc[index] = run_CPDSSS(
            sim_model,
            hxc,
            model_name=name,
            model_path=XH_path,
        )

        if not TRAIN_ONLY:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )

        """Train H(x_cond)"""
        misc.print_border("2/4 calculating H(x_old), T: {0}, iter: {1}".format(T, i + 1))

        sim_model.set_dim_cond()
        name = f"{T-1}T"

        H_cond[index] = run_CPDSSS(
            sim_model,
            X_cond,
            model_name=name,
            model_path=X_path,
        )

        if not TRAIN_ONLY:
            util.io.save(
                (T_range, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )

        """Train H(x_T,x_cond)"""
        misc.print_border("3/4 calculating H(x_T, x_old), T: {0}, iter: {1}".format(T, i + 1))
        sim_model.set_dim_xxc()
        name = f"{T}T"

        H_xxc[index] = run_CPDSSS(
            sim_model,
            X,
            model_name=name,
            model_path=X_path,
        )
        if not TRAIN_ONLY:
            util.io.save(
                (T_range, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )
        """Train H(h,x_T,x_cond)"""
        misc.print_border("4/4 calculating H_(h,x_T,x_old), T: {0}, iter: {1}".format(T, i + 1))
        sim_model.set_dim_joint()
        name = f"{T}T"

        H_joint[index] = run_CPDSSS(
            sim_model,
            joint,
            model_name=name,
            model_path=XH_path,
        )
        MI[index] = H_hxc[index] + H_xxc[index] - H_joint[index] - H_cond[index]

        if not TRAIN_ONLY:
            util.io.save(
                (T_range, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )

    if not TRAIN_ONLY:
        filename = misc.update_filename(path, filename, i + 1)
