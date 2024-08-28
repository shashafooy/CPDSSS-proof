import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc

from datetime import date

import configparser

config = configparser.ConfigParser()
config.read("CPDSSS.ini")
KNN_THREADING = not config["GLOBAL"].getboolean("knn_GPU", False)  # Use threading if GPU not used
SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True


def run_CPDSSS(
    sim_model,
    base_samples,
    test_samples=None,
    old_thread=None,
    old_correction=0,
    model_name="",
    model_path="",
):
    model = ent.load_model(name=model_name, path=model_path) if REUSE_MODEL else None
    if TRAIN_ONLY:
        estimator = ent.learn_model(sim_model, pretrained_model=model, train_samples=base_samples)
        H = 0
    else:
        if KNN_THREADING:
            new_thread, new_correction, estimator = ent.calc_entropy_thread(
                sim_model, base_samples=base_samples, model=model
            )
            if old_thread is not None:  # don't run if first iteration
                knn = old_thread.get_result()
                H = knn + old_correction
        else:
            H, estimator = ent.calc_entropy(sim_model, base_samples=base_samples, model=model)
    if SAVE_MODEL:
        samples = base_samples if test_samples is None else test_samples
        _ = ent.update_best_model(estimator.model, samples, name=model_name, path=model_path)

    if KNN_THREADING:
        return H, new_thread, new_correction
    else:
        return H


"""
Parameters for CPDSSS
"""
N = 2
L = 2
M = int(N / L)
P = N - int(N / L)
T_range = range(8, 10)


"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entropy calc
n_train_samples = 100000


"""
Initialize arrays
"""
MI_tKL = np.empty(len(T_range))
MI_means = np.empty(len(T_range))
MI = np.empty((n_trials, len(T_range))) * np.nan
H_hxc = np.empty((n_trials, len(T_range))) * np.nan
H_xxc = np.empty((n_trials, len(T_range))) * np.nan
H_joint = np.empty((n_trials, len(T_range))) * np.nan
H_cond = np.empty((n_trials, len(T_range))) * np.nan
H_x = np.empty((n_trials, len(T_range))) * np.nan
H_h = np.empty((n_trials, len(T_range))) * np.nan
H_xh = np.empty((n_trials, len(T_range))) * np.nan

best_trn_loss = np.ones((T_range[-1], 4)) * 1e5

H_hxc_thread = None
H_cond_thread = None
H_xxc_thread = None
H_joint_thread = None
H_joint_correction = 0
model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/MI(h,X)/N{N}_L{L}/"

path = base_path + "coarse-fine_75k_x_dims"
path = base_path + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path = f"temp_data/saved_models/{N}N"
X_path = os.path.join(model_path, "X")
XH_path = os.path.join(model_path, "XH")


# fix filename if file already exists
if not TRAIN_ONLY:
    filename = misc.update_filename(path, filename, -1, rename=False)
# model = ent.load_model(8,'CPDSSS_hxc_2T','temp_data/saved_models/2T')


prev_idx = (0, 0)
for i in range(n_trials):
    for k, T in enumerate(T_range):
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        sim_model = CPDSSS(T, N, L)
        # generate base samples based on max dimension
        sim_model.set_dim_joint()
        knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
        X, X_T, X_cond, h = sim_model.get_base_X_h(knn_samples)
        hxc = np.concatenate((X_cond, h), axis=1)
        joint = np.concatenate((X, h), axis=1)

        # generate independent test set on first iteration used for model testing
        # if i == 0:
        #     test_X, _, test_cond, test_h = sim_model.get_base_X_h(knn_samples)
        #     test_hxc = np.concatenate((test_cond, test_h), axis=1)
        #     test_joint = np.concatenate((test_X, test_h), axis=1)

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

        if KNN_THREADING:
            H_joint[prev_idx], H_hxc_thread, H_hxc_correction = run_CPDSSS(
                sim_model,
                hxc,
                old_thread=H_joint_thread,
                old_correction=H_joint_correction,
                model_name=name,
                model_path=XH_path,
            )
            MI[prev_idx] = H_hxc[prev_idx] + H_xxc[prev_idx] - H_joint[prev_idx] - H_cond[prev_idx]
        else:
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
        if KNN_THREADING:
            H_hxc[index], H_cond_thread, H_cond_correction = run_CPDSSS(
                sim_model,
                X_cond,
                old_thread=H_hxc_thread,
                old_correction=H_hxc_correction,
                model_name=name,
                model_path=X_path,
            )
        else:
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

        if KNN_THREADING:
            H_cond[index], H_xxc_thread, H_xxc_correction = run_CPDSSS(
                sim_model,
                X,
                old_thread=H_cond_thread,
                old_correction=H_cond_correction,
                model_name=name,
                model_path=X_path,
            )
        else:
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

        if KNN_THREADING:
            H_xxc[index], H_joint_thread, H_joint_correction = run_CPDSSS(
                sim_model, joint, H_xxc_thread, H_xxc_correction, name, XH_path
            )
            run_CPDSSS(
                sim_model,
                joint,
                old_thread=H_xxc_thread,
                old_correction=H_xxc_correction,
                model_name=name,
                model_path=XH_path,
            )
        else:
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

        # Save this index set for next iteration
        prev_idx = index

    if not TRAIN_ONLY:
        filename = misc.update_filename(path, filename, i + 1)


# get knn H(joint) from final iteraiton
knn = H_joint_thread.get_result()
H_joint[prev_idx] = knn + H_joint_correction
# Combine entropies for mutual information
MI[prev_idx] = H_hxc[prev_idx] + H_xxc[prev_idx] - H_joint[prev_idx] - H_cond[prev_idx]
