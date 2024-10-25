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

RUN_X = True
RUN_HX = True

"""
Parameters for CPDSSS
"""
N = 6
# L = 3
d0 = 3
d1 = N-d0
T_range = range(2, 10)
T_range = [6,7,8,9]
T_range = range(1,6)
T_range = [7,8]
saved_T = []
for item in T_range:
    if item not in saved_T:
        saved_T.append(item)
    if item+1 not in saved_T:
        saved_T.append(item + 1)


def run_CPDSSS(
    sim_model,
    base_samples,
    test_samples=None,
    model_name="",
    model_path="",
):
    model = ent.load_model(name=model_name, path=model_path)
    H, estimator = ent.calc_entropy(sim_model, base_samples=base_samples, model=model)

    samples = base_samples if test_samples is None else test_samples
    _ = ent.update_best_model(estimator.model, samples, name=model_name, path=model_path)

    return H


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
arr_size = (n_trials, len(saved_T))
MI = np.empty(arr_size) * np.nan
H_hxc = np.empty(arr_size) * np.nan
H_xxc = np.empty(arr_size) * np.nan
H_joint = np.empty(arr_size) * np.nan
H_cond = np.empty(arr_size) * np.nan


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
filename = misc.update_filename(path, filename, -1, rename=False)

prev_idx = (0, 0)
for i in range(n_trials):
    for k, T in enumerate(T_range):
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        sim_model = CPDSSS(T, N, d0=d0, d1=d1)
        # generate base samples based on max dimension
        knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
        X, X_T, X_cond, h = sim_model.get_base_X_h(knn_samples)
        joint = np.concatenate((X, h), axis=1)

        """Train H(h,x)"""
        if RUN_HX:
            misc.print_border("1/2 calculating H(h,x), T: {0}, iter: {1}".format(T, i + 1))
            sim_model.set_dim_hxc()
            name = f"{T}T"

            H_HX = run_CPDSSS(sim_model, joint, model_name=name, model_path=XH_path)


            H_joint[i, k] = H_HX
            H_hxc[i, k + 1] = H_HX

            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (saved_T, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )

        """Train H(x)"""
        if RUN_X:
            misc.print_border("2/2 calculating H(x,x_old), T: {0}, iter: {1}".format(T, i + 1))
            sim_model.set_dim_xxc()
            name = f"{T}T"

            H_X = run_CPDSSS(sim_model, X, model_name=name, model_path=XH_path)

            H_xxc[i, k] = H_X
            H_cond[i, k + 1] = H_X

            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (saved_T, MI, H_hxc, H_xxc, H_joint, H_cond, i),
                os.path.join(path, filename),
            )
