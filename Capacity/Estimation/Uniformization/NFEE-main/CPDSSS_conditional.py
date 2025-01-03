import numpy as np

import util.io
import os

from simulators.CPDSSS_models import CPDSSS_Cond
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc

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
    model = ent.load_Cond_MAF_model(name=model_name, path=model_path) if REUSE_MODEL else None

    estimator = ent.learn_cond_MAF_model(
        sim_model, pretrained_model=model, train_samples=base_samples
    )
    if TRAIN_ONLY:
        H = 0
    else:
        H = estimator.calc_ent(samples=base_samples)
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
H_XH = np.empty((n_trials, len(T_range))) * np.nan
H_XX = np.empty((n_trials, len(T_range))) * np.nan


model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/MI(h,X)/conditional/N{N}_d0d1({d0},{d1})/"
path = base_path  # + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path = f"temp_data/saved_models/{N}N_d0d1({d0},{d1})"
X_path = os.path.join(model_path, "X")
XH_path = os.path.join(model_path, "XH")


# fix filename if file already exists

filename = misc.update_filename(path, filename, -1, rename=False)

# T = 2
# for T in range(2, 5):
#     sim_model = CPDSSS_Cond(T, N, 2)
#     n_samples = 100000
#     sim_model.set_Xcond()
#     xxcond = sim_model.sim(n_samples)
#     estimator = ent.learn_cond_MAF_model(sim_model, train_samples=xxcond)
#     H_Xcond = estimator.model.eval_trnloss(xxcond)
#     print(f"T={T}, H(XT|X1...XT-1) = {H_Xcond:.3f}")

#     sim_model.set_XHcond()
#     xhcond = sim_model.sim(n_samples, reuse=True)
#     estimator = ent.learn_cond_MAF_model(sim_model, train_samples=xhcond)
#     H_XHcond = estimator.model.eval_trnloss(xhcond)
#     print(f"H(XT|H,X1...XT-1) = {H_XHcond:.3f}")

#     print(f"MI T={T} is {H_Xcond - H_XHcond:.3f}")


for i in range(n_trials):
    for k, T in enumerate(T_range):
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        sim_model = CPDSSS_Cond(T, N, d0=d0, d1=d1)
        # generate base samples based on max dimension
        sim_model.set_XHcond()
        knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))

        """Train H(xT | h, x1:T-1)"""
        misc.print_border(f"1/4 calculating H(xT | h, x1:T-1), T: {T}, iter: {i+1}")
        name = f"{T-1}T"

        samples = sim_model.sim(knn_samples)
        estimator = ent.learn_cond_MAF_model(sim_model, train_samples=samples)
        H_XH[index] = estimator.calc_ent(samples=samples)

        filename = misc.update_filename(path, filename, i)
        util.io.save(
            (T_range, MI, H_XH, H_XX, i),
            os.path.join(path, filename),
        )

        """Train H(xT |x1:T-1)"""
        misc.print_border(f"1/4 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
        name = f"{T-1}T"

        sim_model.set_Xcond()
        samples = sim_model.sim(reuse=True)
        estimator = ent.learn_cond_MAF_model(sim_model, train_samples=samples)
        H_XX[index] = estimator.calc_ent(samples=samples)
        MI[index] = H_XX - H_XH

        util.io.save(
            (T_range, MI, H_XH, H_XX, i),
            os.path.join(path, filename),
        )
