import numpy as np
from datetime import date

import os
from _utils import set_sys_path

set_sys_path()

from simulators.CPDSSS_models import CPDSSS_Gram_Schmidt
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io


SAVE_MODEL = False
TRAIN_ONLY = False
REUSE_MODEL = False

SAVE_FILE = False

"""
Parameters for CPDSSS
"""
N = 3
# L = 3
d0 = int(N / 2)
d1 = int(N / 2)
d0 = 1
d1 = int(N - d0)
T_range = range(1, 2)
# T_range = range(2, 6)
# T_range = range(5, 7)


"""
Number of iterations
"""
n_trials = 1  # iterations to average
min_knn_samples = 200000  # samples to generate per entropy calc
n_samples = 10000


"""
Initialize arrays
"""
MI_KL = np.empty((n_trials, len(T_range))) * np.nan
MI_KSG = np.empty((n_trials, len(T_range))) * np.nan
H_XH_KL = np.empty((n_trials, len(T_range))) * np.nan
H_XH_KSG = np.empty((n_trials, len(T_range))) * np.nan
H_XX_KL = np.empty((n_trials, len(T_range))) * np.nan
H_XX_KSG = np.empty((n_trials, len(T_range))) * np.nan

H_H_given_X = np.empty((n_trials, len(T_range))) * np.nan


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
        sim_model.set_H_given_X()
        # samples = sim_model.sim(100000 * sim_model.x_dim)
        # H = ent.calc_entropy(sim_model, base_samples=samples, method="both")[0]
        # print(sim_model.chan_entropy() - H)
        # generate base samples based on max dimension
        sim_model.set_x_given_h_oldX()
        # samples = sim_model.sim(knn_samples)

        sim_model.set_T(1)
        # sim_model.set_x_given_oldX()
        # samples = sim_model.sim(n_samples * sim_model.x_dim)
        # misc.print_border(f"H(X)")
        H_X = 4.256  # N=3, d0d1=(1,2)
        # H_X, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="both")
        # H_X = ent.calc_entropy(sim_model, base_samples=samples[0], KNN_only=True, method="ksg")[0]

        # misc.print_border(f"H(X,h)")
        # sim_model.set_HX()
        # samples = sim_model.sim(n_samples * sim_model.x_dim)
        # samples[0] = np.concatenate((samples[0], sim_model.h), axis=1)
        # sim_model.input_dim[0] = N * N + N
        # sim_model.update_x_dim()
        H_XH = 16.944  # N=3, d0d1=(1,2) KNN
        H_XH = 17.0333  # N=3, d0d1=(1,2) MAF
        # H_XH, estimator = ent.calc_entropy(sim_model, base_samples=samples, method="umtksg")
        # H_XH = ent.calc_entropy(sim_model, base_samples=samples[0], KNN_only=True, method="ksg")[0]

        # misc.print_border(f"H(h)")
        H_h = sim_model.chan_entropy()

        # MI = H_X + H_h - H_XH
        # print(f"MI=H(X)+H(H)-H(X,H): {MI}")
        # print(f"Capacity: {MI/H_h*100:.2f}%")

        sim_model.set_H_given_X()
        # samples = sim_model.sim(n_samples * sim_model.x_dim)
        H_cond = 12.7737  # N=3, d0d1=(1,2) cond MAF
        # H_H_given_X = ent.calc_entropy(sim_model, base_samples=samples)[0]

        MI = H_h - H_cond
        # print(f"MI=H(H) - H(H|X): {MI}")
        # print(f"Capacity: {MI/H_h*100:.2f}%")

        """Train H(xT | h, x1:T-1)"""
        misc.print_border(f"calculating H(h | x1:T-1), T: {T}, iter: {i+1}")
        model_path = os.path.join(base_model_path, "H_given_X")
        sim_model.set_H_given_X()
        samples = sim_model.sim(n_samples * sim_model.x_dim)

        model = ent.load_model(name=name, path=model_path) if REUSE_MODEL else None
        if TRAIN_ONLY:
            estimator = ent.learn_model(sim_model, model, train_samples=samples)
        else:
            H_H_given_X[index], estimator = ent.calc_entropy(
                sim_model, model=model, base_samples=samples
            )
            # H_XH_KL[index], H_XH_KSG[index] = H[0], H[1]

        if SAVE_MODEL:
            _ = ent.update_best_model(estimator.model, samples, name=name, path=model_path)

        if SAVE_FILE:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
                os.path.join(path, filename),
            )
        del estimator

        H_h = sim_model.chan_entropy()
        MI = H_h - H_H_given_X[index]
        print(f"MI=H(H) - H(H|X): {MI}")
        print(f"Capacity: {MI/H_h*100:.2f}%")

        # """Train H(xT |x1:T-1)"""
        # misc.print_border(f"2/2 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
        # model_path = os.path.join(base_model_path, "X")

        # sim_model.set_x_given_oldX()
        # samples = sim_model.sim(knn_samples)

        # model = ent.load_model(name=name, path=model_path) if REUSE_MODEL else None
        # if TRAIN_ONLY:
        #     estimator = ent.learn_model(sim_model, model, train_samples=samples)
        # else:
        #     H, estimator = ent.calc_entropy(
        #         sim_model, model=model, base_samples=samples, method="both"
        #     )
        #     H_XX_KL[index], H_XX_KSG[index] = H[0], H[1]
        # MI_KL[index] = H_XX_KL[index] - H_XH_KL[index]
        # MI_KSG[index] = H_XX_KSG[index] - H_XH_KSG[index]

        # if SAVE_MODEL:
        #     _ = ent.update_best_model(estimator.model, samples, name=name, path=model_path)

        # if SAVE_FILE:
        #     util.io.save(
        #         (T_range, MI_KL, MI_KSG, H_XH_KL, H_XH_KSG, H_XX_KL, H_XX_KSG, i),
        #         os.path.join(path, filename),
        #     )
        # del estimator
