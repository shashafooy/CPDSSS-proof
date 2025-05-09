import numpy as np
from datetime import date

import os
from _utils import set_sys_path

set_sys_path()

import simulators.CPDSSS_models as models  # imort CPDSSS_Cond, CPDSSS_Cond_Complex
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io
import util.misc


SAVE_MODEL = True
TRAIN_ONLY = False
KNN_ONLY = False
REUSE_MODEL = True

SAVE_FILE = True

"""
Parameters for CPDSSS
"""
N = 6
# L = 3
d0 = int(N / 2)
d1 = int(N / 2)
d0 = 3
d1 = int(N - d0)
T_range = range(1, 10)
# T_range = range(2, 6)
# T_range = range(5, 7)


"""
Number of iterations
"""
n_trials = 5  # iterations to average
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
base_path = f"temp_data/CPDSSS_data/h_given_x_orthonormal/N{N}_d0d1({d0},{d1})/"
path = base_path  # + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path_h_given_x = f"temp_data/saved_models/h_given_x_orthonormal/{N}N_d0d1({d0},{d1})"
# model_path_x_given_h = f"temp_data/saved_models/x_given_h/{N}N_d0d1({d0},{d1})"
# model_path_x = f"temp_data/saved_models/X/{N}N_d0d1({d0},{d1})"


# fix filename if file already exists
if SAVE_FILE:
    filename = misc.update_filename(path, filename, -1, rename=False)

misc.print_border(f"Evaluating N={N}, d0={d0}, d1={d1}")

for i in range(n_trials):
    sim_model = models.CPDSSS_Cond(2, N, d0=d0, d1=d1, use_fading=True, whiten=False)
    # sim_model.set_T(1)
    # sim_model.set_H_given_X()
    # sim_model.sim(1000)
    # roughly equivalent to N=6, d0d1=(4,2)
    # sim_model_ZC = models.CPDSSS_Cond_Complex(0, 2, d0=1, d1=1, use_fading=True)
    # sim_model_ZC = models.CPDSSS_Cond_Complex(0, 2, L=2, whiten=True)
    # sim_model_ZC = models.CPDSSS_Gram_Schmidt(0, 2, 2)

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

        # sim_model_ZC.set_T(T)
        # sim_model_ZC.set_H_given_X()
        # samples = sim_model_ZC.sim(n_train_samples * sim_model_ZC.x_dim)
        # H_hx, estimator = ent.calc_entropy(
        #     sim_model_ZC, base_samples=np.concatenate(samples, axis=1), KNN_only=True, method="ksg"
        # )
        # H_x, estimator = ent.calc_entropy(
        #     sim_model_ZC, base_samples=samples[1], KNN_only=True, method="ksg"
        # )
        # MI = sim_model_ZC.chan_entropy() - (H_hx - H_x)
        # print(f"complex MI T={T}, N=2, d0d1=(1,1):\nH(h)={sim_model_ZC.chan_entropy()}")
        # print(f"MI(h,x)={MI}")
        # print(f"Capacity {MI / sim_model_ZC.chan_entropy() * 100:.1f}%")

        # H_ZC, estimator = ent.calc_entropy(sim_model_ZC, base_samples=samples)
        # MI = sim_model_ZC.chan_entropy() - H_ZC
        # if SAVE_MODEL:
        #     ent.update_best_model(
        #         estimator.model, sim_model=sim_model_ZC, name=name, path=model_path_h_given_x
        #     )
        # print(f"complex MI T={T}, N=2, d0d1=(1,1):\nH(h)={sim_model_ZC.chan_entropy()}")
        # print(f"MI(h,x)={MI}")
        # print(f"Capacity {MI / sim_model_ZC.chan_entropy() * 100}%")

        # continue

        misc.print_border(f"Evaluating H(h|x) T={T}, iter: {i+1}, d0={d0}, d1={d1}")
        print(f"H(h) = {sim_model.chan_entropy()}")
        sim_model.set_T(T)
        sim_model.set_H_given_X()
        samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
        model = ent.load_model(name=name, path=model_path_h_given_x) if REUSE_MODEL else None

        if TRAIN_ONLY:
            estimator = ent.learn_model(
                sim_model, model, train_samples=samples, fine_tune_only=False
            )
        else:
            H_h_given_x[index], estimator = ent.calc_entropy(
                sim_model,
                model=model,
                base_samples=samples,
                KNN_only=KNN_ONLY,
                fine_tune_only=False,
            )
            MI[index] = sim_model.chan_entropy() - H_h_given_x[index]
            MI_h = sim_model.chan_entropy() - H_h_given_x[index]
        if SAVE_MODEL:
            samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)
            _ = ent.update_best_model(
                estimator.model, samples=samples, name=name, path=model_path_h_given_x
            )

        if SAVE_FILE:
            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (T_range, MI, H_h_given_x),
                os.path.join(path, filename),
            )

        # old CPDSSS MI=0.289 cap=10.82%
        print(f"MI=H(H) - H(H|X): {MI[index]}")
        print(f"Capacity: {MI[index]/sim_model.chan_entropy()*100:.2f}%")

        continue

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

        print(f"Capacity: {MI[index]/sim_model.chan_entropy()*100:.2f}%")

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
