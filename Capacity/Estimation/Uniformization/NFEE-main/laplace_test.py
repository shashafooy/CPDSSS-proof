from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl_ksg, kl, ksg

TRAIN_ONLY = False


# knn_samples = 200000
n_train_samples = 100000
n_trials = 100
N_range = range(1, 20)
method = "both"
patience = 5

H_unif_KL = np.empty((n_trials, len(N_range))) * np.nan
H_unif_KSG = np.empty((n_trials, len(N_range))) * np.nan
H_KL_laplace = np.empty((n_trials, len(N_range))) * np.nan
H_KSG_laplace = np.empty((n_trials, len(N_range))) * np.nan

iter = 0
MSE_uniform = np.inf
MSE_KL = np.inf

path = "temp_data/laplace_test/high_epoch"
today = date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

model_path = "temp_data/saved_models/laplace"
new_model_path = "temp_data/saved_models/new_models/laplace"


for i in range(n_trials):
    for ni, N in enumerate(N_range):
        model_name = f"{N}N"

        knn_samples = n_train_samples * N
        sim_laplace = mod.Laplace(mu=0, b=2, N=N)
        true_H_laplace = sim_laplace.entropy()
        laplace_base = sim_laplace.sim(n_samples=knn_samples)

        if TRAIN_ONLY:
            print(f"training laplace N= {N}")
            estimator = ent.learn_model(sim_model=sim_laplace, train_samples=laplace_base)
            trn_loss = ent.update_best_model(
                estimator.model, laplace_base, name=model_name, path=model_path
            )
            print(f"current train loss: {trn_loss}")
            print(f"theoretical entropy: {sim_laplace.entropy()}")
        else:
            misc.print_border("Calculate H(x) laplace, N={}, iter: {}".format(N, i + 1))
            model = ent.load_model(name=model_name, path=model_path, sim_model=sim_laplace)
            if method == "umtksg":
                H_unif_KSG[i, ni], estimator = ent.calc_entropy(
                    sim_model=sim_laplace,
                    base_samples=laplace_base,
                    method=method,
                    model=model,
                )
                H_KSG_laplace[i, ni] = ksg(laplace_base)
                print(f"error: {np.abs(true_H_laplace - H_KSG_laplace[i,ni])}")
            elif method == "umtkl":
                H_unif_KL[i, ni], estimator = ent.calc_entropy(
                    sim_model=sim_laplace,
                    base_samples=laplace_base,
                    method=method,
                    model=model,
                )
                H_KL_laplace[i, ni] = kl(laplace_base)
                print(f"error: {np.abs(true_H_laplace - H_KL_laplace[i,ni])}")

            else:  # Both methods
                H_unif_KL[i, ni], H_unif_KSG[i, ni], estimator = ent.calc_entropy(
                    sim_model=sim_laplace,
                    base_samples=laplace_base,
                    method=method,
                    model=model,
                )
                H_KL_laplace[i, ni], H_KSG_laplace[i, ni] = kl_ksg(laplace_base)
                print(f"KL error: {np.abs(true_H_laplace - H_KL_laplace[i,ni])}")
                print(f"KSG error: {np.abs(true_H_laplace - H_KSG_laplace[i,ni])}")
                print(f"tKL error: {np.abs(true_H_laplace - H_unif_KL[i,ni])}")
                print(f"tKSG error: {np.abs(true_H_laplace - H_unif_KSG[i,ni])}")

            _ = ent.update_best_model(
                estimator.model, laplace_base, name=model_name, path=model_path
            )

            filename = misc.update_filename(path, filename, i)
            util.io.save(
                (N_range, H_unif_KL, H_unif_KSG, H_KL_laplace, H_KSG_laplace, i),
                os.path.join(path, filename),
            )
