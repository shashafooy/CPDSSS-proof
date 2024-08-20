from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl, ksg


knn_samples = 200000
n_train_samples = 100000
n_trials = 100
N_range = range(11, 21)
method = "both"
patience = 5
val_tol = 0.1
# method='both'

H_unif_KL = np.empty((n_trials, len(N_range))) * np.nan
H_unif_KSG = np.empty((n_trials, len(N_range))) * np.nan
H_KL_laplace = np.empty((n_trials, len(N_range))) * np.nan
H_KSG_laplace = np.empty((n_trials, len(N_range))) * np.nan

iter = 0
MSE_uniform = np.inf
MSE_KL = np.inf

path = "temp_data/laplace_test"
today = date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, iter=iter, rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    for ni, N in enumerate(N_range):
        sim_laplace = mod.Laplace(mu=0, b=2, N=N)
        true_H_laplace = sim_laplace.entropy()
        laplace_base = sim_laplace.sim(n_samples=knn_samples)

        misc.print_border("Calculate H(x) laplace, N={}, iter: {}".format(N, i + 1))
        if method == "umtksg":
            H_unif_KSG[i, ni] = ent.calc_entropy(
                sim_model=sim_laplace,
                n_samples=n_train_samples,
                base_samples=laplace_base,
                val_tol=val_tol,
                patience=patience,
                method=method,
            )
            H_KSG_laplace[i, ni] = ksg(laplace_base)
        elif method == "umtkl":
            H_unif_KL[i, ni] = ent.calc_entropy(
                sim_model=sim_laplace,
                n_samples=n_train_samples,
                base_samples=laplace_base,
                val_tol=0.01,
                patience=3,
                method=method,
            )
            H_KL_laplace[i, ni] = kl(laplace_base)
        else:
            H_unif_KL[i, ni], H_unif_KSG[i, ni] = ent.calc_entropy(
                sim_model=sim_laplace,
                n_samples=n_train_samples,
                base_samples=laplace_base,
                val_tol=0.01,
                method=method,
            )
            H_KL_laplace[i, ni] = kl(laplace_base)
            H_KSG_laplace[i, ni] = ksg(laplace_base)

        # H_unif_KL[i,ni],H_unif_KSG[i,ni] = ent.calc_entropy(sim_model = sim_laplace, n_samples = n_train_samples,base_samples=laplace_base,val_tol=0.01,method='both')

        # MSE_uniform = np.mean((H_unif_KL[:i+1,N-1] - true_H_laplace)**2)
        # MSE_KL = np.mean((H_KL_laplace[:i+1,N-1] - true_H_laplace)**2)
        # print("laplace entropy MSE: {}\nlaplace KL entropy MSE: {}".format(MSE_uniform,MSE_KL))

        if N == N_range[-1]:
            iter = iter + 1
            filename = misc.update_filename(path, filename, iter, rename=True)
        util.io.save(
            (N_range, H_unif_KL, H_unif_KSG, H_KL_laplace, H_KSG_laplace, iter),
            os.path.join(path, filename),
        )
