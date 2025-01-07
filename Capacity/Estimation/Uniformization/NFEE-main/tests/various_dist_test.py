from datetime import date
import os
from _utils import set_sys_path

set_sys_path()

import numpy as np
from misc_CPDSSS.entropy_util import MAF as ent
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl


knn_samples = 200000
n_train_samples = 10000
n_trials = 10


H_laplace = np.empty((n_trials, 1))
H_cauchy = np.empty((n_trials, 1))
H_logistic = np.empty((n_trials, 1))
H_KL_laplace = np.empty((n_trials, 1))
H_KL_cauchy = np.empty((n_trials, 1))
H_KL_logistic = np.empty((n_trials, 1))

iter = 0

path = "temp_data/dist_test"
today = date.today().strftime("%b_%d")
filename = ent.update_filename(path, "", knn_samples, today, iter, rename=False)
# util.io.save((H_single_exp,H_double_exp,iter),os.path.join(path,filename))

sim_laplace = mod.Laplace(0, 2, N=2)
sim_cauchy = mod.Cauchy(0, 0.5, N=2)
sim_logistic = mod.Logistic(0, 2, N=2)

true_H_laplace = sim_laplace.entropy()
true_H_cauchy = sim_cauchy.entropy()
true_H_logistic = sim_logistic.entropy()

for i in range(n_trials):
    laplace_base = sim_laplace.sim(n_samples=knn_samples)
    cauchy_base = sim_cauchy.sim(n_samples=knn_samples)
    logistic_base = sim_logistic.sim(n_samples=knn_samples)

    ent.print_border("Calculate H(x) laplace, iter: {}".format(i + 1))
    H_laplace[i] = ent.calc_entropy(
        sim_model=sim_laplace, n_samples=n_train_samples, base_samples=laplace_base
    )
    H_KL_laplace[i] = kl(laplace_base)
    MSE_uniform = 1 / (i + 1) * np.linalg.norm(H_laplace[: i + 1] - true_H_laplace, 2) ** 2
    MSE_KL = 1 / (i + 1) * np.linalg.norm(H_KL_laplace[: i + 1] - true_H_laplace, 2) ** 2
    print("laplace entropy MSE: {}\nlaplace KL entropy MSE: {}".format(MSE_uniform, MSE_KL))

    ent.print_border("Calculate H(x) cauchy, iter: {}".format(i + 1))
    H_cauchy[i] = ent.calc_entropy(
        sim_model=sim_cauchy, n_samples=n_train_samples, base_samples=cauchy_base
    )
    H_KL_cauchy[i] = kl(cauchy_base)
    MSE_uniform = 1 / (i + 1) * np.linalg.norm(H_cauchy[: i + 1] - true_H_cauchy, 2) ** 2
    MSE_KL = 1 / (i + 1) * np.linalg.norm(H_KL_cauchy[: i + 1] - true_H_cauchy, 2) ** 2
    print("cauchy entropy MSE: {}\ncauchy KL entropy MSE: {}".format(MSE_uniform, MSE_KL))

    ent.print_border("Calculate H(x) logistic, iter: {}".format(i + 1))
    H_logistic[i] = ent.calc_entropy(
        sim_model=sim_logistic, n_samples=n_train_samples, base_samples=logistic_base
    )
    H_KL_logistic[i] = kl(logistic_base)
    MSE_uniform = 1 / (i + 1) * np.linalg.norm(H_logistic[: i + 1] - true_H_logistic, 2) ** 2
    MSE_KL = 1 / (i + 1) * np.linalg.norm(H_KL_logistic[: i + 1] - true_H_logistic, 2) ** 2
    print("logistic entropy MSE: {}\nlogistic KL entropy MSE: {}".format(MSE_uniform, MSE_KL))
