from datetime import date, timedelta
import time
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io
import matplotlib.pyplot as plt

from ent_est import entropy
from ml.trainers import ModelCheckpointer
import ml.step_strategies as ss


# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


n_train_samples = 100000
n_trials = 20
val_tol = None
# val_tol = 0.1
patience = 5
N = 15
method = "both"
# batch_size = np.power(2,[5,6,7,8,9,10,11])
batch_size = 256
num_trainings = 3

val_error = np.empty((n_trials, num_trainings)) * np.nan
test_error = np.empty((n_trials, num_trainings)) * np.nan
duration = np.empty((n_trials, num_trainings)) * np.nan
step_sizes = 0.001 * np.logspace(0, -(num_trainings - 1), num_trainings)
KL = np.empty((n_trials)) * np.nan
KSG = np.empty((n_trials)) * np.nan

import re

pattern = r"error:\s*(0\.\d+)"

iter = 0

path = "temp_data/retrain/coarse-fine_step"
today = date.today().strftime("%b_%d")
filename = "retrained_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, iter=iter, rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

sim_laplace = mod.Laplace(mu=0, b=2, N=N)
true_H_laplace = sim_laplace.entropy()

thread = None

for i in range(n_trials):

    n_train = int(n_train_samples * sim_laplace.x_dim)
    laplace_base = sim_laplace.sim(n_samples=n_train)
    base_net = ent.create_MAF_model(sim_laplace.x_dim, rng=np.random)
    checkpointer = ModelCheckpointer(base_net)
    checkpointer.checkpoint()
    # Use same rng seed for each minibatch

    best_err = float("inf")
    estimator = entropy.UMestimator(sim_laplace, base_net)

    for j in range(num_trainings):
        start_time = time.time()
        estimator.learn_transformation(
            n_samples=n_train,
            patience=patience,
            show_progress=True,
            minibatch=batch_size,
            val_tol=val_tol,
            coarse_fine_tune=False,
            step=ss.Adam(a=step_sizes[j]),
        )
        end_time = time.time()
        tot_time = str(timedelta(seconds=int(end_time - start_time)))
        print(f"run {j+1}, learning time: {tot_time}")

        # checkpointer.checkpoint()

        fig = plt.gcf()
        ax = fig.axes[0]
        title = ax.get_title()

        # Get validation error from plot title
        val_error[i, j] = re.search(pattern, title).group(1)
        test_error[i, j] = abs(true_H_laplace - base_net.eval_trnloss(laplace_base))
        duration[i, j] = int(end_time - start_time)

        plt.close()

        if val_error[i, j] < best_err:
            best_err = val_error[i, j]
            checkpointer.checkpoint()
        else:
            checkpointer.restore()

        util.io.save(
            (step_sizes, N, val_error, test_error, duration, KL, KSG), os.path.join(path, filename)
        )

    # get previous knn value from thread
    if i > 0:
        KL[i - 1], KSG[i - 1] = thread.get_result(print_time=True)

    # thread = misc.BackgroundThread(target = ent.knn_entropy,args=(estimator,laplace_base[:10000],1,method))
    # thread.start()

    estimator.samples = laplace_base
    uniform, correction = estimator.uniform_correction()
    thread = estimator.start_knn_thread(uniform, method=method)

    # KL[i-1],KSG[i-1] = ent.knn_entropy(estimator,laplace_base,method=method)

    filename = misc.update_filename(path, filename, i + 1, rename=True)
    util.io.save((step_sizes, N, val_error, duration, KL, KSG), os.path.join(path, filename))

# get thread results from last iteration
KL[-1], KSG[-1] = thread.get_result(print_time=True)
util.io.save(
    (step_sizes, N, val_error, test_error, duration, KL, KSG), os.path.join(path, filename)
)
