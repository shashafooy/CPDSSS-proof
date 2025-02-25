from datetime import date, timedelta
import time
import os

from _utils import set_sys_path

set_sys_path()

import numpy as np
from misc_CPDSSS.entropy_util import MAF as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io
import matplotlib.pyplot as plt


from ent_est import entropy
from ml.trainers import ModelCheckpointer


# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


knn_samples = 100000
n_train_samples = 100000
n_trials = 20
# val_tol = 0.001
# val_tol = 0.5

show_progress = False
patience = 5
N = 20
batch_size = np.power(2, [10, 11])


error = np.empty((n_trials, len(batch_size))) * np.nan
duration = np.empty((n_trials, len(batch_size))) * np.nan
H_reuse = np.empty((n_trials, len(batch_size))) * np.nan
H_sim = np.empty((n_trials, len(batch_size))) * np.nan


path = f"temp_data/batch_size/{N}N_{misc.int_to_short_string(n_train_samples)}_train"
today = date.today().strftime("%b_%d")
filename = "batch_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

sim_laplace = mod.Laplace(mu=0, b=2, N=N)
true_H_laplace = sim_laplace.entropy()

thread = None
thread_2 = None

for i in range(n_trials):

    n_train = int(n_train_samples * sim_laplace.x_dim)
    laplace_base = sim_laplace.sim(n_samples=n_train)
    base_net = ent.create_model(sim_laplace.x_dim, rng=np.random)
    checkpointer = ModelCheckpointer(base_net)
    checkpointer.checkpoint()
    # Use same rng seed for each minibatch
    seed = np.random.randint(1, 1000)

    for mi, mini_batch in enumerate(batch_size):
        misc.print_border(f"Training using minibatch {mini_batch}, iter {i}")
        # net=cp(base_net)
        checkpointer.restore()
        estimator = entropy.UMestimator(sim_laplace, base_net)
        estimator.samples = laplace_base
        start_time = time.time()
        estimator.learn_transformation(
            n_samples=n_train,
            patience=patience,
            show_progress=show_progress,
            minibatch=int(mini_batch),
            rng=np.random.default_rng(seed=seed),
        )
        end_time = time.time()
        tot_time = str(timedelta(seconds=int(end_time - start_time)))
        print("learning time: ", tot_time)

        if thread is not None:
            H_reuse[old_idx] = thread.get_result() + correction

        uniform, correction = estimator.uniform_correction(laplace_base)
        thread = estimator.start_knn_thread(uniform)

        # if thread_2 is not None:
        #     H_sim[old_idx] = thread_2.get_result() + correction_2

        # uniform,correction_2 = estimator.uniform_correction(sim_laplace.sim(n_train))
        # thread_2 = estimator.start_knn_thread(uniform)

        old_idx = (i, mi)

        error[i, mi] = np.abs(estimator.model.eval_trnloss(laplace_base) - true_H_laplace)
        duration[i, mi] = int(end_time - start_time)

        if show_progress:
            fig = plt.gcf()
            ax = fig.axes[0]
            title = ax.get_title()
            ax.set_title(f"{title}\n time: {tot_time}")
            plt.savefig(f"figs/MAF_batch/batch_{mini_batch}.png")
            plt.close()

        filename = misc.update_filename(path, filename, i, rename=True)
        util.io.save((batch_size, N, error, duration, H_sim, H_reuse), os.path.join(path, filename))
