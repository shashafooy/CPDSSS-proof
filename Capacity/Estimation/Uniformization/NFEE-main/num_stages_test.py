from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl_ksg
import ml.step_strategies as ss


# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


knn_samples = 1000000
n_train_samples = 100000
n_trials = 100
val_tol = None
# val_tol = 0.5
patience = 5
N = 15
method = "both"
# layers = [2,3,4]
stages = np.arange(12, 19, 2)  # theano gradient breaks for stages>=20
stages = [6, 7, 8, 10, 11]
stages = [1, 2, 3, 4]
minibatch = 128
fine_tune = False
hidden = [100, 100]
step = ss.Adam(a=0.0001)


H_unif_KL = np.empty((n_trials, len(stages))) * np.nan
H_unif_KSG = np.empty((n_trials, len(stages))) * np.nan
H_KL = np.empty((n_trials)) * np.nan
H_KSG = np.empty((n_trials)) * np.nan

MSE_uniform = np.inf
MSE_KL = np.inf

path = "temp_data/stages_test/15N_1M_knn"
today = date.today().strftime("%b_%d")
filename = "num_stages_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, iter=-1, rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

H_thread1 = None
H_thread2 = None

for i in range(n_trials):
    sim_laplace = mod.Laplace(mu=0, b=2, N=N)
    true_H_laplace = sim_laplace.entropy()
    laplace_base = sim_laplace.sim(n_samples=knn_samples)

    # KNN might take awhile, run in background
    # thread = misc.BackgroundThread(target = kl_ksg,args=(laplace_base,))
    # thread.start()

    for ns, n_stages in enumerate(stages):

        if H_thread1 is not None:
            H_unif_KL[old_idx1], H_unif_KSG[old_idx1] = H_thread1.get_result() + H_correction1
            # H_unif_KL[old_idx2],H_unif_KSG[old_idx2] = H_thread2.get_result() + H_correction2

        misc.print_border("Calculate H(x) laplace, stages={} iter: {}".format(n_stages, i + 1))
        estimator = ent.learn_MAF_model(
            sim_laplace,
            n_samples=n_train_samples,
            n_hiddens=hidden,
            n_stages=n_stages,
            mini_batch=minibatch,
            step=step,
        )
        print(f"final test loss {estimator.model.eval_trnloss(laplace_base)}")
        H_unif_KL[i, ns], H_unif_KSG[i, ns] = ent.knn_entropy(
            estimator, laplace_base, method=method
        )
        print(f"KL entropy {H_unif_KL[i,ns]}\nKSG entropy {H_unif_KSG[i,ns]}")
        # uniform, H_correction1 = estimator.uniform_correction(laplace_base)
        # H_thread1 = estimator.start_knn_thread(uniform, method=method)

        # uniform,H_correction2 = estimator.uniform_correction(sim_laplace.sim(knn_samples))
        # H_thread2 = estimator.start_knn_thread(uniform,method=method)
        #
        old_idx1 = (i, ns)
        # old_idx1=(2*i,ns)
        # old_idx2=(2*i+1,ns)

        util.io.save((stages, H_unif_KL, H_unif_KSG, H_KL, H_KSG), os.path.join(path, filename))

    # Wait if thread is still not finished

    filename = misc.update_filename(path, filename, i + 1, rename=True)
    util.io.save((stages, H_unif_KL, H_unif_KSG, H_KL, H_KSG), os.path.join(path, filename))
