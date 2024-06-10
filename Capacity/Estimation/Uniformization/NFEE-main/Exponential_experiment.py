from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
from simulators.CPDSSS_models import Exponential, Exponential_sum
import util.io







knn_samples = 200000
n_train_samples = 10000
n_trials = 100

lambda1=1.5
lambda2=0.5

H_single_exp = np.empty((n_trials,1))*np.nan
H_double_exp = np.empty((n_trials,1))*np.nan
iter=0

path = 'temp_data/exponential'
today=date.today().strftime("%b_%d")
filename = ent.update_filename(path,'',knn_samples,today,iter,rename=False)
util.io.save((H_single_exp,H_double_exp,iter),os.path.join(path,filename))

sim_model_single = Exponential(lamb=lambda1)
sim_model_double = Exponential_sum(lambda1=lambda1,lambda2=lambda2)
true_H_single = sim_model_single.entropy()
true_H_double = sim_model_double.entropy()

for i in range(n_trials):
    exp_base_single = sim_model_single.sim(n_samples=knn_samples)
    exp_base_double = sim_model_double.sim(n_samples=knn_samples)

    ent.print_border("Calculate H(x) single exponential, iter: {}".format(i+1))
    H_single_exp[i] = ent.calc_entropy(sim_model = sim_model_single, n_samples = n_train_samples,base_samples=exp_base_single)
    util.io.save((H_single_exp,H_double_exp,iter),os.path.join(path,filename))
    MSE_single = 1/(i+1) * np.linalg.norm(H_single_exp[:i+1] - true_H_single,2)^2
    print("single exponential entropy MSE: {}".format(MSE_single))

    ent.print_border("Calculate H(x) summed exponential, iter: {}".format(i+1))
    H_double_exp[i] = ent.calc_entropy(sim_model = sim_model_double, n_samples = n_train_samples,base_samples=exp_base_double)
    util.io.save((H_single_exp,H_double_exp,iter),os.path.join(path,filename))
    MSE_double = 1/(i+1) * np.linalg.norm(H_double_exp[:i+1] - true_H_double,2)^2
    np.linalg.norm()
    print("double exponential entropy MSE: {}".format(MSE_single))


    filename = ent.update_filename(path,filename,knn_samples,today,i+1)

