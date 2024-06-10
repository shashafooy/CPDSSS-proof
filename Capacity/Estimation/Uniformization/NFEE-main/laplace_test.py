from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl







knn_samples = 200000
n_train_samples = 10000
n_trials = 100
N_range=range(1,10)


H_laplace = np.empty((n_trials,len(N_range)))
H_KL_laplace = np.empty((n_trials,len(N_range)))

iter=0
MSE_uniform=np.inf
MSE_KL=np.inf

path = 'temp_data/laplace_test'
today=date.today().strftime("%b_%d")
filename = ent.update_filename(path,'',knn_samples,today,iter,rename=False)
util.io.save((H_laplace,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    sim_laplace = mod.Laplace(0,2,N=N)
    true_H_laplace = sim_laplace.entropy()

    for N in N_range:    
        laplace_base = sim_laplace.sim(n_samples=knn_samples)
        

        ent.print_border("Calculate H(x) laplace, N={}, iter: {}".format(N,i+1))
        H_laplace[i,N-1] = ent.calc_entropy(sim_model = sim_laplace, n_samples = n_train_samples,base_samples=laplace_base)    
        H_KL_laplace[i,N-1] = kl(laplace_base)
        MSE_uniform = 1/(i+1) * np.linalg.norm(H_laplace[:i+1,N-1] - true_H_laplace,2)**2
        MSE_KL = 1/(i+1) * np.linalg.norm(H_KL_laplace[:i+1,N-1] - true_H_laplace,2)**2
        print("laplace entropy MSE: {}\nlaplace KL entropy MSE: {}".format(MSE_uniform,MSE_KL))

        if N==N_range[-1]:
            iter=iter+1
            ent.update_filename(path,filename,knn_samples,today,iter,rename=True)
        util.io.save((H_laplace,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

