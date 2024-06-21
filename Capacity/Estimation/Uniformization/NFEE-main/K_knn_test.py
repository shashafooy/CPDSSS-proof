from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl,ksg




# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


knn_samples = 100000
n_train_samples = 30000
n_trials = 100
val_tol = 0.01
# N_range=range(1,11)
N=15
method='both'
k_list=list(range(1,21))

# method='both'

H_unif_KL = np.empty((n_trials,len(k_list)))*np.nan
H_unif_KSG = np.empty((n_trials,len(k_list)))*np.nan
H_KL_laplace = np.empty((n_trials))*np.nan
H_KSG_laplace = np.empty((n_trials))*np.nan
    
iter=0
MSE_uniform=np.inf
MSE_KL=np.inf

path = 'temp_data/k_knn_test/1M_knn'
today=date.today().strftime("%b_%d")
filename = "knn_data({})".format(today)
filename = ent.update_filename(path=path,old_name=filename,iter=iter,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    sim_laplace = mod.Laplace(mu=0,b=2,N=N)
    true_H_laplace = sim_laplace.entropy()        
    laplace_base = sim_laplace.sim(n_samples=knn_samples)
    H_KL_laplace[i] = kl(laplace_base)
    H_KSG_laplace[i] = ksg(laplace_base)

    
    

    ent.print_border("Calculate H(x) laplace, iter: {}".format(i+1))            
    H_unif_KL[i,:],H_unif_KSG[i,:] = ent.calc_entropy(
        sim_model = sim_laplace, 
        n_samples = n_train_samples,
        base_samples=laplace_base,
        val_tol=val_tol,
        method=method,
        k=k_list)                    

    

    
    filename=ent.update_filename(path,filename,i+1,rename=True)
    util.io.save((k_list,H_unif_KL,H_unif_KSG,H_KL_laplace,H_KSG_laplace),os.path.join(path,filename))
    

