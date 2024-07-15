from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl_ksg




# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


knn_samples = 1000000
n_train_samples = 100000
n_trials = 100
val_tol = None
# val_tol = 0.5
patience=5
N=15
method='both'
# layers = [2,3,4]
stages = np.arange(12,19,2) #theano gradient breaks for stages>=20


H_unif_KL = np.empty((n_trials,len(stages)))*np.nan
H_unif_KSG = np.empty((n_trials,len(stages)))*np.nan
H_KL = np.empty((n_trials))*np.nan
H_KSG = np.empty((n_trials))*np.nan
    
iter=0
MSE_uniform=np.inf
MSE_KL=np.inf

path = 'temp_data/stages_test/15N_1M_knn'
today=date.today().strftime("%b_%d")
filename = "num_stages_data({})".format(today)
filename = misc.update_filename(path=path,old_name=filename,iter=iter,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    sim_laplace = mod.Laplace(mu=0,b=2,N=N)
    true_H_laplace = sim_laplace.entropy()        
    laplace_base = sim_laplace.sim(n_samples=knn_samples)

    # KNN might take awhile, run in background
    thread = misc.BackgroundThread(target = kl_ksg,args=(laplace_base,))
    thread.start()
   

    for ns,n_stages in enumerate(stages):              
        
        misc.print_border("Calculate H(x) laplace, stages={} iter: {}".format(n_stages,i+1))    
        estimator = ent.learn_model(sim_laplace,n_train_samples,val_tol,patience,n_stages=n_stages)

        H_unif_KL[2*i,ns],H_unif_KSG[2*i,ns] = ent.knn_entropy(estimator,laplace_base,method=method)                    
        H_unif_KL[2*i+1,ns],H_unif_KSG[2*i+1,ns] = ent.knn_entropy(estimator,sim_laplace.sim(knn_samples),method=method)                    

        #if thread finishes early, get results, but only once
        if ~thread.used_result() and not thread.is_alive():
            H_KL[i],H_KSG[i] = thread.get_result()
        util.io.save((stages,H_unif_KL,H_unif_KSG,H_KL,H_KSG),os.path.join(path,filename))

    #Wait if thread is still not finished
    H_KL[i],H_KSG[i] = thread.get_result() 

    filename=misc.update_filename(path,filename,i+1,rename=True)
    util.io.save((stages,H_unif_KL,H_unif_KSG,H_KL,H_KSG),os.path.join(path,filename))

