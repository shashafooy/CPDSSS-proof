from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl_ksg




# hh = ent.time_exec(lambda: kl(mod.Laplace(mu=0,b=2,N=10).sim(n_samples=100000),k=5))


knn_samples = 1000000
n_train_samples = 30000
n_trials = 100
val_tol = 0.01
# N_range=range(1,11)
N=15
method='both'
layers = [2,3,4]
nodes = [100,150,200]

# method='both'

H_unif_KL = np.empty((n_trials,len(layers),len(nodes)))*np.nan
H_unif_KSG = np.empty((n_trials,len(layers),len(nodes)))*np.nan
H_KL_laplace = np.empty((n_trials))*np.nan
H_KSG_laplace = np.empty((n_trials))*np.nan
    
iter=0
MSE_uniform=np.inf
MSE_KL=np.inf

path = 'temp_data/laplace_test/hidden_layers/15N_1M_knn'
today=date.today().strftime("%b_%d")
filename = "hidden_layer_data({})".format(today)
filename = ent.update_filename(path=path,old_name=filename,iter=iter,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    sim_laplace = mod.Laplace(mu=0,b=2,N=N)
    true_H_laplace = sim_laplace.entropy()        
    laplace_base = sim_laplace.sim(n_samples=knn_samples)
    # ent.print_border("Starting iteration {i} KL KSG in background")

    # KNN might take awhile, run in background
    thread = ent.BackgroundThread(target = kl_ksg,args=(laplace_base,))
    thread.start()

    # knn_thread = threading.Thread(target=kl_ksg,name="KL KSG",args=laplace_base)

    # H_KL_laplace[i],H_KSG_laplace[i] = kl_ksg(laplace_base)
    for ni,n_layers in enumerate(layers):    
        for nj,n_nodes in enumerate(nodes):              
            hidden_layers = [n_nodes]*n_layers           
            ent.print_border("Calculate H(x) laplace, Nodes={}, Layers={}, iter: {}".format(n_nodes,n_layers,i+1))            
            H_unif_KL[i,ni,nj],H_unif_KSG[i,ni] = ent.calc_entropy(
                sim_model = sim_laplace, 
                n_samples = n_train_samples,
                base_samples=laplace_base,
                val_tol=val_tol,
                method=method,
                n_hiddens=hidden_layers)                    

            util.io.save((layers,nodes,H_unif_KL,H_unif_KSG,H_KL_laplace,H_KSG_laplace),os.path.join(path,filename))

    
    H_KL_laplace[i],H_KSG_laplace[i] = thread.get_result() #get results after waiting for thread

    filename=ent.update_filename(path,filename,i+1,rename=True)
    util.io.save((layers,nodes,H_unif_KL,H_unif_KSG,H_KL_laplace,H_KSG_laplace),os.path.join(path,filename))

