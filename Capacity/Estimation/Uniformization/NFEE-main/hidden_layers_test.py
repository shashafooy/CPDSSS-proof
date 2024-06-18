from datetime import date
import os
import numpy as np
import misc_CPDSSS.entropy_util as ent
import simulators.CPDSSS_models as mod
import util.io

from ent_est.entropy import kl,ksg







knn_samples = 200000
n_train_samples = 10000
n_trials = 100
# N_range=range(1,11)
N=8
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

path = 'temp_data/laplace_test/hidden_layers'
today=date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = ent.update_filename(path=path,old_name=filename,iter=iter,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))


for i in range(n_trials):
    sim_laplace = mod.Laplace(mu=0,b=2,N=N)
    true_H_laplace = sim_laplace.entropy()        
    laplace_base = sim_laplace.sim(n_samples=knn_samples)
    H_KL_laplace[i] = kl(laplace_base)
    H_KSG_laplace[i] = ksg(laplace_base)
    for ni,n_layers in enumerate(layers):    
        for nj,n_nodes in enumerate(nodes):              
            hidden_layers = [n_nodes]*n_layers           

            ent.print_border("Calculate H(x) laplace, Nodes={}, Layers={}, iter: {}".format(n_nodes,n_layers,i+1))            
            H_unif_KL[i,ni,nj],H_unif_KSG[i,ni] = ent.calc_entropy(sim_model = sim_laplace, n_samples = n_train_samples,base_samples=laplace_base,val_tol=0.05,method=method,n_hiddens=hidden_layers)                    

            util.io.save((layers,nodes,H_unif_KL,H_unif_KSG,H_KL_laplace,H_KSG_laplace),os.path.join(path,filename))

    
    filename=ent.update_filename(path,filename,i+1,rename=True)
    util.io.save((layers,nodes,H_unif_KL,H_unif_KSG,H_KL_laplace,H_KSG_laplace),os.path.join(path,filename))

