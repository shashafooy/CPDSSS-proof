import numpy as np
from scipy import stats

import util.io
import os

from ent_est import entropy
from simulators.complex import mvn
from simulators.CPDSSS_models import CPDSSS


def UM_KL_Gaussian(x):
    std_x=np.std(x,axis=0)
    z=stats.norm.cdf(x)
    return tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x),axis=1)))

def create_model(n_inputs, rng):
    n_hiddens=[50,50]
    act_fun='tanh'
    n_mades=10

    import ml.models.mafs as mafs

    return mafs.MaskedAutoregressiveFlow(
                n_inputs=n_inputs,
                n_hiddens=n_hiddens,
                act_fun=act_fun,
                n_mades=n_mades,
                mode='random',
                rng=rng
            )

def calc_entropy(sim_model,base_samples=None,n_samples=100):
    net=create_model(sim_model.x_dim, rng=np.random)
    estimator = entropy.UMestimator(sim_model,net)
    estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim/2))
    estimator.samples = estimator.samples if base_samples is None else base_samples
    H,_,_,_ = estimator.calc_ent(reuse_samples=True, method='umtkl')
    return H


N=2
L=2
M=int(N/L)
P=N-int(N/L)



T_range = range(N,10)
n_trials = 100
n_samples = 1000



MI_tKL = np.empty(len(T_range))
MI_means = np.empty(len(T_range))


MI_cum = np.empty((n_trials,len(T_range)))
H_gxc_cum=np.empty((n_trials,len(T_range)))
H_xxc_cum=np.empty((n_trials,len(T_range)))
H_joint_cum=np.empty((n_trials,len(T_range)))
H_cond_cum=np.empty((n_trials,len(T_range)))

        


sim_G = mvn(rho=0.0, dim_x=N*M)
sim_Q = mvn(rho=0.0, dim_x=N*P)

for i in range(n_trials):        

    # MI_cum = np.empty(n_trials)
    # sim_mdl = mvn(rho=0.0, dim_x=d)
    
    # sim_S = mvn(rho=0.0, dim_x=int(M*T))
    # sim_V = mvn(rho=0.0, dim_x=int(P*T))
    
    
    
    # cal2 = np.empty(n_trials)
    # cal3 = np.empty(n_trials)
    # cal4 = np.empty(n_trials)


            
    for k, T in enumerate(T_range):
        sim_model = CPDSSS(T,N,L)

        n_sims = n_samples

        X,X_T,X_cond,G = sim_model.get_base_X_G(n_sims)
        gxc=np.concatenate((X_cond,G),axis=1)
        joint=np.concatenate((X,G),axis=1)



        if(T>N): #Use previous calculations
            H_gxc = H_joint
            H_cond = H_xxc
        else:
            first_tx_model = CPDSSS(T-1,N,L)

            first_tx_model.set_use_G_flag(g_flag=True)
            print("-"*25 + "\ncalculating H_gxc, T: {0}, iter: {1}".format(T,i+1))
            print("-"*25)
            H_gxc = calc_entropy(sim_model=first_tx_model,n_samples=n_sims,base_samples=gxc)

            first_tx_model.set_use_G_flag(g_flag=False)
            print("-"*25 + "\ncalculating H_cond, T: {0}, iter: {1}".format(T,i+1))
            print("-"*25)
            H_cond = calc_entropy(sim_model=first_tx_model,n_samples=n_sims,base_samples=X_cond)
            # H_gxc = UM_KL_Gaussian(np.concatenate((xCond_term,g_term),axis=1))
            # H_cond = UM_KL_Gaussian(xCond_term)

        sim_model.set_use_G_flag(False)
        print("-"*25 + "\ncalculating H_xxc, T: {0}, iter: {1}".format(T,i+1))
        print("-"*25)
        H_xxc = calc_entropy(sim_model=sim_model,n_samples=n_sims,base_samples=X)
        sim_model.set_use_G_flag(True)
        print("-"*25 + "\ncalculating H_joint, T: {0}, iter: {1}".format(T,i+1))
        print("-"*25)
        H_joint = calc_entropy(sim_model=sim_model,n_samples=n_sims,base_samples=joint)
        # H_xxc = UM_KL_Gaussian(np.concatenate((xCond_term,xT_term),axis=1))
        # H_joint = UM_KL_Gaussian(np.concatenate((xCond_term,xT_term,g_term),axis=1))

        H_gxc_cum[i,k]=H_gxc
        H_xxc_cum[i,k]=H_xxc
        H_joint_cum[i,k]=H_joint
        H_cond_cum[i,k]=H_cond
        MI_cum[i,k] = H_gxc + H_xxc - H_joint - H_cond
        util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum), os.path.join('temp_data', 'CPDSSS_data_dump')) 
        # z=stats.norm.cdf(xT_term)
        # H_xxc = tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(xT_term),axis=1)))

        
        # mse1[n,k] = np.mean((cal1-true_val)**2)
        # mse2[n,k] = np.mean((cal2-true_val)**2)
        # mse3[n,k] = np.mean((cal3-true_val)**2)
        # mse4[n,k] = np.mean((cal4-true_val)**2)

MI_tKL=np.mean(MI_cum,axis=0)
MI_means=np.mean(H_gxc_cum,axis=0) + np.mean(H_xxc_cum,axis=0) - np.mean(H_joint_cum,axis=0) - np.mean(H_cond_cum,axis=0)
MI_cumsum = np.cumsum(MI_means)

    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
            
fig, ax = plt.subplots(1,1)
# ax.set_yscale("log")
            
# ax.plot([N*t for t in T_range], H_KL, marker='o', color='b', linestyle=':', label='H KL', mfc='none')
# ax.plot(T_range, np.sqrt(mse4[0]), marker='o', color='b', linestyle='-', label='UM-tKSG', mfc='none')   
dims=[N*t for t in T_range] 
# ax[0].plot(dims, MI_tKL, marker='x', color='r', linestyle=':', label='MI individual')
ax.plot(dims, MI_means, marker='x', color='b', linestyle='-', label='MI means')
# ax.plot(T_range, np.sqrt(mse2[0]), marker='x', color='r', linestyle='-', label='KSG')
        
ax.set_xlabel('dimension')
ax.set_ylabel('Mutual Information')
ax.set_title("Individual conditional MI")
plt.savefig('figs/MI_cond_CPDSSSS')

plt.cla()
fig, ax = plt.subplots(1,1)
ax.plot(dims,MI_cumsum,marker='x',linestyle='-')
ax.set_xlabel('dimension')
ax.set_ylabel('Mutual Information')
ax.set_title('Cumulative MI')


plt.savefig('figs/MI_cumulative_CPDSSSS')
        
import util.io
import os
util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum), os.path.join('temp_data', 'gaussian_experiment')) 