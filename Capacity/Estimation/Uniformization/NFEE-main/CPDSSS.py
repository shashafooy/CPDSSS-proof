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

def calc_entropy(sim_model,n_samples=100):
    net=create_model(sim_model.x_dim, rng=np.random)
    estimator = entropy.UMestimator(sim_model,net)
    estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim/2))
    H,_,_,_ = estimator.calc_ent(reuse_samples=False, method='umtkl')
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


        # sim_S = mvn(rho=0.0, dim_x=int(M*T))
        # sim_V = mvn(rho=0.0, dim_x=int(P*T))

        n_sims = n_samples

        # s = sim_S.sim(n_samples=n_sims).reshape((n_sims,M,T))
        # # s=np.reshape(s,[n_sims,M,T])
        # v = sim_V.sim(n_samples=n_sims).reshape((n_sims,P,T))
        # # v=np.reshape(v,[P,T,n_sims])
        # G = sim_G.sim(n_samples=n_sims).reshape((n_sims,N,M))
        # # G=np.reshape(G,[N,M,n_sims])
        # Q = sim_Q.sim(n_samples=n_sims).reshape((n_sims,N,P))
        # # Q=np.reshape(Q,[N,P,n_sims])

        # #dims = (samples,N,T), matrix multiplication over last 2 dimensions
        # # X=np.matmul(np.transpose(G,(2,0,1)),np.transpose(s,(2,0,1))) + np.matmul(np.transpose(Q,(2,0,1)),np.transpose(v,(2,0,1)))
        # X=np.matmul(G,s)+np.matmul(Q,v)

        # g_term = G[:,:,0]
        # xT_term = X[:,:,T-1]
        # xCond_term = X[:,:,0:T-1].reshape((n_sims,N*(T-1)),order='F') #order 'F' needed to make arrays stack instead of interlaced
        # xT_term = s[:,:,T-1]-.reshape((n_sims,M*(T-1)),order='F')

        if(T>N): #Use previous calculations
            H_gxc = H_joint
            H_cond = H_xxc
        else:
            first_tx_model = CPDSSS(T-1,N,L)

            first_tx_model.set_use_G_flag(g_flag=True)
            H_gxc = calc_entropy(sim_model=first_tx_model,n_samples=n_sims)

            first_tx_model.set_use_G_flag(g_flag=False)
            H_cond = calc_entropy(sim_model=first_tx_model,n_samples=n_sims)
            # H_gxc = UM_KL_Gaussian(np.concatenate((xCond_term,g_term),axis=1))
            # H_cond = UM_KL_Gaussian(xCond_term)

        sim_model.set_use_G_flag(False)
        H_xxc = calc_entropy(sim_model=sim_model,n_samples=n_sims)
        sim_model.set_use_G_flag(True)
        H_joint = calc_entropy(sim_model=sim_model,n_samples=n_sims)
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