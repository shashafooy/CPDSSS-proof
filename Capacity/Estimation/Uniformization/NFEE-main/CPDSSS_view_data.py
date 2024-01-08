import util.io
import os
import matplotlib.pyplot as plt
import numpy as np


T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter = util.io.load(os.path.join('temp_data', 'CPDSSS_data_dump'))

valid_iter = range(0,completed_iter)
MI_cum = MI_cum[valid_iter,:]

MI_mean = np.mean(MI_cum[0:completed_iter,:],axis=0)
H_gxc_mean = np.mean(H_gxc_cum[0:completed_iter,:],axis=0)
H_xxc_mean = np.mean(H_xxc_cum[0:completed_iter,:],axis=0)
H_joint_mean = np.mean(H_joint_cum[0:completed_iter,:],axis=0)
H_cond_mean = np.mean(H_cond_cum[0:completed_iter,:],axis=0)





fig,ax=plt.subplots(2,2)
fig.suptitle('Entropy increase per added transmission')

ax[0,0].cla(),ax[0,0].plot(H_gxc_mean[1:]-H_gxc_mean[0:-1])
ax[0,0].set_title('H(g,x_cond)'),ax[0,0].set_ylabel('delta H()'),ax[0,0].set_xlabel('T')
ax[1,0].cla(),ax[1,0].plot(H_joint_mean[1:]-H_joint_mean[0:-1])
ax[1,0].set_title('H(g,x,x_cond)'),ax[1,0].set_ylabel('delta H()'),ax[1,0].set_xlabel('T')
ax[0,1].cla(),ax[0,1].plot(H_cond_mean[1:]-H_cond_mean[0:-1])
ax[0,1].set_title('H(x_cond)'),ax[0,1].set_ylabel('delta H()'),ax[0,1].set_xlabel('T')
ax[1,1].cla(),ax[1,1].plot(H_xxc_mean[1:]-H_xxc_mean[0:-1])
ax[1,1].set_title('H(x,x_cond)'),ax[1,1].set_ylabel('delta H()'),ax[1,1].set_xlabel('T')

fig.tight_layout()


fig2,ax2=plt.subplots(1,2)
ax2[0].cla(),ax2[0].plot(T_range,MI_mean),ax2[0].set_title('MI increase per T'),ax2[0].set_xlabel('T')
ax2[1].cla(),ax2[1].plot(T_range,np.cumsum(MI_mean)),ax2[1].set_title('total MI'),ax2[1].set_xlabel('T')
fig2.tight_layout()








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


