import util.io
import os
import matplotlib.pyplot as plt
import numpy as np


"""
Load and combine all datasets
"""
max_T=0
min_T=0


filepath='temp_data/CPDSSS_data/50k_samples'
for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter = util.io.load(os.path.join(filepath, filename))

    max_T = max_T if max(T_range) <= max_T else max(T_range)
    min_T = min_T if min(T_range) >= min_T else min(T_range)

    if 'MI_tot' not in locals():
        MI_tot = np.empty((0,np.size(T_range)))
        H_gxc_tot = np.empty((0,np.size(T_range)))
        H_xxc_tot = np.empty((0,np.size(T_range)))
        H_joint_tot = np.empty((0,np.size(T_range)))
        H_cond_tot = np.empty((0,np.size(T_range)))


    iter = range(0,completed_iter)
    MI_tot = np.append(MI_tot,MI_cum[iter,:],axis=0)
    H_gxc_tot=np.append(H_gxc_tot,H_gxc_cum[iter,:],axis=0)
    H_xxc_tot=np.append(H_xxc_tot,H_xxc_cum[iter,:],axis=0)
    H_joint_tot=np.append(H_joint_tot,H_joint_cum[iter,:],axis=0)
    H_cond_tot=np.append(H_cond_tot,H_cond_cum[iter,:],axis=0)

MI_mean = np.mean(MI_tot,axis=0)
H_gxc_mean = np.mean(H_gxc_tot,axis=0)
H_xxc_mean = np.mean(H_xxc_tot,axis=0)
H_joint_mean = np.mean(H_joint_tot,axis=0)
H_cond_mean = np.mean(H_cond_tot,axis=0)





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

fig1,ax1=plt.subplots(2,2)
fig1.suptitle('Entropy increase per added transmission')
diff=H_gxc_mean[1:]-H_gxc_mean[0:-1]
yerr = np.std(H_gxc_tot[:,1:],axis=0) + np.std(H_gxc_tot[:,:-1],axis=0)
ax1[0,0].cla(),ax1[0,0].errorbar(T_range[:-1],diff,yerr=yerr)
ax1[0,0].set_title('H1(g,x_cond)'),ax1[0,0].set_ylabel('delta H()'),ax1[0,0].set_xlabel('T')
ax1[1,0].cla(),ax1[1,0].errorbar(T_range[:-1],H_joint_mean[1:]-H_joint_mean[0:-1])
ax1[1,0].set_title('H1(g,x,x_cond)'),ax1[1,0].set_ylabel('delta H()'),ax1[1,0].set_xlabel('T')
ax1[0,1].cla(),ax1[0,1].errorbar(T_range[:-1],H_cond_mean[1:]-H_cond_mean[0:-1])
ax1[0,1].set_title('H1(x_cond)'),ax1[0,1].set_ylabel('delta H()'),ax1[0,1].set_xlabel('T')
ax1[1,1].cla(),ax1[1,1].errorbar(T_range[:-1],H_xxc_mean[1:]-H_xxc_mean[0:-1])
ax1[1,1].set_title('H1(x,x_cond)'),ax1[1,1].set_ylabel('delta H()'),ax1[1,1].set_xlabel('T')

fig.tight_layout()

fig2,ax2=plt.subplots(1,2)
ax2[0].cla(),ax2[0].plot(T_range,MI_mean),ax2[0].set_title('MI increase per T'),ax2[0].set_xlabel('T')
ax2[1].cla(),ax2[1].plot(T_range,np.cumsum(MI_mean)),ax2[1].set_title('total MI'),ax2[1].set_xlabel('T')
fig2.tight_layout()

fig3,ax3=plt.subplots(1,2)
ax3[0].cla(),ax3[0].errorbar(T_range,MI_mean,yerr=np.var(MI_tot,axis=0)),ax3[0].set_title('MI increase per T'),ax3[0].set_xlabel('T')
ax3[1].cla(),ax3[1].errorbar(T_range,np.cumsum(MI_mean),yerr=np.cumsum(np.var(MI_tot,axis=0))),ax3[1].set_title('total MI'),ax3[1].set_xlabel('T')
fig3.tight_layout()

fig4,ax4=plt.subplots(1,2)
T_matrix=np.tile(np.array(T_range),(MI_tot.shape[0],1))
ax4[0].cla(),ax4[0].scatter(T_matrix,MI_tot),ax4[0].set_title('MI increase per T'),ax4[0].set_xlabel('T')
ax4[1].cla(),ax4[1].scatter(T_matrix,np.cumsum(MI_tot,axis=1)),ax4[1].set_title('total MI'),ax4[1].set_xlabel('T')
fig4.tight_layout()

plt.show()
