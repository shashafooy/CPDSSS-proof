import util.io
import os
import matplotlib.pyplot as plt
import numpy as np


def append_data(old_data, old_T_range, new_data, new_T_range):
    """
    Adjust both datasets to match their T_range before appending
    """
    old_width=old_data.shape[1]
    new_width=new_data.shape[1]

    min_diff = min(old_T_range) - min(new_T_range)
    if(min_diff < 0): #Pad new data
        # new_data = np.insert(new_data,range(0,abs(min_diff)),np.empty((1,abs(min_diff))),axis=1)
        # new_data = np.insert(new_data,range(0,abs(min_diff)),np.nan,axis=1)
        new_data = np.insert(new_data,[0]*abs(min_diff),np.nan,axis=1)
    elif(min_diff > 0): #Pad old data
        # old_data = np.insert(old_data,0:min_diff,np.empty((1,min_diff)),axis=1)
        #insert nan to first min_diff columns
        old_data = np.insert(old_data,[0]*min_diff,np.nan,axis=1)

    max_diff = max(old_T_range) - max(new_T_range)
    if(max_diff > 0): #Pad new data
        new_data = np.insert(new_data,[new_data.shape[1]]*max_diff,np.nan,axis=1)
        # new_data = np.append(new_data,np.empty((new_data.shape[0],max_diff)),axis=1)
    elif(max_diff < 0): #Pad old data
        old_data = np.insert(old_data,[old_data.shape[1]]*abs(max_diff),np.nan,axis=1)
        # old_data = np.append(old_data,np.empty((old_data.shape[0],abs(max_diff))),axis=1)
    

    return np.append(old_data,new_data,axis=0), range(min(old_T_range[0],new_T_range[0]),max(old_T_range[-1],new_T_range[-1])+1)


"""
Load and combine all datasets
"""
max_T=0
min_T=0


#util.io.save((T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter), os.path.join(filepath,filename)) 
base_path = 'temp_data/CPDSSS_data/'
filepaths = [base_path+'50k_high_epoch', base_path + '50k_samples']
filepath = base_path+'50k_tol_0.1_patience_10'
# filepath=filepaths[1]
idx=0
# for idx,filepath in enumerate(filepaths):
for filename in os.listdir(filepath):
    filename=os.path.splitext(filename)[0] #remove extention
    T_range, MI_cum,H_gxc_cum,H_xxc_cum,H_joint_cum,H_cond_cum,completed_iter = util.io.load(os.path.join(filepath, filename))

    if 'MI_tot' not in locals():
        MI_tot = np.empty((0,np.size(T_range)))
        H_gxc_tot = np.empty((0,np.size(T_range)))
        H_xxc_tot = np.empty((0,np.size(T_range)))
        H_joint_tot = np.empty((0,np.size(T_range)))
        H_cond_tot = np.empty((0,np.size(T_range)))
        old_range = T_range

    # append_data(MI_tot,T_range,H_gxc_tot,range(2,8))
    # max_T = max_T if max(T_range) <= max_T else max(T_range)
    # min_T = min_T if min(T_range) >= min_T else min(T_range)


    iter = range(0,completed_iter)

    '''Experiment to only grab T=2,3 from the 50k_samples'''
    if idx == 1:
        if T_range[0] >3: #does not contain values for T=2,3
            continue
        if T_range[0] == 3 : #only has T=3 
            T_range = range(3,4)
        else:
            T_range = range(2,4)


        # T_range=range(2,4)
        MI_cum=MI_cum[:,0:len(T_range)]
        H_gxc_cum=H_gxc_cum[:,0:len(T_range)]
        H_xxc_cum=H_xxc_cum[:,0:len(T_range)]
        H_joint_cum=H_joint_cum[:,0:len(T_range)]
        H_cond_cum=H_cond_cum[:,0:len(T_range)]
    

    MI_tot,_ = append_data(MI_tot,old_range,MI_cum[iter,:],T_range)
    H_gxc_tot,_=append_data(H_gxc_tot,old_range,H_gxc_cum[iter,:],T_range)
    H_xxc_tot,_=append_data(H_xxc_tot,old_range,H_xxc_cum[iter,:],T_range)
    H_joint_tot,_=append_data(H_joint_tot,old_range,H_joint_cum[iter,:],T_range)
    H_cond_tot,old_range=append_data(H_cond_tot,old_range,H_cond_cum[iter,:],T_range)

MI_mean = np.nanmean(MI_tot,axis=0)
H_gxc_mean = np.nanmean(H_gxc_tot,axis=0)
H_xxc_mean = np.nanmean(H_xxc_tot,axis=0)
H_joint_mean = np.nanmean(H_joint_tot,axis=0)
H_cond_mean = np.nanmean(H_cond_tot,axis=0)
T_range = old_range


'''
Experiment combining data but with offset of 1 (like reusing old data). 
The target entropy should remain the same and this gives better accuracy
'''
H_gxc_mean = np.nanmean(np.append(H_gxc_tot[:,:-1],H_joint_tot[:,1:],axis=1),axis=0)
temp = np.empty(H_gxc_tot.shape)*np.nan
temp=np.insert(H_joint_tot[:,:-1],0,np.nan,axis=1) #insert column of nan to align matrices
H_gxc_mean = np.nanmean(np.append(H_gxc_tot,temp,axis=0),axis=0)

# temp = temp = np.empty(H_joint_tot.shape)*np.nan
temp=np.insert(H_gxc_tot[:,1:],H_joint_tot.shape[1]-1,np.nan,axis=1)
H_joint_mean = np.nanmean(np.append(temp,H_joint_tot,axis=0),axis=0)

temp=np.insert(H_cond_tot[:,1:],H_joint_tot.shape[1]-1,np.nan,axis=1) #insert column of nan to align matrices
H_xxc_mean = np.nanmean(np.append(H_xxc_tot,temp,axis=0),axis=0)

temp=np.insert(H_xxc_tot[:,:-1],0,np.nan,axis=1)
H_cond_mean = np.nanmean(np.append(temp,H_cond_tot,axis=0),axis=0)

MI_mean = H_gxc_mean + H_xxc_mean - H_joint_mean - H_cond_mean

import math 
H_G = 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.eye(2)))


fig1,ax1=plt.subplots(2,2)
fig1.suptitle('Entropy increase per added transmission')

ax1[0,0].cla(),ax1[0,0].plot(H_gxc_mean[1:]-H_gxc_mean[0:-1])
ax1[0,0].set_title('H(g,x_cond)'),ax1[0,0].set_ylabel('delta H()'),ax1[0,0].set_xlabel('T')
ax1[1,0].cla(),ax1[1,0].plot(H_joint_mean[1:]-H_joint_mean[0:-1])
ax1[1,0].set_title('H(g,x,x_cond)'),ax1[1,0].set_ylabel('delta H()'),ax1[1,0].set_xlabel('T')
ax1[0,1].cla(),ax1[0,1].plot(H_cond_mean[1:]-H_cond_mean[0:-1])
ax1[0,1].set_title('H(x_cond)'),ax1[0,1].set_ylabel('delta H()'),ax1[0,1].set_xlabel('T')
ax1[1,1].cla(),ax1[1,1].plot(H_xxc_mean[1:]-H_xxc_mean[0:-1])
ax1[1,1].set_title('H(x,x_cond)'),ax1[1,1].set_ylabel('delta H()'),ax1[1,1].set_xlabel('T')

fig1.tight_layout()

fig2,ax2=plt.subplots(2,2)
fig2.suptitle('Entropy increase per added transmission')

diff=H_gxc_mean[1:]-H_gxc_mean[0:-1]
yerr = np.nanvar(H_gxc_tot[:,1:],axis=0) + np.nanvar(H_gxc_tot[:,:-1],axis=0)
ax2[0,0].cla(),ax2[0,0].errorbar(T_range[:-1],diff,yerr=yerr)
ax2[0,0].set_title('H1(g,x_cond)'),ax2[0,0].set_ylabel('delta H()'),ax2[0,0].set_xlabel('T')

diff=H_joint_mean[1:]-H_joint_mean[0:-1]
yerr = np.nanvar(H_joint_tot[:,1:],axis=0) + np.nanvar(H_joint_tot[:,:-1],axis=0)
ax2[1,0].cla(),ax2[1,0].errorbar(T_range[:-1],diff,yerr=yerr)
ax2[1,0].set_title('H1(g,x,x_cond)'),ax2[1,0].set_ylabel('delta H()'),ax2[1,0].set_xlabel('T')

diff=H_cond_mean[1:]-H_cond_mean[0:-1]
yerr = np.nanvar(H_cond_tot[:,1:],axis=0) + np.nanvar(H_cond_tot[:,:-1],axis=0)
ax2[0,1].cla(),ax2[0,1].errorbar(T_range[:-1],diff,yerr=yerr)
ax2[0,1].set_title('H1(x_cond)'),ax2[0,1].set_ylabel('delta H()'),ax2[0,1].set_xlabel('T')

diff=H_xxc_mean[1:]-H_xxc_mean[0:-1]
yerr = np.nanvar(H_xxc_tot[:,1:],axis=0) + np.nanvar(H_xxc_tot[:,:-1],axis=0)
ax2[1,1].cla(),ax2[1,1].errorbar(T_range[:-1],diff,yerr=yerr)
ax2[1,1].set_title('H1(x,x_cond)'),ax2[1,1].set_ylabel('delta H()'),ax2[1,1].set_xlabel('T')

fig2.tight_layout()

fig3,ax3=plt.subplots(1,2)
temp_range = range(1,max(T_range)+1)
MI_mean=np.insert(MI_mean,0,0)
ax3[0].cla(),ax3[0].plot(temp_range,MI_mean),ax3[0].set_title('MI increase per T'),ax3[0].set_xlabel('T')
ax3[1].cla(),ax3[1].plot(temp_range,np.cumsum(MI_mean),label = 'I(X,G)')
ax3[1].axhline(y=H_G,linestyle='dashed', label = 'H(G)'),ax3[1].set_title('total MI'),ax3[1].set_xlabel('T')
ax3[1].legend()

fig3.tight_layout()

fig4,ax4=plt.subplots(1,2)
yerr=np.insert(np.nanvar(MI_tot,axis=0),0,0)
ax4[0].cla(),ax4[0].errorbar(temp_range,MI_mean,yerr=yerr),ax4[0].set_title('MI increase per T, error bars'),ax4[0].set_xlabel('T')
ax4[1].cla(),ax4[1].errorbar(temp_range,np.cumsum(MI_mean),yerr=np.cumsum(yerr)),ax4[1].set_title('total MI'),ax4[1].set_xlabel('T')
fig4.tight_layout()

fig5,ax5=plt.subplots(1,2)
T_matrix=np.tile(np.array(T_range),(MI_tot.shape[0],1))
ax5[0].cla(),ax5[0].scatter(T_matrix,MI_tot),ax5[0].set_title('MI increase per T'),ax5[0].set_xlabel('T')
ax5[1].cla(),ax5[1].scatter(T_matrix,np.cumsum(MI_tot,axis=1)),ax5[1].set_title('total MI'),ax5[1].set_xlabel('T')
fig5.tight_layout()

plt.show()

