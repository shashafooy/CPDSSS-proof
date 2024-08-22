import numpy as np
import util.io
import os
import matplotlib.pyplot as plt

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS import entropy_util as ent
from misc_CPDSSS import util as misc

from datetime import date

import configparser

config = configparser.ConfigParser()
config.read("CPDSSS.ini")
KNN_THREADING = not config["GLOBAL"].getboolean("knn_GPU", False)  # Use threading if GPU not used


REUSE_MODEL = False


"""
Parameters for CPDSSS
"""
N = 6
L = 2
T = 3
stages = range(1,10)


"""
Number of iterations
"""
knn_samples = 1000000  # samples to generate per entropy calc



"""
Generate samples
"""
misc.print_border(f"Generating {T}T CPDSSS samples")
sim_model = CPDSSS(T, N, L)
# generate base samples based on max dimension
sim_model.set_dim_joint()    
X, X_T, X_cond, h = sim_model.get_base_X_h(knn_samples)
hxc = np.concatenate((X_cond, h), axis=1)
joint = np.concatenate((X, h), axis=1)

"""
Initialize arrays
"""
corr = np.empty((len(stages)+1,sim_model.x_dim,sim_model.x_dim))*np.nan

corr[0] = np.dot(joint.T,joint)/sim_model.x_dim
fig,ax = plt.subplots(1,2)
fig.suptitle("Original data correlation")
ax[0].imshow(corr[0]),ax[0].set_title("correlation")
ax[1].imshow(np.log(np.abs(corr[0]))),ax[1].set_title("log correlation")
"""
File names
"""
today = date.today().strftime("%b_%d")
path = f"temp_data/indep_CPDSSS/N{N}"

filename = "correlation_data"
# filename = misc.update_filename(path, filename, -1, rename=False)


model_path = f"temp_data/saved_models/indep_test/"



for n_stage in stages:
    name=f"{n_stage}_stages"

    misc.print_border(f"Training with {n_stage} stages")
    
    model = ent.load_model(name=name,path=model_path)
    if model is None:    
        sim_model.set_dim_joint()
        estimator = ent.learn_model(sim_model,train_samples=joint,n_stages=n_stage)
        model=estimator.model
        ent.update_best_model(model,joint,name=name,path=model_path)
    u=model.calc_random_numbers(joint)
    corr[n_stage]=np.dot(u.T,u)/u.shape[0]

    util.io.save(
        (N,T,corr), os.path.join(path, filename)
    )
    fig,ax = plt.subplots(1,2)
    fig.suptitle(f"{n_stage} MAF stages correlation, loss = {model.eval_trnloss(joint):.3f}")
    ax[0].imshow(corr[n_stage]),ax[0].set_title("correlation")
    ax[1].imshow(np.log(np.abs(corr[n_stage]))),ax[1].set_title("log correlation")

plt.show()
