import numpy as np
import numpy as np

import os
import re

from simulators.CPDSSS_models import CPDSSS_Cond
from misc_CPDSSS.entropy_util import Cond_MAF as ent


def compare_models(samples, name1, base_folder, name2, new_model_folder):
    model = ent.create_model(samples.shape[1])
    current_loss = ent.load_model(model, name1, base_folder).eval_trnloss(samples)
    new_loss = ent.load_model(model, name2, new_model_folder).eval_trnloss(samples)
    print(f"{name1} current loss: {current_loss:.3f}, new loss: {new_loss:.3f}")
    if new_loss < current_loss:
        ent.save_model(model, name1, base_folder)
    else:
        model = ent.load_model(model, name1, base_folder)
        ent.save_model(model, name2, new_model_folder)


"""
Parameters for CPDSSS
"""
N = 6
L = 3
d0 = 3
d1 = N - d0

# min_samples = 2000000  # samples to generate per entropy calc
n_train_samples = 100000
max_T = 9

"""
Generate data
"""

current_model_path = f"temp_data/saved_models/conditional/{N}N_d0d1({d0},{d1})"
new_model_path = f"temp_data/saved_models/conditional/new_models/{N}N_d0d1({d0},{d1})"

base_folder_X = os.path.join(current_model_path, "X")
base_folder_XH = os.path.join(current_model_path, "XH")
new_folder_X = os.path.join(new_model_path, "X")
new_folder_XH = os.path.join(new_model_path, "XH")


sim_model = CPDSSS_Cond(2, N, d0=d0, d1=d1)
# generate base samples based on max dimension
knn_samples = int(n_train_samples * sim_model.x_dim)

print(f"Current models: {current_model_path}\nNew models: {new_model_path}")
if os.path.exists(new_folder_X):
    for file in os.listdir(new_folder_X):
        T = int(re.match("^\d{1,2}", file).group())
        sim_model.set_T(T)
        sim_model.set_Xcond()
        name = f"{T}T"

        samples = sim_model.sim(knn_samples, reuse_GQ=True)

        # Compare with new model files
        print(f"checking loss for {T}T X")
        new_model = ent.load_model(name=name, path=new_folder_X)
        _ = ent.update_best_model(new_model, samples, name=name, path=base_folder_X)
        # compare_models(X_samp, name, base_folder_X, name, new_folder_X)
        os.remove(os.path.join(new_folder_X, name + ".pkl"))
if os.path.exists(new_folder_XH):
    for file in os.listdir(new_folder_XH):
        T = int(re.match("^\d{1,2}", file).group())
        sim_model.set_T(T)
        sim_model.set_XHcond()
        name = f"{T}T"

        samples = sim_model.sim(knn_samples, reuse_GQ=True)
        # Compare with new model files
        print(f"checking loss for {T}T XH")
        new_model = ent.load_model(name=name, path=new_folder_XH)
        _ = ent.update_best_model(new_model, samples, name=name, path=base_folder_XH)
        # compare_models(XH_samp, name, base_folder_XH, name, new_folder_XH)
        os.remove(os.path.join(new_folder_XH, name + ".pkl"))
