import numpy as np
import numpy as np

import os
import re

from simulators.CPDSSS_models import CPDSSS_Cond
from misc_CPDSSS.entropy_util import Cond_MAF as ent


"""
Parameters for CPDSSS
"""
N = 12
L = 3
d0 = 3
d1 = N - d0

# min_samples = 2000000  # samples to generate per entropy calc
n_train_samples = 100000

"""
Generate data
"""

base_folder = "temp_data/saved_models/new_models/conditional_noFading"
for model_folder in os.listdir(base_folder):
    curr_path = os.path.join(base_folder, model_folder)
    match = re.match(r"(\d{1,2})N_d0d1\((\d{1,2}),\s*(\d{1,2})\)", model_folder)
    N = int(match.group(1))
    d0 = int(match.group(2))
    d1 = int(match.group(3))
    assert N == d0 + d1, "invalid d0d1"

    sim_model = CPDSSS_Cond(2, N, d0=d0, d1=d1, use_fading=False)

    for folder in os.listdir(curr_path):
        print(f"comparing models {model_folder}/{folder}")
        for file in os.listdir(os.path.join(curr_path, folder)):
            T = int(re.match("^\d{1,2}", file).group())
            name = file.split(".")[0]
            sim_model.set_T(T)

            if folder == "X":
                sim_model.set_Xcond()
            else:
                sim_model.set_XHcond()

            new_model_path = os.path.join(curr_path, folder)
            segments = new_model_path.split("/")
            segments.remove("new_models")
            old_model_path = "/".join(segments)

            samples = sim_model.sim(n_train_samples * sim_model.x_dim, reuse_GQ=True)

            # Compare with new model files
            print(f"checking loss for {T}T")
            new_model = ent.load_model(name=name, path=new_model_path)
            _ = ent.update_best_model(new_model, samples, name=name, path=old_model_path)
            # compare_models(X_samp, name, base_folder_X, name, new_folder_X)
            os.remove(os.path.join(new_model_path, name + ".pkl"))
        os.rmdir(os.path.join(curr_path, folder))
    # os.rmdir(curr_path)
