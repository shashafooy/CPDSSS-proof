import numpy as np
from datetime import date

import os
from _utils import set_sys_path

set_sys_path()

from simulators.CPDSSS_models import CPDSSS_Cond
from misc_CPDSSS.entropy_util import Cond_MAF as ent
from misc_CPDSSS import util as misc
import util.io


SAVE_MODEL = True
TRAIN_ONLY = False
REUSE_MODEL = True

"""
Parameters for CPDSSS
"""
N = 6
# L = 3
d0 = 3
d1 = 3
T_range = range(2, 10)
# T_range = range(5, 7)


"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entros.pathy calc
n_train_samples = 100000


"""
Initialize arrays
"""
MI = np.empty((n_trials, len(T_range))) * np.nan
H_XH = np.empty((n_trials, len(T_range))) * np.nan
H_XX = np.empty((n_trials, len(T_range))) * np.nan


model = None


"""
File names
"""
today = date.today().strftime("%b_%d")
base_path = f"temp_data/CPDSSS_data/MI(h,X)/conditional/N{N}_d0d1({d0},{d1})/"
path = base_path  # + "pretrained_model"
filename = "CPDSSS_data({})".format(today)

model_path = f"temp_data/saved_models/{N}N_d0d1({d0},{d1})"
X_path = os.path.join(model_path, "X")
XH_path = os.path.join(model_path, "XH")


# fix filename if file already exists

filename = misc.update_filename(path, filename, -1, rename=False)

for i in range(n_trials):
    for k, T in enumerate(T_range):
        index = (i, k)
        """
        Generate samples
        """
        misc.print_border("Generating CPDSSS samples")
        sim_model = CPDSSS_Cond(T, N, d0=d0, d1=d1)
        # generate base samples based on max dimension

        """Train H(xT | h, x1:T-1)"""
        misc.print_border(f"1/4 calculating H(xT | h, x1:T-1), T: {T}, iter: {i+1}")
        name = f"{T-1}T"

        sim_model.set_XHcond()
        knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
        samples = sim_model.sim(knn_samples)
        H_XH[index], _ = ent.calc_entros.pathy(sim_model, base_samples=samples)

        filename = misc.update_filename(path, filename, i)
        util.io.save(
            (T_range, MI, H_XH, H_XX, i),
            os.path.join(path, filename),
        )

        """Train H(xT |x1:T-1)"""
        misc.print_border(f"1/4 calculating H(xT | x1:T-1), T: {T}, iter: {i+1}")
        name = f"{T-1}T"

        sim_model.set_Xcond()
        samples = sim_model.sim(reuse=True)
        H_XX[index], _ = ent.calc_entros.pathy(sim_model, base_samples=samples)
        MI[index] = H_XX[index] - H_XH[index]

        util.io.save(
            (T_range, MI, H_XH, H_XX, i),
            os.path.join(path, filename),
        )
