import numpy as np
import util.io
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import configparser
from datetime import date

from simulators.CPDSSS_models import CPDSSS
from misc_CPDSSS.entropy_util import MAF as ent
from misc_CPDSSS import util as misc


plt.rcParams["text.usetex"] = True


config = configparser.ConfigParser()
config.read("CPDSSS.ini")
KNN_THREADING = not config["GLOBAL"].getboolean("knn_GPU", False)  # Use threading if GPU not used


LOAD_SAVED = False

"""
Parameters for CPDSSS
"""
N = 6
L = 2
T = 3
stages = range(1, 10)


"""
Number of iterations
"""
knn_samples = 100000  # samples to generate per entropy calc

"""
File names
"""
today = date.today().strftime("%b_%d")
path = f"temp_data/indep_CPDSSS/N{N}"

filename = "correlation_data"
# filename = misc.update_filename(path, filename, -1, rename=False)


model_path = f"temp_data/saved_models/indep_test/"


if LOAD_SAVED:
    N, T, corr = util.io.load(os.path.join(path, filename))
else:

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
    corr = np.empty((len(stages) + 1, sim_model.x_dim, sim_model.x_dim)) * np.nan

    corr[0] = np.dot(joint.T, joint) / sim_model.x_dim

    for n_stage in stages:
        name = f"{n_stage}_stages"

        misc.print_border(f"Training with {n_stage} stages")

        model = ent.load_model(name=name, path=model_path)
        if model is None:
            sim_model.set_dim_joint()
            estimator = ent.learn_model(sim_model, train_samples=joint, n_stages=n_stage)
            model = estimator.model
            ent.update_best_model(model, joint, name=name, path=model_path)
        u = model.calc_random_numbers(joint)
        corr[n_stage] = np.dot(u.T, u) / u.shape[0]

        util.io.save((N, T, corr), os.path.join(path, filename))


# Get max,min of all correlation to scale colorbar
corr_db = 10 * np.log10(np.abs(corr))
color_min = min([corr_db[i].min() for i in range(len(stages) + 1)])
color_max = max([corr_db[i].max() for i in range(len(stages) + 1)])
norm = mcolors.Normalize(color_min, color_max)

ticks = [i * N for i in range(T + 1)]
labels = [r"$\mathbf{x}_{" + str(i) + "}$" for i in range(1, T + 1)] + [r"$\mathbf{h}$"]


plt.figure()
plt.imshow(corr_db[0], norm=norm)
plt.title("original data correlation")
plt.yticks(ticks, labels)
plt.xticks(ticks, labels)
cbar = plt.colorbar()
cbar.set_label("Magnitude (dB)")

for i in [1, stages[-1]]:
    plt.figure()
    plt.imshow(corr_db[i], norm=norm)
    plt.title(f"{i} MAF stages correlation")
    plt.yticks(ticks, labels)
    plt.xticks(ticks, labels)
    cbar = plt.colorbar()
    cbar.set_label("Magnitude (dB)")


plt.show()
