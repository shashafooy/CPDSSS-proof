import numpy as np
import os

from simulators.CPDSSS_models import CPDSSS as CPDSSS
from misc_CPDSSS.entropy_util import MAF as ent
from misc_CPDSSS import util as misc


"""
Parameters for CPDSSS
"""
N = 6
L = 3
d0 = 3
d1 = N - d0
T_range = range(1, 6)
# T_range = [7, 8]
T_range = [6]
# T_range = [7,8,9]
REUSE = False
TRAIN_X = True
TRAIN_XH = True

"""
Number of iterations
"""
n_trials = 100  # iterations to average
min_knn_samples = 2000000  # samples to generate per entropy calc
n_train_samples = 100000
patience = 5
model = None


"""
File names
"""
orig_model_path = f"temp_data/saved_models/{N}N_d0d1({d0},{d1})"
new_model_path = f"temp_data/saved_models/new_models/{N}N_d0d1({d0},{d1})"

X_orig_path = os.path.join(orig_model_path, "X")
XH_orig_path = os.path.join(orig_model_path, "XH")

X_path = os.path.join(new_model_path, "X")
XH_path = os.path.join(new_model_path, "XH")


for i in range(n_trials):
    misc.print_border("Generating CPDSSS samples")
    sim_model = CPDSSS(T_range[-1], N, d0=d0, d1=d1)
    # generate base samples based on max dimension
    sim_model.set_dim_joint()
    knn_samples = int(max(min_knn_samples, 0.75 * n_train_samples * sim_model.x_dim))
    knn_samples = int(0.75 * n_train_samples * (10 * 6))
    X, XT, Xcond, h = sim_model.get_base_X_h(knn_samples)

    for k, T in enumerate(T_range):
        X_samp = X[:, : T * N]
        XH_samp = np.concatenate((X_samp, h), axis=1)

        name = f"{T}T"

        if TRAIN_X:
            misc.print_border(f"training H(X), T: {T}, iter: {i+1}")
            sim_model.x_dim = N * T
            if REUSE:
                model = ent.load_model(name=name, path=X_path)
                model = model if model is not None else ent.load_model(name=name, path=X_orig_path)
            model = ent.learn_model(
                sim_model, train_samples=X_samp, model=model, patience=patience
            ).model
            _ = ent.update_best_model(model, X_samp, name=name, path=X_path)
            model = ent.load_model(name=name, path=X_path)
            new_loss = model.eval_trnloss(X_samp)
            orig_model = ent.load_model(model, name, X_orig_path)
            orig_loss = orig_model.eval_trnloss(X_samp) if orig_model is not None else np.inf
            print(f"original loss: {orig_loss:.3f}")
            print(f"new loss: {new_loss:.3f}")
            model = None

        if TRAIN_XH:
            misc.print_border(f"training H(X,h), T: {T}, iter: {i+1}")
            sim_model.x_dim = N * T + N
            if REUSE:
                model = ent.load_model(name=name, path=XH_path)
                model = model if model is not None else ent.load_model(name=name, path=XH_orig_path)

            model = ent.learn_model(
                sim_model, train_samples=XH_samp, model=model, patience=patience
            ).model
            _ = ent.update_best_model(model, XH_samp, name=name, path=XH_path)
            model = ent.load_model(name=name, path=XH_path)
            new_loss = model.eval_trnloss(XH_samp)
            orig_model = ent.load_model(model, name, XH_orig_path)
            orig_loss = orig_model.eval_trnloss(XH_samp) if orig_model is not None else np.inf
            print(f"original loss: {orig_loss:.3f}")
            print(f"new loss: {new_loss:.3f}")
            model = None
