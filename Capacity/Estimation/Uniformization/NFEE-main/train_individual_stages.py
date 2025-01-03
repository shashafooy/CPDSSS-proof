from datetime import date
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ent_est import entropy
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io


import ml.step_strategies as ss


def plot_model_output(model, input, rows=4, cols=4):
    x = np.linspace(stats.norm.ppf(1e-6), stats.norm.ppf(1 - 1e-6), 100)
    x_pdf = stats.norm.pdf(x)

    fig, ax = plt.subplots(rows, cols)
    _row = 0
    _col = 0
    ax[_row, _col].plot(x, x_pdf, lw=2)
    ax[_row, _col].hist(input, bins=100, density=True), ax[_row, _col].set_title("Original Data")
    for i, _ in enumerate(n_stages):
        _col = (_col + 1) % cols
        _row = _row + 1 if _col == 0 else _row

        ax[_row, _col].plot(x, x_pdf, lw=2)
        ax[_row, _col].hist(model.calc_random_numbers(input, i), bins=100, density=True)
        ax[_row, _col].set_title(f"Stage {i+1} output")


n_train_samples = 200000
n_trials = 100
USE_PRETRAINED = False
val_tol = 0.0005
# fine_tune = not use_pretrained

TRAIN_ONE_SHOT = True

N = 10
n_stages = range(1, 4)

cols = 2
rows = int(np.ceil((n_stages[-1] + 1) / cols))

model_path = "temp_data/saved_models/laplace"
indep_stage_path = os.path.join(model_path, "indep_stages")


path = "temp_data/indep_stage_train"
today = date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = misc.update_filename(path=path, old_name=filename, rename=False)

H_indep = np.empty((n_trials, len(n_stages))) * np.nan
H_all_stages = np.empty((n_trials)) * np.nan


for k in range(n_trials):

    sim_laplace = mod.Laplace(mu=0, b=2, N=N)
    H_true = sim_laplace.entropy()
    laplace_base = sim_laplace.sim(n_train_samples * sim_laplace.x_dim)

    name = f"{n_stages[-1]}_stages"

    # Train all stages together
    misc.print_border(f"Iter {k}, Train all stages together")
    if TRAIN_ONE_SHOT:
        model = (
            ent.load_MAF_model(name=name, path=model_path, sim_model=sim_laplace)
            if USE_PRETRAINED
            else None
        )
        estimator = ent.learn_MAF_model(
            sim_laplace,
            model,
            n_train_samples,
            laplace_base,
            coarse_fine_tune=True,
            n_stages=n_stages[-1],
        )
        H_all_stages[k] = estimator.model.eval_trnloss(laplace_base)
        _ = ent.update_best_model(estimator.model, laplace_base, name=name, path=model_path)
        plot_model_output(estimator.model, laplace_base, rows=rows, cols=cols)

    n_inputs = sim_laplace.x_dim
    # load models
    model = ent.load_MAF_model(name=name, path=indep_stage_path, sim_model=sim_laplace)
    if model is None:
        fine_tune = True
        model = ent.create_MAF_model(n_inputs, n_mades=n_stages[-1], sim_model=sim_laplace)

    all_parms = model.parms
    trn_loss = model.trn_loss

    for i, ns in enumerate(n_stages):
        misc.print_border(f"Train stage {ns}, target H={H_true:.3f}")
        model.trn_loss = model.stage_loss[i]
        model.parms = model.mades[i].parms + model.bns[i].parms
        model._eval_trn_loss = None

        estimator = ent.learn_MAF_model(
            sim_laplace, model=model, train_samples=laplace_base, coarse_fine_tune=False
        )

        H_indep[k, i] = model.eval_trnloss(laplace_base)
    plot_model_output(model, laplace_base, rows=rows, cols=cols)
    plt.suptitle("Train Stages Individually")

    model.parms = all_parms
    model.trn_loss = trn_loss
    model._eval_trn_loss = None

    # _ = ent.learn_model(sim_laplace,pretrained_model=model,train_samples=laplace_base,fine_tune=False)
    # plot_model_output(model,laplace_base,rows=rows,cols=cols)
    # plt.suptitle("Train Stages Together")

    _ = ent.update_best_model(model, laplace_base, name=name, path=indep_stage_path)
    print(f"True entropy: {H_true}")
    util.io.save((H_all_stages, H_indep, n_stages[-1], N), os.path.join(filename, path))
