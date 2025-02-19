from _utils import set_sys_path

set_sys_path()
from datetime import timedelta
import util.io
import os
import matplotlib.pyplot as plt
import numpy as np

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import viewData

plt.rcParams["text.usetex"] = True


"""
Load and combine all datasets
"""

REMOVE_OUTLIERS = True
COMBINE_ENTROPIES = True

filepath = "temp_data/retrain/coarse-fine_step"
N = 15

for filename in os.listdir(filepath):
    filename = os.path.splitext(filename)[0]  # remove extention
    _step_sizes, N, _val_error, _duration, _unif_KL, _unif_KSG = util.io.load(
        os.path.join(filepath, filename)
    )
    # _step_sizes = list(range(1,_val_error.shape[1]+1))

    # Initialize arrays
    if "val_error" not in locals():
        step_sizes = _step_sizes
        val_error = np.empty((0, len(_step_sizes)))
        test_error = np.empty((0, len(_step_sizes)))
        duration = np.empty((0, len(_step_sizes)))
        unif_KL = np.empty((0))
        unif_KSG = np.empty((0))

    val_error, _ = viewData.align_and_concatenate(
        val_error, _val_error, (step_sizes), (_step_sizes)
    )
    # test_error,_ = viewData.align_and_concatenate(duration,_duration,(step_sizes),(_step_sizes))
    duration, step_sizes = viewData.align_and_concatenate(
        duration, _duration, (step_sizes), (_step_sizes)
    )
    unif_KL, _ = viewData.align_and_concatenate(unif_KL, _unif_KL, (step_sizes), (_step_sizes))
    unif_KSG, _ = viewData.align_and_concatenate(unif_KSG, _unif_KSG, (step_sizes), (_step_sizes))

_, KL, KSG = util.io.load("temp_data/KL_KSG/N15_1M/kl_ksg")

viewData.clean_data(val_error)
viewData.clean_data(duration)
viewData.clean_data(unif_KL)
viewData.clean_data(unif_KSG)

# Remove any data that is outside of 3 standard deviations. These data points can be considered outliers.
if REMOVE_OUTLIERS:
    val_error = viewData.remove_outlier(val_error)
    duration = viewData.remove_outlier(duration)
    unif_KL = viewData.remove_outlier(unif_KL)
    unif_KSG = viewData.remove_outlier(unif_KSG)


mean_val_err = np.nanmean(val_error, axis=0)
mean_duration = np.round(np.nanmean(duration, axis=0))
mean_unif_KL = np.nanmean(unif_KL, axis=0)
mean_unif_KSG = np.nanmean(unif_KSG, axis=0)
mean_KL = np.nanmean(KL, axis=0)
mean_KSG = np.nanmean(KSG, axis=0)


var_err = np.nanvar(val_error, axis=0)
var_duration = np.nanvar(duration, axis=0)
var_unif_KL = np.nanvar(unif_KL, axis=0)
var_unif_KSG = np.nanvar(unif_KSG, axis=0)
var_KL = np.nanvar(KL, axis=0)
var_KSG = np.nanvar(KSG, axis=0)


H_true = Laplace(mu=0, b=2, N=N).entropy()

# num_trainings = val_error.shape[1]
# step_sizes = range(1,num_trainings+1)

# PLOTS

min_sec = []
for i in range(len(step_sizes)):
    min_sec.append(str(timedelta(seconds=mean_duration[i])))
    print(f"Mean training {i+1} duration: {min_sec[i]}")
tot_sec = np.cumsum(mean_duration)[-1].astype(np.int32).tolist()
print(f"Total train time: {str(timedelta(seconds = tot_sec))}")


plt.figure(1)
# plt.axhline(y=H_true,linestyle='--')
plt.plot(step_sizes, mean_val_err)
plt.xscale("log")
plt.title("validation error after retraining")
plt.xlabel("times model trained")
plt.ylabel("H(x) error")


plt.figure(2)
plt.axhline(y=H_true, linestyle="--")
plt.axhline(mean_KL, color="r")
plt.axhline(mean_KSG, color="b")
plt.axhline(mean_unif_KL, color="m")
plt.axhline(mean_unif_KSG, color="c")
plt.yscale("log")
plt.title("H(x) for different number of retrainings")
plt.legend(["True H(x)", "KL H(x)", "KSG H(x)", "unif KL H(x)", "unif KSG H(x)"])
plt.xlabel("SGD step size")
plt.ylabel("H(x)")

plt.show()
