import numpy as np
import numpy as np

import os
import re

import simulators.CPDSSS_models as simMod
from misc_CPDSSS.entropy_util import Cond_MAF, MAF


# def compare_MAF_models(samples, name1, base_folder, name2, new_model_folder):
#     model = MAF.create_model(samples.shape[1])
#     current_loss = MAF.load_model(model, name1, base_folder).eval_trnloss(samples)
#     new_loss = MAF.load_model(model, name2, new_model_folder).eval_trnloss(samples)
#     print(f"{name1} current loss: {current_loss:.3f}, new loss: {new_loss:.3f}")
#     if new_loss < current_loss:
#         MAF.save_model(model, name1, base_folder)
#     else:
#         model = MAF.load_model(model, name1, base_folder)
#         MAF.save_model(model, name2, new_model_folder)


# def compare_condMAF_models(samples, name1, base_folder, name2, new_model_folder):
#     model = Cond_MAF.create_model(samples.shape[1])
#     current_loss = Cond_MAF.load_model(model, name1, base_folder).eval_trnloss(samples)
#     new_loss = Cond_MAF.load_model(model, name2, new_model_folder).eval_trnloss(samples)
#     print(f"{name1} current loss: {current_loss:.3f}, new loss: {new_loss:.3f}")
#     if new_loss < current_loss:
#         Cond_MAF.save_model(model, name1, base_folder)
#     else:
#         model = Cond_MAF.load_model(model, name1, base_folder)
#         Cond_MAF.save_model(model, name2, new_model_folder)


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
T = 10
sigma_A = 1
sigma_n = 2
n_samples = 2 * N * T * n_train_samples
A = np.random.normal(0, np.sqrt(sigma_A), (n_samples, N, N))
x = np.random.normal(0, 1, (n_samples, N, T))
n = np.random.normal(0, np.sqrt(sigma_n), (n_samples, N, T))
y = np.matmul(A, x) + n
sim_model = simMod.Gaussian(0, 1)
samples = [
    y.reshape(n_samples, N * T, order="F"),
    x.reshape(n_samples, N * T, order="F"),
]


base_folder = "temp_data/saved_models/new_models/conditional_tests"
for model_folder in os.listdir(base_folder):
    curr_path = os.path.join(base_folder, model_folder)
    # find N value in folder name
    N = int(re.findall(r"(\d{1,2})(?=N)", model_folder)[0])

    for folder in os.listdir(curr_path):
        if os.path.isfile(os.path.join(curr_path, folder)):
            continue
        use_MAF = False if "cond" in folder else True
        is_MAF_XY = True if use_MAF and "XY" in folder else False

        print(f"checking loss for folder {folder}")
        for file in os.listdir(os.path.join(curr_path, folder)):
            # find T value in file name
            T = int(re.findall(r"(\d+)(?=T)", file)[0])
            name = file.split(".")[0]

            # set paths for new model and original saved model
            new_model_path = os.path.join(curr_path, folder)
            segments = new_model_path.split("/")
            segments.remove("new_models")
            original_model_path = "/".join(segments)

            # Generate samples

            if use_MAF:
                if is_MAF_XY:
                    test_samples = np.concatenate([x[:, : N * T] for x in samples], axis=1)
                else:
                    test_samples = samples[1][:, : N * T]
                from misc_CPDSSS.entropy_util import MAF as ent
            else:
                test_samples = [x[:, : N * T] for x in samples]
                from misc_CPDSSS.entropy_util import Cond_MAF as ent

            # Compare with new model files
            print(f"checking loss for {T}T")

            new_model = ent.load_model(name=name, path=new_model_path)
            _ = ent.update_best_model(new_model, test_samples, name=name, path=original_model_path)
            # compare_models(X_samp, name, base_folder_X, name, new_folder_X)
            os.remove(os.path.join(new_model_path, name + ".pkl"))
        # os.rmdir(os.path.join(curr_path, folder))
    # os.rmdir(curr_path)
