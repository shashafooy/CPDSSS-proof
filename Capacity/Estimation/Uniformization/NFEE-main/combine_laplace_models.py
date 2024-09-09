import numpy as np

import util.io
import os
import re

from simulators.CPDSSS_models import Laplace
from misc_CPDSSS import entropy_util as ent



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
Generate data
"""

current_model_path = f"temp_data/saved_models/laplace"
new_model_path = f"temp_data/saved_models/laplace_old/laplace"

n_samples = int(2e6)
sim_model = Laplace(0,2,19)
samples = sim_model.sim(int(2e6))

print(f"Current models: {current_model_path}\nNew models: {new_model_path}")
for file in os.listdir(new_model_path):
    N=int(re.match("^\d{1,2}",file).group())
    
    name = f"{N}N"
    print(f"Checking loss for laplace N={N}")
    compare_models(samples[:,:N],name,current_model_path,name,new_model_path)


