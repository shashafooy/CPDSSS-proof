from datetime import date
import os
import numpy as np
from ent_est import entropy
import misc_CPDSSS.entropy_util as ent
import misc_CPDSSS.util as misc
import simulators.CPDSSS_models as mod
import util.io

import ml.step_strategies as ss








n_train_samples = 10000
n_trials = 100
use_pretrained = True
# fine_tune = not use_pretrained

N=16
n_stages = range(1,15)

model_path = 'temp_data/saved_models/laplace'
indep_stage_path = os.path.join(model_path,'indep_stages')


path = 'temp_data/indep_stage_train'
today=date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = misc.update_filename(path=path,old_name=filename,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

H_indep = np.empty((len(n_stages)))*np.nan





sim_laplace = mod.Laplace(mu=0,b=2,N=N)
true_H = sim_laplace.entropy()
laplace_base = sim_laplace.sim(n_train_samples * sim_laplace.x_dim)

#Train all stages together
misc.print_border("Train all stages together")
name = f'{n_stages[-1]}_stages'
# model = ent.load_model(name=name,path=model_path)
# fine_tune = True if model is None else False
# step = ss.Adam() if fine_tune else ss.Adam(a=1e-5)
# estimator = ent.learn_model(laplace_base,model,n_train_samples,laplace_base,fine_tune=fine_tune,step=step)
# H_all_stages = estimator.model.eval_trnloss(laplace_base)
# _ = ent.update_best_model(estimator.model,laplace_base,name=name,path=model_path)


n_inputs = sim_laplace.x_dim
#load models
# models = [ent.load_model(name=f'{i}_stages',path=indep_stage_path) for i in n_stages]
#create model if it doesn't exist
# models = [ent.create_model(n_inputs,n_mades=1) if model is None else model for model in models]

misc.print_border("Train stages independently")
model = ent.load_model(name=name,path=indep_stage_path)
if model is None:
    fine_tune=True
    model = ent.create_model(n_inputs,n_mades=n_stages[-1])

all_parms = model.parms

# input_x = laplace_base
for i,ns in enumerate(n_stages):
    misc.print_border(f"Train stage {ns}")
    model.trn_loss = model.stage_loss[i]
    model.parms = model.mades[i].parms + model.bns[i].parms



    # if model is None:
    #     fine_tune=True
    #     step = ss.Adam()
    #     model = ent.create_model(n_inputs,n_mades=1)
    #     models[i-1] = model
    # else:
    #     fine_tune=False
    #     step = ss.Adam(a=1e-5)


    # if model is None:
    #     fine_tune=True
    #     step = ss.Adam()
    #     model = ent.create_model(n_inputs,n_mades=n_stages[-1])

    estimator = entropy.UMestimator(sim_laplace,model,laplace_base)
    estimator.learn_transformation(
        n_samples = int(n_train_samples*sim_laplace.x_dim),
        fine_tune=fine_tune,
        step = ss.Adam() if fine_tune else ss.Adam(a=1e-5)
    )
    H_indep[i] = model.eval_stageloss(laplace_base,i)

model.parms = all_parms
_ = ent.update_best_model(model,laplace_base,name=name,path=indep_stage_path)
     



