from datetime import date
import os
import warnings
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
fine_tune=False
val_tol=0.0005
# fine_tune = not use_pretrained

N=16
n_stages = range(1,16)

model_path = 'temp_data/saved_models/laplace'
indep_stage_path = os.path.join(model_path,'indep_stages')


path = 'temp_data/indep_stage_train'
today=date.today().strftime("%b_%d")
filename = "laplace_data({})".format(today)
filename = misc.update_filename(path=path,old_name=filename,rename=False)
# util.io.save((N_range,H_unif_KL,H_KL_laplace,MSE_uniform,MSE_KL,iter),os.path.join(path,filename))

H_indep = np.empty((n_trials,len(n_stages)))*np.nan
H_all_stages = np.empty((n_trials))*np.nan



for k in range(n_trials):

    sim_laplace = mod.Laplace(mu=0,b=2,N=N)
    true_H = sim_laplace.entropy()
    laplace_base = sim_laplace.sim(n_train_samples * sim_laplace.x_dim)

    #Train all stages together
    misc.print_border(f"Iter {k}, Train all stages together")
    name = f'{n_stages[-1]}_stages'
    model = ent.load_model(name=name,path=model_path)
    fine_tune = True if model is None else False
    step = ss.Adam() if fine_tune else ss.Adam(a=1e-5)
    estimator = ent.learn_model(sim_laplace,model,n_train_samples,laplace_base,fine_tune=fine_tune,step=step)
    H_all_stages[k] = estimator.model.eval_trnloss(laplace_base)
    _ = ent.update_best_model(estimator.model,laplace_base,name=name,path=model_path)
    # print(f'True entropy: {true_H:.3f}')

    n_inputs = sim_laplace.x_dim
    #load models
    # models = [ent.load_model(name=f'{i}_stages',path=indep_stage_path) for i in n_stages]
    # fine_tune=True
    # #create model if it doesn't exist
    # models = [ent.create_model(n_inputs,n_mades=1) if model is None else model for model in models]
    # x = [laplace_base]
    # logdet = []
    model = ent.load_model(name=name,path=indep_stage_path)
    if model is None:
        fine_tune=True
        model = ent.create_model(n_inputs,n_mades=n_stages[-1])

    all_parms = model.parms
    trn_loss = model.trn_loss

    # input_x = laplace_base

    for i,ns in enumerate(n_stages):
        misc.print_border(f"Train stage {ns}, target H={true_H:.3f}")
        model.trn_loss = model.stage_loss[i]
        model.parms = model.mades[i].parms + model.bns[i].parms
        is_valid  =  np.all([np.isin(m.get_value(),[0,1]).all() for m in model.masks])
        if not is_valid:
            warnings.warn("Warning.....Masks are not binary values, possible corruption. Not saving model weights or masks")

        # estimator = entropy.UMestimator(sim_laplace,models[i],x[i])
        estimator = entropy.UMestimator(sim_laplace,model,laplace_base)
        estimator.learn_transformation(
            n_samples = int(n_train_samples*sim_laplace.x_dim),
            fine_tune=fine_tune,
            step = ss.Adam() if fine_tune else ss.Adam(a=1e-5),
            val_tol=val_tol
        )
        # x.append(models[i].calc_random_numbers(x[i]))
        # logdet.append(models[i].logdet_jacobi_u(x[i]))
        # L = -n_inputs/2*np.log(2*np.pi) - 0.5*np.sum(x[-1]**2,axis=1) + np.sum(logdet,axis=0)
        # H_indep[i] = -np.mean(L)
        H_indep[k,i] = model.eval_stageloss(laplace_base,i)
        print(f'Stage entropy: {H_indep[k,i]}')

    model.parms = all_parms
    model.trn_loss=trn_loss
    _ = ent.update_best_model(model,laplace_base,name=name,path=indep_stage_path)
    print(f'True entropy: {true_H}')     
    util.io.save((H_all_stages,H_indep,n_stages[-1],N),os.path.join(filename,path))


