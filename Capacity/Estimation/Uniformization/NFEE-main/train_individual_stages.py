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


def plot_model_output(model,input):
    x=np.linspace(stats.norm.ppf(1e-6),stats.norm.ppf(1-1e-6),100)
    x_pdf = stats.norm.pdf(x)

    fig,ax=plt.subplots(rows,cols)
    _row=0
    _col=0
    ax[_row,_col].plot(x,x_pdf,lw=2)
    ax[_row,_col].hist(input,bins=100,density=True),ax[_row,_col].set_title("Original Data")
    for i,_ in enumerate(n_stages):
        _col = (_col+1)%cols
        _row=_row+1 if _col==0 else _row

        ax[_row,_col].plot(x,x_pdf,lw=2)
        ax[_row,_col].hist(model.calc_random_numbers(input,i),bins=100,density=True)
        ax[_row,_col].set_title(f"Stage {i+1} output")






n_train_samples = 100000
n_trials = 100
use_pretrained = True
val_tol=0.0005
# fine_tune = not use_pretrained

train_one_shot = True

N=10
n_stages = range(1,8)

cols=4
rows=int(np.ceil((n_stages[-1]+1)/cols))

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

    name = f'{n_stages[-1]}_stages'

    #Train all stages together
    misc.print_border(f"Iter {k}, Train all stages together")    
    if train_one_shot:
        model = ent.load_model(name=name,path=model_path,sim_model=sim_laplace) if use_pretrained else None
        fine_tune = True if model is None else False
        step = ss.Adam() if fine_tune else ss.Adam(a=1e-5)
        estimator = ent.learn_model(sim_laplace,model,n_train_samples,laplace_base,fine_tune=fine_tune,step=step,n_stages=n_stages[-1])
        H_all_stages[k] = estimator.model.eval_trnloss(laplace_base)
        _ = ent.update_best_model(estimator.model,laplace_base,name=name,path=model_path)
        plot_model_output(estimator.model,laplace_base)

    n_inputs = sim_laplace.x_dim
    #load models
    # models = [ent.load_model(name=f'{i}_stages',path=indep_stage_path) for i in n_stages]
    # fine_tune=True
    # #create model if it doesn't exist
    # models = [ent.create_model(n_inputs,n_mades=1) if model is None else model for model in models]
    # x = [laplace_base]
    # logdet = []
    model = ent.load_model(name=name,path=indep_stage_path,sim_model=sim_laplace)
    if model is None:
        fine_tune=True
        model = ent.create_model(n_inputs,n_hiddens=[512]*10,n_mades=n_stages[-1],sim_model=sim_laplace)

    all_parms = model.parms
    trn_loss = model.trn_loss

    # input_x = laplace_base

    for i,ns in enumerate(n_stages):
        misc.print_border(f"Train stage {ns}, target H={true_H:.3f}")
        model.trn_loss = model.stage_loss[i]
        model.parms = model.mades[i].parms + model.bns[i].parms
        
        # estimator = entropy.UMestimator(sim_laplace,models[i],x[i])
        print(f'Starting Loss: {model.eval_stageloss(laplace_base,i):.3f}')
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
        print(f'End loss: {H_indep[k,i]:.3f}')
    plot_model_output(model,laplace_base)

    model.parms = all_parms
    model.trn_loss=trn_loss
    _ = ent.update_best_model(model,laplace_base,name=name,path=indep_stage_path)
    print(f'True entropy: {true_H}')     
    util.io.save((H_all_stages,H_indep,n_stages[-1],N),os.path.join(filename,path))


