"""
Functions for generating the model and entropy
"""

from datetime import timedelta
import gc
import os
import time
import re
import multiprocessing as mp
import theano
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from ent_est import entropy
import ml.models.mafs as mafs
import util.io
from ml.trainers import ModelCheckpointer
import ml.step_strategies as ss
import ml.loss_functions as lf


dtype = theano.config.floatX


class _MAF_helper(ABC):
    @staticmethod
    def UM_KL_Gaussian(x):
        std_x = np.std(x, axis=0)
        z = stats.norm.cdf(x)
        return entropy.tkl(z) - np.mean(np.log(np.prod(stats.norm.pdf(x), axis=1)))

    @staticmethod
    def _save_model(model, name, path="temp_data/saved_models"):
        parms = [np.empty_like(p.get_value()) for p in model.parms]
        masks = [np.empty_like(m.get_value()) for m in model.masks]
        for i, p in enumerate(model.parms):
            parms[i] = p.get_value().copy()
        for i, m in enumerate(model.masks):
            masks[i] = m.get_value().copy()

        return parms, masks, model
        util.io.save(
            [parms, masks, model.n_inputs, model.n_hiddens, model.n_mades], os.path.join(path, name)
        )

    @classmethod
    def load_model(
        cls, model=None, name="model_name", path="temp_data/saved_models", sim_model=None
    ):
        try:
            params, masks, n_inputs, n_hiddens, n_mades = util.io.load(os.path.join(path, name))
        except FileNotFoundError:
            print(f"model {name} not found at {path}")
            return None

        model = (
            cls.create_model(n_inputs, n_hiddens=n_hiddens, n_mades=n_mades, sim_model=sim_model)
            if model is None
            else model
        )

        # assert len(params) == len(
        #     model.parms
        # ), "number of parameters is not the same, likely due to different number of stages"
        # assert (
        #     params[0].shape[0] == model.parms[0].get_value().shape[0]
        # ), f"invalid model input dimension. Expected {model.parms[0].get_value().shape[0]}, got {params[0].shape[0]}"
        # assert (
        #     params[0].shape[1] == model.parms[0].get_value().shape[1]
        # ), f"invalid model, number of nodes per hidden layer. Expected {model.parms[0].get_value().shape[1]}, got {params[0].shape[1]}"

        # if given model doesn't match loaded model, remake model
        if len(params) != len(model.parms) or params[0].shape != model.parms[0].get_value().shape:
            model = cls.create_model(
                n_inputs, n_hiddens=n_hiddens, n_mades=n_mades, sim_model=sim_model
            )

        for i, p in enumerate(params):
            model.parms[i].set_value(p.astype(dtype))

        for i, m in enumerate(masks):
            model.masks[i].set_value(m.astype(dtype))

        return model

    @staticmethod
    @abstractmethod
    def save_model(model, name, path="temp_data/saved_models"):
        pass

    @staticmethod
    @abstractmethod
    def create_model(
        n_inputs, rng=np.random, n_hiddens=[200, 200, 200], n_mades=14, sim_model=None
    ):
        pass

    @classmethod
    def update_best_model(
        cls,
        model,
        samples=None,
        sim_model=None,
        best_trn_loss=1e5,
        name="model_name",
        path="temp_data/saved_models",
    ):
        """Compare the given model with the saved model {name} located at {path}. If new model has lower training loss, save to given file

        Args:
            model (MaskedAutoregressiveFlow): new model to compare against
            samples (_type_): samples to find the training loss for
            best_trn_loss (_type_): current best training loss
            name (_type_): name of saved model
            path (str, optional): path to the model file. Defaults to 'temp_data/saved_models'.

        Returns:
            _type_: best error
        """
        if samples is None:
            assert sim_model is not None, "sim_model must be provided if samples is None"
            samples = sim_model.sim(100000 * sim_model.x_dim, reuse_GQ=True)

        new_loss = model.eval_trnloss(samples)
        checkpointer = ModelCheckpointer(model)
        checkpointer.checkpoint()

        # if best_trn_loss == np.Inf:
        old_model = cls.load_model(model, name, path)

        if old_model is not None:
            best_trn_loss = old_model.eval_trnloss(samples)
        checkpointer.restore()

        print(f"Saved best test loss: {best_trn_loss:.3f}, new model test loss: {new_loss:.3f}")
        if best_trn_loss > new_loss:
            print("new best loss")
            cls.save_model(model, name, path)
            return new_loss
        else:
            return best_trn_loss

    @classmethod
    def calc_entropy(
        cls,
        sim_model,
        n_train=10000,
        base_samples=None,
        model=None,
        reuse=True,
        method="umtksg",
        KNN_only=False,
        fine_tune=True,
    ):
        """Calculate entropy by uniformizing the data by training a neural network and evaluating the knn entropy on the uniformized points.
        This method does not implement any speed up from threading

        Args:
            sim_model (_type_): model used to generate points from target distribution. Must have method sim()
            n_train (_type_): number of samples used in training. This will be scaled by the dimensionality of the data.
            base_samples (numpy): Samples to be used in entropy estimate derived from sim_model.
            model (MaskedAutoregressiveFlow): Pretrained neural net. Create new model if set to None. Defaults to None.
            reuse (Boolean,optional): Set to True to use base_samples for both training and knn. Generates new samples for training if set to False. Default True
            method (str, optional): type of knn metric to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.


        Returns:
            _type_: entropy estimate
        """
        n_hiddens = [max(4 * sim_model.x_dim, 200)] * 3
        base_samples = base_samples if reuse else None
        if not KNN_only:
            estimator = cls.learn_model(
                sim_model,
                model,
                n_train,
                train_samples=base_samples,
                n_hiddens=n_hiddens,
                fine_tune=fine_tune,
            )
            # regenerate samples after training to be more generalized
            n_samp = (
                base_samples[0].shape[0]
                if isinstance(base_samples, list)
                else base_samples.shape[0]
            )
            base_samples = sim_model.sim(n_samp)
            print(f"New samples Loss: {estimator.model.eval_trnloss(base_samples):.4f}")
        else:
            if model is not None:
                print(f"Model Loss: {model.eval_trnloss(base_samples):.4f}")
            estimator = entropy.UMestimator(sim_model, model, base_samples)

        start_time = time.time()
        H = estimator.calc_ent(samples=base_samples, method=method)
        end_time = time.time()
        print(f"knn time: {str(timedelta(seconds = int(end_time - start_time)))}")
        if method == "both":
            print(f"tKL H={H[0]:.4f}\ntKSG H={H[1]:.4f}")
        else:
            print(f"knn H={H:.4f}")

        for i in range(3):
            gc.collect()

        # if method == "both":
        #     return (H[0], H[1]), estimator
        # else:
        return H, estimator

    @classmethod
    def calc_entropy_thread(
        sim_model, n_train=10000, base_samples=None, model=None, reuse=True, method="umtksg"
    ):
        """Train the MAF model, evaluate the uniformizing correction term, and launch the knn algorithm as a thread

        Args:
            sim_model (_type_): model used to generate points from target distribution. Must have method sim()
            n_train (_type_): number of samples used in training. This will be scaled by the dimensionality of the data.
            base_samples (numpy): Samples to be used in entropy estimate derived from sim_model.
            model (MaskedAutoregressiveFlow): Pretrained neural net. Create new model if set to None. Defaults to None.
            base_samples (_type_,optional): samples generated from the target distribution to be used in knn. Default None
            method (str, optional): type of knn metric to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.


        Returns:
            (thread,numpy): return started thread handle used for calculating entropy and the associated entropy correction term
        """
        estimator = _MAF_helper.learn_model(
            sim_model, model, n_train, train_samples=base_samples if reuse else None
        )
        # estimator.samples=base_samples
        uniform, correction = estimator.uniform_correction(base_samples)
        thread = estimator.start_knn_thread(uniform, method=method)
        return thread, correction, estimator

    @classmethod
    def learn_model(
        cls,
        sim_model,
        pretrained_model=None,
        n_samples=100,
        train_samples=[None],
        val_tol=0.0005,
        patience=5,
        n_hiddens=[200, 200, 200],
        n_stages=14,
        mini_batch=256,
        step=ss.Adam(),
        show_progress=False,
        fine_tune=True,
    ):
        """Create a MAF model and train it with the given parameters

        Args:
            sim_model (_type_): model to generate points from target distribution
            pretrained_model (MaskedAutoregressiveFlow, optional): pretrained neural net model. Create new model with random weights if set to none. Default to none
            n_samples (int, optional): number of samples to train on. Scaled by sim_model dimension. Defaults to 100.
            val_tol (float, optional): validation tolerance threshold to decide if model has improved. Defaults to 0.001.
            patience (int, optional): number of epochs without improvement before exiting training. Defaults to 5.
            n_hiddens (list, optional): number of hidden layers and nodes in a list. Defaults to [200,200].
            n_stages (int, optional): number of MAF stages. Defaults to 14.
            mini_batch (int, optional): Batch size for training. Defaults to 1024
            fine_tune (bool, optional): Set to True to run training twice, first with large step size, then a smaller step size. Defaults to True.
            show_progress (bool,optional): Set to true to print training curve. Defaults to False

        Returns:
            entropy.UMestimator: estimator object used for training and entropy calculation
        """

        if fine_tune and pretrained_model is not None:
            mini_batch = mini_batch * 4
            # step = ss.Adam(a=5e-5)
            # mini_batch = 1024
            patience = 10
            step = ss.Adam(a=1e-5)
        else:
            fine_tune = False  # Don't fine tune if model is not pretrained

        if pretrained_model is None:

            pretrained_model = cls.create_model(
                sim_model.input_dim, n_hiddens=n_hiddens, n_mades=n_stages, sim_model=sim_model
            )

        train_samples = train_samples if isinstance(train_samples, list) else [train_samples]

        regularizer = lf.WeightDecay(pretrained_model.parms, 1e-6)

        estimator = entropy.UMestimator(sim_model, pretrained_model, train_samples)
        if train_samples[0] is not None:
            print(f"Starting Loss: {pretrained_model.eval_trnloss(train_samples):.3f}")
        start_time = time.time()
        # estimator.learn_transformation(n_samples = int(n_samples*sim_model.x_dim*np.log(sim_model.x_dim) / 4),val_tol=val_tol,patience=patience)
        n_samples = (
            n_samples * sim_model.x_dim if train_samples[0] is None else train_samples[0].shape[0]
        )
        estimator.learn_transformation(
            n_samples=n_samples,
            val_tol=val_tol,
            patience=patience,
            coarse_fine_tune=fine_tune,
            minibatch=mini_batch,
            step=step,
            show_progress=show_progress,
            regularizer=regularizer,
        )
        end_time = time.time()
        print("learning time: ", str(timedelta(seconds=int(end_time - start_time))))
        if train_samples[0] is not None:
            print(f"Final Loss: {pretrained_model.eval_trnloss(train_samples):.4f}")

        return estimator

    @staticmethod
    def knn_entropy(estimator: entropy.UMestimator, base_samples=None, k=1, method="umtksg"):
        """Wrapper function to time knn entropy calculation from the given estimator
        Does not use any threading for speed up

        Args:
            estimator (entropy.UMestimator): estimator containing trained model
            base_samples (_type_, optional): distribution samples to be used for knn. Defaults to None.
            k (int, optional): k neighbors value for knn. Defaults to 1.
            method (str, optional): type of knn metric to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.

        Returns:
            _type_: Entropy value. If method='both' is used, then return tuple with entropy using KL and KSG
        """
        start_time = time.time()
        H = estimator.calc_ent(samples=base_samples, method=method, k=k)
        end_time = time.time()
        print("knn time: ", str(timedelta(seconds=int(end_time - start_time))))

        return H


class MAF(_MAF_helper):
    @staticmethod
    def save_model(model, name, path="temp_data/saved_models"):
        parms, masks, model = _MAF_helper._save_model(model, name, path="temp_data/saved_models")

        util.io.save(
            [parms, masks, model.n_inputs, model.n_hiddens, model.n_mades], os.path.join(path, name)
        )

    @staticmethod
    def create_model(
        n_inputs, rng=np.random, n_hiddens=[200, 200, 200], n_mades=14, sim_model=None
    ):
        """Generate a multi stage Masked Autoregressive Flow (MAF) model
        George Papamakarios, Theo Pavlakou, and Iain Murray. “Masked Autoregressive Flow for Density Estimation”

        Args:
            n_inputs (_type_): dimension of the input sample
            rng (_type_): type of rng generator to use
            n_hiddens (list, optional): number of hidden layers and hidden nodes per MAF stage. Defaults to [100,100].
            n_mades (int, optional): number of MAF stages. Defaults to 14.

        Returns:
            _type_: MAF model
        """
        # pdf = sim_model.logpdf if sim_model is not None else None
        entropy = sim_model.entropy() if sim_model is not None else None

        return mafs.MaskedAutoregressiveFlow(
            n_inputs=n_inputs,
            n_hiddens=n_hiddens,
            act_fun="tanh",
            n_mades=n_mades,
            input_order="random",
            mode="random",
            rng=rng,
            target_entropy=entropy,
        )


class Cond_MAF(_MAF_helper):
    @staticmethod
    def save_model(model, name, path="temp_data/saved_models"):
        parms, masks, model = _MAF_helper._save_model(model, name, path="temp_data/saved_models")

        util.io.save(
            [parms, masks, [model.n_inputs, model.n_givens], model.n_hiddens, model.n_mades],
            os.path.join(path, name),
        )

    @staticmethod
    def create_model(
        n_inputs, rng=np.random, n_hiddens=[200, 200, 200], n_mades=14, sim_model=None
    ):
        """Generate a multi stage Masked Autoregressive Flow (MAF) model
        George Papamakarios, Theo Pavlakou, and Iain Murray. “Masked Autoregressive Flow for Density Estimation”

        Args:
            n_inputs (_type_): dimension of the input sample
            rng (_type_): type of rng generator to use
            n_hiddens (list, optional): number of hidden layers and hidden nodes per MAF stage. Defaults to [100,100].
            n_mades (int, optional): number of MAF stages. Defaults to 14.

        Returns:
            _type_: MAF model
        """

        return mafs.ConditionalMaskedAutoregressiveFlow(
            n_givens=max(n_inputs[1], 1),
            n_inputs=n_inputs[0],
            n_hiddens=n_hiddens,
            act_fun="tanh",
            n_mades=n_mades,
            output_order="random",
            mode="random",
            rng=rng,
        )
