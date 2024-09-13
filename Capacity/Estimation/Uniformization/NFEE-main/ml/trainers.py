import os
import sys
import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import abc
import warnings

import ml.step_strategies as ss
import ml.data_streams as ds
import util.math
import util.ml

# from itertools import izip as zip


dtype = theano.config.floatX


class SGD_Template:
    """
    Minibatch stochastic gradient descent. Supports early stopping on validation set.
    This is an abstract template that has to be implemented by separate subclasses.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, model, trn_data, trn_target, val_data, val_target):
        """
        :param model: the model to be trained
        :param trn_data: training inputs and (possibly) training targets
        :param trn_target: theano variable representing the training target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_target: theano variable representing the validation target
        """

        self.val_target = val_target
        # prepare training data
        self.n_trn_data = self._get_n_data(trn_data)
        self.trn_data = [theano.shared(x.astype(dtype), borrow=True) for x in trn_data]
        self.minibatch = 100  # default value, may be overridden in train()
        self.monitor_every = None

        # compile theano function for a single training update
        self.trn_inputs = [model.input] if trn_target is None else [model.input, trn_target]
        self.make_update = None  # to be implemented by a subclass

        """Do not use batch_norm_stats due to memory consumption if dataset is too large"""
        # if model uses batch norm, compile a theano function for setting up stats.
        # A large enough dataset, and thus validation set should not need to evaluate batch norm mean/variances
        if getattr(model, "batch_norm", False) and self.n_trn_data < 50000:
            self.batch_norm_givens = [(bn.m, bn.bm) for bn in model.bns] + [
                (bn.v, bn.bv) for bn in model.bns
            ]
            self.set_batch_norm_stats = theano.function(
                inputs=[],
                givens=list(zip(self.trn_inputs, self.trn_data)),
                updates=[(bn.bm, bn.m) for bn in model.bns] + [(bn.bv, bn.v) for bn in model.bns],
            )
        else:
            self.batch_norm_givens = []
            self.set_batch_norm_stats = None

        # if validation data is given, then set up validation too
        self.do_validation = val_data is not None

        if self.do_validation:

            # prepare validation data
            self.n_val_data = self._get_n_data(val_data)
            self.val_data = [theano.shared(x.astype(dtype), borrow=True) for x in val_data]

            # compile theano function for validation
            self.val_inputs = [model.input] if val_target is None else [model.input, val_target]
            self.validate = None  # to be implemented by a subclass

            # create checkpointer to store best model
            self.checkpointer = ModelCheckpointer(model)
            self.best_val_loss = float("inf")

        # initialize some variables
        self.trn_loss = float("inf")
        self.idx_stream = ds.IndexSubSampler(self.n_trn_data, rng=np.random.RandomState(42))

    @abc.abstractmethod
    def reduce_step_size(self):
        """Method to update the step size used for self.make_update"""
        return

    def train(
        self,
        minibatch=None,
        tol=None,
        maxepochs=None,
        monitor_every=None,
        patience=None,
        logger=sys.stdout,
        show_progress=False,
        val_in_same_plot=True,
        val_Tol=None,
        fine_tune=False,
    ):
        """
        Trains the model.
        :param minibatch: minibatch size
        :param tol: tolerance
        :param maxepochs: maximum number of epochs
        :param monitor_every: monitoring frequency
        :param patience: maximum number of validation steps to wait for improvement before early stopping
        :param logger: logger for logging messages. If None, no logging takes place
        :param show_progress: if True, plot training and validation progress
        :param val_in_same_plot: if True, plot validation progress in same plot as training progress
        :param val_tol: Tolerance to decide if validation loss has improved enough
        :return: None
        """

        # parse input
        assert minibatch is None or util.math.isposint(
            minibatch
        ), "Minibatch size must be a positive integer or None."
        assert tol is None or tol > 0.0, "Tolerance must be positive or None."
        assert (
            maxepochs is None or maxepochs > 0.0
        ), "Maximum number of epochs must be positive or None."
        assert (
            monitor_every is None or monitor_every > 0.0
        ), "Monitoring frequency must be positive or None."
        assert patience is None or util.math.isposint(
            patience
        ), "Patience must be a positive integer or None."
        assert isinstance(show_progress, bool), "store_progress must be boolean."
        assert isinstance(val_in_same_plot, bool), "val_in_same_plot must be boolean."

        # initialize some variables
        self.iter = 0
        progress_epc = []
        progress_trn = []
        progress_val = []
        self.minibatch = self.n_trn_data if minibatch is None else minibatch
        maxiter = (
            float("inf")
            if maxepochs is None
            else np.ceil(maxepochs * self.n_trn_data / float(self.minibatch))
        )
        self.monitor_every = (
            float("inf")
            if monitor_every is None
            else np.ceil(monitor_every * self.n_trn_data / float(self.minibatch))
        )
        patience = float("inf") if patience is None else patience
        patience_left = patience
        best_epoch = None
        fine_tune_epoch = None
        logger = open(os.devnull, "w") if logger is None else logger

        # main training loop
        while True:

            # make update to parameters
            trn_loss = self.make_update(self.idx_stream.gen(self.minibatch))
            diff = self.trn_loss - trn_loss
            self.iter += 1
            self.trn_loss = trn_loss

            if self.iter % self.monitpythor_every == 0:

                epoch = self.iter * float(self.minibatch) / self.n_trn_data

                # do validation
                if self.do_validation:
                    if self.set_batch_norm_stats is not None:
                        self.set_batch_norm_stats()
                    val_loss = self.validate()
                    patience_left -= 1

                    if np.isnan(val_loss):
                        self.checkpointer.restore()

                    if val_loss < self.best_val_loss:
                        val_diff = self.best_val_loss - val_loss

                        # Save best validation if above tolerance
                        if val_Tol is not None and val_diff < val_Tol:
                            self.best_val_loss = self.best_val_loss
                            patience_left = patience_left
                        else:
                            self.best_val_loss = val_loss
                            patience_left = patience  # only reset patience if the improvement is larger than tolerance
                            logger.write(">")  # marking for best epoch

                        self.checkpointer.checkpoint()  # Still save best epoch, even if improvement is minor
                        best_epoch = epoch

                # monitor progress
                if show_progress:
                    progress_epc.append(epoch)
                    progress_trn.append(trn_loss)
                    if self.do_validation:
                        progress_val.append(val_loss)

                # log info
                if self.do_validation:
                    logger.write(
                        "Epoch = {0:.2f}, training loss = {1:.3f}, validation loss = {2:.3f}\n".format(
                            epoch, trn_loss, val_loss
                        )
                    )
                else:
                    logger.write(
                        "Epoch = {0:.2f}, training loss = {1:.3f}\n".format(epoch, trn_loss)
                    )

            # check for convergence
            if (tol is not None and abs(diff) < tol) or self.iter >= maxiter or patience_left <= 0:
                if self.do_validation:
                    self.checkpointer.restore()
                if self.set_batch_norm_stats is not None:
                    self.set_batch_norm_stats()
                if tol is not None and abs(diff) < tol:
                    logger.write("Below training tolerance\n")
                if patience_left <= 0:
                    logger.write("Ran out of patience\n")

                # check if we will do fine tuning
                if fine_tune:
                    logger.write("Start fine tuning\n")
                    # Reduce step size and resume training loop
                    self.reduce_step_size()
                    patience_left = patience
                    fine_tune = False
                    fine_tune_epoch = epoch

                    continue
                else:
                    self.release_shared_data()
                    break

        # plot progress
        if show_progress:

            if self.do_validation:

                if val_in_same_plot:
                    fig, ax = plt.subplots(1, 1)
                    ax.semilogx(progress_epc, progress_trn, "b", label="training")
                    ax.semilogx(progress_epc, progress_val, "r", label="validation")
                    # if fine_tune_epoch is not None:
                    #     ax.vlines(
                    #         fine_tune_epoch,
                    #         ax.get_ylim()[0],
                    #         ax.get_ylim()[1],
                    #         color="g",
                    #         linestyles="dashed",
                    #         label="Fine tuning",
                    #     )
                    if self.val_target is not None:
                        ax.axhline(self.val_target, label="target")
                        ax.set_title(
                            f"Training progress, error: {np.abs(self.best_val_loss-self.val_target):.3f}"
                        )
                    ax.set_ylim((0, ax.get_ylim()[1]))
                    ax.set_xlabel("epochs")
                    ax.set_ylabel("loss")
                    ax.set_title("Model Learning Curve")
                    ax.grid()
                    ax.legend()

                else:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.semilogx(progress_epc, progress_trn, "b")
                    ax2.semilogx(progress_epc, progress_val, "r")
                    ax1.vlines(
                        best_epoch,
                        ax1.get_ylim()[0],
                        ax1.get_ylim()[1],
                        color="g",
                        linestyles="dashed",
                        label="best",
                    )
                    ax2.vlines(
                        best_epoch,
                        ax2.get_ylim()[0],
                        ax2.get_ylim()[1],
                        color="g",
                        linestyles="dashed",
                        label="best",
                    )
                    ax2.set_xlabel("epochs")
                    ax1.set_ylabel("training loss")
                    ax2.set_ylabel("validation loss")

            else:
                fig, ax = plt.subplots(1, 1)
                ax.semilogx(progress_epc, progress_trn, "b")
                ax.set_xlabel("epochs")
                ax.set_ylabel("training loss")
                ax.legend()

            plt.show(block=False)

    @staticmethod
    def _get_n_data(data):
        """
        Given a list of data matrices, returns the number of data. Also checks if the matrix sizes are consistent.
        """

        n_data_list = set([x.shape[0] for x in data])
        assert len(n_data_list) == 1, "number of datapoints is not consistent"
        n_data = list(n_data_list)[0]

        return n_data

    def release_shared_data(self):
        for share in self.val_data:
            share.set_value([[]])
        for share in self.trn_data:
            share.set_value([[]])


class SGD(SGD_Template):
    """
    Minibatch stochastic gradient descent. Can work with a variety of step strategies,
    """

    def __init__(
        self,
        model,
        trn_data,
        trn_loss,
        trn_target=None,
        val_data=None,
        val_loss=None,
        val_target=None,
        step=ss.Adam(),
        max_norm=None,
    ):
        """
        Constructs and configures the trainer.
        :param model: the model to be trained
        :param trn_data: training inputs and (possibly) training targets
        :param trn_loss: theano variable representing the training loss to minimize
        :param trn_target: theano variable representing the training target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_loss: theano variable representing the validation loss
        :param val_target: theano variable representing the validation target
        :param step: step size strategy object
        :param max_norm: constrain the gradients to have this maximum norm, ignore if None
        """

        assert isinstance(step, ss.StepStrategy), "step must be a step strategy object"

        SGD_Template.__init__(self, model, trn_data, trn_target, val_data, val_target)

        self.trn_loss_func = trn_loss
        self.model = model
        self.step = step
        # compile theano function for a single training update
        idx = tt.ivector("idx")
        grads = tt.grad(trn_loss, model.parms)
        grads = [tt.switch(tt.isnan(g), 0.0, g) for g in grads]
        self.grads = grads if max_norm is None else util.ml.total_norm_constraint(grads, max_norm)
        # generate function for gradient descent using the given step algorithm and gradient
        self.make_update = theano.function(
            inputs=[idx],
            outputs=trn_loss,
            givens=list(zip(self.trn_inputs, [x[idx] for x in self.trn_data])),
            updates=step.updates(model.parms, self.grads),
        )

        if self.do_validation:

            # compile theano function for validation
            self.validate = theano.function(
                inputs=[],
                outputs=val_loss,
                givens=list(zip(self.val_inputs, self.val_data)) + self.batch_norm_givens,
            )
            if self.set_batch_norm_stats is not None:
                self.set_batch_norm_stats()
            self.best_val_loss = self.validate()

    def reduce_step_size(self):
        """update the step size to use smaller step size for fine tuning during training"""
        self.step.reset_shared()
        new_a = np.asarray(self.step.a_t.get_value() * 0.05).astype(theano.config.floatX)
        self.step.a_t.set_value(new_a)
        # also increase minibatch size, but don't exceed length of trn data
        # self.minibatch = (
        #     self.minibatch * 2 if self.minibatch * 2 < self.n_trn_data else self.minibatch
        # )
        if self.minibatch * 2 < self.n_trn_data:
            self.minibatch = self.minibatch * 2
            self.monitor_every = int(self.monitor_every / 2)
            self.iter = int(self.iter / 2)
        # step = ss.Adam(a=self.step.a*0.05, bm=self.step.bm, bv=self.step.bv, eps=self.step.eps)

        # idx = tt.ivector('idx')
        # self.make_update = theano.function(
        #     inputs=[idx],
        #     outputs=self.model.trn_loss,
        #     givens=list(zip(self.trn_inputs, [x[idx] for x in self.trn_data])),
        #     updates=step.updates(self.model.parms, self.grads)
        # )


class WeightedSGD(SGD_Template):
    """
    Minibatch stochastic gradient descent, where the loss per datapoint can be weighted.
    """

    def __init__(
        self,
        model,
        trn_data,
        trn_losses,
        trn_weights=None,
        trn_reg=None,
        trn_target=None,
        val_data=None,
        val_losses=None,
        val_weights=None,
        val_reg=None,
        val_target=None,
        step=ss.Adam(),
        max_norm=None,
    ):
        """
        :param model: the model to be trained
        :param trn_data: training inputs and (possibly) training targets
        :param trn_losses: theano variable representing the training losses at training points
        :param trn_weights: weights for training points
        :param trn_reg: theano variable representing the training regularizer
        :param trn_target: theano variable representing the training target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_losses: theano variable representing the validation losses at validation points
        :param val_weights: weights for validation points
        :param val_reg: theano variable representing the validation regularizer
        :param val_target: theano variable representing the validation target
        :param step: step size strategy object
        :param max_norm: constrain the gradients to have this maximum norm, ignore if None
        """

        assert isinstance(step, ss.StepStrategy), "step must be a step strategy object"

        SGD_Template.__init__(self, model, trn_data, trn_target, val_data, val_target)

        # prepare training weights
        trn_weights = np.ones(self.n_trn_data, dtype=dtype) if trn_weights is None else trn_weights
        trn_weights = theano.shared(trn_weights.astype(dtype), borrow=True)

        # prepare training regularizer
        trn_reg = 0.0 if trn_reg is None else trn_reg

        # compile theano function for a single training update
        idx = tt.ivector("idx")
        trn_loss = tt.mean(trn_weights[idx] * trn_losses) + trn_reg
        grads = tt.grad(trn_loss, model.parms)
        grads = [tt.switch(tt.isnan(g), 0.0, g) for g in grads]
        grads = grads if max_norm is None else util.ml.total_norm_constraint(grads, max_norm)
        self.make_update = theano.function(
            inputs=[idx],
            outputs=trn_loss,
            givens=list(zip(self.trn_inputs, [x[idx] for x in self.trn_data])),
            updates=step.updates(model.parms, grads),
        )

        if self.do_validation:

            # prepare validation weights
            val_weights = (
                np.ones(self.n_val_data, dtype=dtype) if val_weights is None else val_weights
            )
            val_weights = theano.shared(val_weights.astype(dtype), borrow=True)

            # prepare validation regularizer
            val_reg = 0.0 if val_reg is None else val_reg

            # compile theano function for validation
            self.validate = theano.function(
                inputs=[],
                outputs=tt.mean(val_weights * val_losses) + val_reg,
                givens=list(zip(self.val_inputs, self.val_data)) + self.batch_norm_givens,
            )

    def reduce_step_size(self):
        return super().reduce_step_size()


class ModelCheckpointer:
    """
    Helper class which makes checkpoints of a given model.
    Currently one checkpoint is supported; checkpointing twice overwrites previous checkpoint.
    """

    def __init__(self, model):
        """
        :param model: A machine learning model to be checkpointed.
        """
        self.model = model
        self.checkpointed_parms = [np.empty_like(p.get_value()) for p in model.parms]
        self.checkpointed_masks = [np.empty_like(m.get_value()) for m in model.masks]
        self.checkpoint()

    def checkpoint(self):
        """
        Checkpoints current model. Overwrites previous checkpoint.
        """
        is_valid = np.all([np.isin(m.get_value(), [0, 1]).all() for m in self.model.masks])
        if not is_valid:
            warnings.warn(
                "Warning.....Masks are not binary values, possible corruption. Not saving model weights or masks"
            )
            return
        for i, p in enumerate(self.model.parms):
            self.checkpointed_parms[i] = p.get_value().copy()
        for i, m in enumerate(self.model.masks):
            self.checkpointed_masks[i] = m.get_value().copy()

    def restore(self):
        """
        Restores last checkpointed model.
        """
        for i, p in enumerate(self.checkpointed_parms):
            self.model.parms[i].set_value(p)
        for i, m in enumerate(self.checkpointed_masks):
            self.model.masks[i].set_value(m.astype(dtype))
