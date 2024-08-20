from __future__ import division

# from itertools import zip
import numpy as np
import theano
import theano.tensor as tt

dtype = theano.config.floatX


class StepStrategy:
    """Abstract class for the step size strategy of stochastic gradient training."""

    def updates(self, parms, grads):
        """Given current gradient, return a list of updates to be made."""
        raise NotImplementedError("This is an abstract method and should be overriden.")


class ConstantStep(StepStrategy):
    """Step size strategy where the learning rate is held constant."""

    def __init__(self, step):
        """
        Constructor.
        :param step: the constant step size to be used
        """
        assert step > 0.0, "Step size must be positive."
        self.step = step

    def updates(self, parms, grads):
        """No updates to be made; step size is held constant throughout."""
        new_parms = [p - self.step * g for p, g in zip(parms, grads)]
        return list(zip(parms, new_parms))


class LinearDecay(StepStrategy):
    """Step size strategy where the learning rate is linearly decreased so as to
    hit zero after a specified number of iterations."""

    def __init__(self, init, maxiter):
        """
        Constructor.
        :param init: initial step size
        :param maxiter: maximum number of iterations.
        """
        assert init > 0.0, "Step size must be positive."
        assert (
            isinstance(maxiter, int) and maxiter > 0
        ), "Maximum number of iterations must be a positive integer."

        self.init = init
        self.maxiter = maxiter

    def updates(self, parms, grads):
        """Next step is linearly decayed."""
        step = theano.shared(np.asarray(self.init, dtype=theano.config.floatX), name="step")
        new_step = step - self.init / self.maxiter
        new_parms = [p - step * g for p, g in zip(parms, grads)]
        return [(step, new_step)] + list(zip(parms, new_parms))


class AdaDelta(StepStrategy):
    """ADADELTA step size strategy. For details, see:
    M. D. Zeiler, "ADADELTA: An adaptive learning rate method", arXiv, 2012."""

    def __init__(self, rho=0.95, eps=1.0e-6):
        """Constructor. Sets adadelta's hyperparameters."""
        assert eps > 0, "eps must be positive."
        assert 0 < rho < 1, "rho must be strictly between 0 and 1."

        self.eps = eps
        self.rho = rho

    def updates(self, parms, grads):
        """Return a list of updates to be made, both to adadelta's accumulators and the parameters."""

        acc_gs = [
            theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True) for p in parms
        ]
        acc_ds = [
            theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True) for p in parms
        ]

        new_acc_gs = [self.rho * ag + (1 - self.rho) * g**2 for g, ag in zip(grads, acc_gs)]
        ds = [
            tt.sqrt((ad + self.eps) / (ag + self.eps)) * g
            for g, ag, ad in zip(grads, new_acc_gs, acc_ds)
        ]
        new_acc_ds = [self.rho * ad + (1 - self.rho) * d**2 for d, ad in zip(ds, acc_ds)]
        new_parms = [p - d for p, d in zip(parms, ds)]

        return (
            list(zip(acc_gs, new_acc_gs))
            + list(zip(acc_ds, new_acc_ds))
            + list(zip(parms, new_parms))
        )


class Adam(StepStrategy):
    """Adam step size strategy. For details, see:
    D. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015."""

    def __init__(self, a=0.001, bm=0.9, bv=0.999, eps=1.0e-8):
        """Constructor. Sets adam's hyperparameters."""
        assert a > 0, "a must be positive."
        assert 0 < bm < 1, "bm must be strictly between 0 and 1."
        assert 0 < bv < 1, "bv must be strictly between 0 and 1."
        assert eps > 0, "eps must be positive."

        self.a = np.asarray(a).astype(theano.config.floatX)
        self.a_t = theano.shared(self.a)  # allows for a to be changed later on
        self.bm = np.asarray(bm, dtype=dtype)
        self.bv = np.asarray(bv, dtype=dtype)
        self.eps = np.asarray(eps, dtype=dtype)

        self.bm_t = None
        self.bv_t = None
        self.acc_m = None
        self.acc_v = None

    def updates(self, parms, grads):
        """Return a list of updates to be made, both to adams's running averages and the parameters."""

        self.bm_t = theano.shared(np.asarray(self.bm).astype(theano.config.floatX))
        self.bv_t = theano.shared(np.asarray(self.bv).astype(theano.config.floatX))

        new_bm_t = self.bm_t * self.bm
        new_bv_t = self.bv_t * self.bv

        self.acc_m = [
            theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True) for p in parms
        ]
        self.acc_v = [
            theano.shared(np.zeros_like(p.get_value(borrow=True)), borrow=True) for p in parms
        ]

        new_acc_m = [
            self.bm * am + (1 - self.bm).astype(dtype) * g for g, am in zip(grads, self.acc_m)
        ]
        new_acc_v = [
            self.bv * av + (1 - self.bv).astype(dtype) * g**2 for g, av in zip(grads, self.acc_v)
        ]

        step = self.a_t * tt.sqrt(1 - new_bv_t) / (1 - new_bm_t)
        eps = self.eps * (1 - new_bv_t)
        ds = [step * am / tt.sqrt(av + eps) for am, av in zip(new_acc_m, new_acc_v)]

        new_parms = [p - d for p, d in zip(parms, ds)]

        return (
            list(zip([self.bm_t, self.bv_t], [new_bm_t, new_bv_t]))
            + list(zip(self.acc_m, new_acc_m))
            + list(zip(self.acc_v, new_acc_v))
            + list(zip(parms, new_parms))
        )

    def reset_shared(self):
        if self.bm_t is not None:
            # reset only if updates() has run to initialize shared variables
            self.bm_t.set_value(np.asarray(self.bm).astype(theano.config.floatX))
            self.bv_t.set_value(np.asarray(self.bv).astype(theano.config.floatX))

            for am, av in zip(self.acc_m, self.acc_v):
                am.set_value(np.zeros_like(am.get_value(borrow=True)))
                av.set_value(np.zeros_like(av.get_value(borrow=True)))
        self.a_t.set_value(self.a)
