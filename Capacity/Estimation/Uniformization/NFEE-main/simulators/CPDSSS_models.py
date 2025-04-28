import abc
from datetime import timedelta
import gc
import numpy as np
import scipy.linalg as lin
import scipy.special as spec
import scipy.stats as stats
from simulators.complex import mvn, mvn_complex
import math
import theano.tensor as tt
import theano
import time
import sys

import util.misc
from util.math import zadoff_chu

dtype = theano.config.floatX
USE_GPU = False


# from memory_profiler import profile


class _distribution:
    __metaclass__ = abc.ABCMeta

    def __init__(self, x_dim=-1, rng=np.random):
        self._x_dim = x_dim
        self._input_dim = x_dim
        self.rng = rng.default_rng()

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def x_dim(self):
        return self._x_dim

    @input_dim.setter
    def input_dim(self, val):
        # Validate the input to ensure it's either an int or a list of ints
        if isinstance(val, int):
            pass  # Valid case for int
        elif isinstance(val, list) and all(isinstance(x, int) for x in val):
            pass  # Valid case for list of integers
        else:
            raise ValueError("input_dim must be an integer or a list of integers.")

        self._input_dim = val
        self.update_x_dim()

    def update_x_dim(self):
        if isinstance(self.input_dim, list):
            self._x_dim = sum(self.input_dim)
        else:
            self._x_dim = self.input_dim

    @x_dim.setter
    def x_dim(self, val):
        self._x_dim = val
        self._input_dim = val

    @abc.abstractmethod
    def sim(self):
        """Method to generate sample from the distribution"""
        return

    def entropy(self):
        """Method returning the entropy of the given distribution. Use default method if entropy unknown.
        Defaults to returning none."""
        return None

    def pdf(self, x):
        """PDF of the distribution"""
        return 0

    def logpdf(self, x):
        """log(pdf) of the distribution. Useful for entropy calculations"""
        return 0


class CPDSSS(_distribution):
    def __init__(self, num_tx, N, L=None, d0=None, d1=None, use_fading=True):
        super().__init__(x_dim=N * num_tx + N)

        self.N = N
        self.L = L

        self._use_chan = False

        if L is not None:
            assert N / L % 1 == 0, "N/L must be an integer"
            self.sym_N = int(N / L)
            self.noise_N = N - self.sym_N
            self.G_slice = range(0, N, L)
            # if USE_GPU:
            self.tt_G_slice = theano.shared(np.arange(0, N, L, dtype=np.int32))
        elif d0 is not None and d1 is not None:
            assert d0 + d1 == N, "d0+d1 must be equal to N"
            self.sym_N = d0
            self.noise_N = min(d1, N - 1)  # cannot make Q with d1=N
            if d0 != 0 and N / d0 % 1 == 0:  # same as using L
                self.G_slice = range(0, N, int(N / d0))
                # if USE_GPU:
                self.tt_G_slice = theano.shared(np.arange(0, N, int(N / d0), dtype=np.int32))
            elif d0 == 0:  # treat d0 as 1 and make G=0
                self.G_slice = range(0, 1)
                self.tt_G_slice = theano.shared(np.arange(0, 1, dtype=np.int32))
            else:
                self.G_slice = range(0, d0)
                # if USE_GPU:
                self.tt_G_slice = theano.shared(np.arange(0, d0, dtype=np.int32))
        else:
            raise ValueError(
                "Invalid input. Require either L or d0,d1 as inputs. N/L must be an integer, or d0+d1=N"
            )

        self.set_T(num_tx)

        self.sim_S = mvn(rho=0.0, dim_x=self.sym_N * self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.noise_N * self.T)
        self.sim_H = mvn(rho=0.0, dim_x=self.N)

        # self.use_chan_in_sim()
        self._sim_chan_only = False

        self.fading = (
            np.exp(-np.arange(self.N) / 3).astype(dtype)
            if use_fading
            else np.ones(self.N).astype(dtype)
        )

        self.h = np.empty((0, N))
        self.G = np.empty((0, N, self.sym_N), dtype=dtype)
        self.Q = np.empty((0, N, self.noise_N), dtype=dtype)

        # For larger N, the regularization factor may need to increase so H^T*H is not singular
        if self.N > 10:
            self.eye = 0.0005 * np.eye(self.N).astype(dtype)
            self.tt_eye = 0.0005 * tt.eye(self.N, dtype=dtype)
        else:
            self.eye = 0.0001 * np.eye(self.N).astype(dtype)
            self.tt_eye = 0.0001 * tt.eye(self.N, dtype=dtype)

        self.tt_GQ_func = None

    def sim(self, n_samples=1000, reuse_GQ=True):
        """wrapper allowing inherited class to reuse while _sim() retains the core functionality

        Args:
            n_samples (int, optional): number of samples to generate. Defaults to 1000.
        """
        return self._sim(n_samples, reuse_GQ)

    def _sim(self, n_samples=1000, reuse_GQ=True):
        """Generate samples X and (optional) G for CPDSSS

        Args:
            n_samples (int, optional): number of input samples. Defaults to 1000.

        Returns:
            numpy: (n_samples, dim_x) array of generated values
        """
        N = self.sym_N + self.noise_N  # This may not match self.N if evaluating complex values

        s = self.sim_S.sim(n_samples=n_samples).reshape((n_samples, self.sym_N, self.T))

        v = self.sim_V.sim(n_samples=n_samples).reshape((n_samples, self.noise_N, self.T))

        # channel is reused, append new samples if needed
        new_samples = n_samples - self.h.shape[0] if reuse_GQ else n_samples
        new_h = (
            (self.sim_H.sim(n_samples=new_samples) * np.sqrt(self.fading))
            if new_samples > 0
            else np.empty((0, N), dtype=s.dtype)
        )
        self.h = np.concatenate((self.h, new_h), axis=0, dtype=new_h.dtype) if reuse_GQ else new_h
        del new_h
        # self.h = (self.sim_H.sim(n_samples=n_samples) * np.sqrt(self.fading)).astype(dtype)

        if self._sim_chan_only:  # return early if we only need channel
            self.samples = self.h
            return self.h

        self.sim_GQ(reuse_GQ, new_samples)
        if self.sym_N == N:
            X = np.matmul(self.G[:n_samples, :, :], s)
        else:
            X = np.matmul(self.G[:n_samples, :, :], s) + np.matmul(self.Q[:n_samples, :, :], v)
        del s, v
        gc.collect()
        joint_X = X[:, :, : self.T].reshape(
            (n_samples, N * self.T), order="F"
        )  # order 'F' needed to make arrays stack instead of interlaced
        del X
        gc.collect()
        if self._use_chan:
            return np.concatenate((joint_X, self.h[:n_samples]), axis=1)
        else:
            return joint_X

        return samples

    # @profile
    def sim_GQ(self, reuse, new_samples=0):
        n_samples, N = self.h.shape
        # if stored G samples is less than number of h samples, generate more G,Q
        if reuse and self.G.shape[0] >= n_samples:
            return

        curr_dtype = self.h.dtype  # used if new_samples is complex

        # new_samples = n_samples - self.G.shape[0] if reuse else n_samples
        G = np.empty((0, N, self.sym_N), dtype=curr_dtype)
        Q = np.empty((0, N, self.noise_N), dtype=curr_dtype)

        split_N = max(np.floor(new_samples / 100000), 1)
        sections = np.array_split(range(n_samples - new_samples, n_samples), split_N)

        # self._gen_batch_GQ_sample(self.h[sections[0]])
        if self.tt_GQ_func is None and USE_GPU:
            self.tt_GQ_func = self._gen_tt_GQ_func()

        # For large N, inv(H^T*H + delta*eye(N)) can be singular.
        # Regenerate h if this is the case and run again.
        util.misc.printProgressBar(0, split_N, "G,Q generation")
        start = time.time()
        for i, section in enumerate(sections):
            singular = True
            while singular:
                try:
                    if USE_GPU:
                        new_G, new_Q = self.tt_GQ_func(self.h[section, :])
                    else:  # using batch numpy CPU functions is MUCH faster
                        new_G, new_Q = self._gen_batch_GQ_sample(self.h[section, :])
                    G = np.concatenate((G, new_G), axis=0)
                    Q = np.concatenate((Q, new_Q), axis=0)
                    singular = False
                    util.misc.printProgressBar(i + 1, split_N, "G,Q generation ")
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as inst:  # regenerate h if inv(H'H) is singular
                    self.h[section] = (
                        self.sim_H.sim(n_samples=len(section)) * np.sqrt(self.fading)
                    ).astype(dtype)
                    util.misc.printProgressBar(i, split_N, "Singular, rerun")

        self.G = np.concatenate((self.G, G), axis=0) if reuse else G
        self.Q = np.concatenate((self.Q, Q), axis=0) if reuse else Q
        print(f"G,Q time {str(timedelta(seconds=int(time.time() - start)))}")

    # def _gen_GQ_sample(self, h):

    #     # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
    #     # A=E.T @ H
    #     # Slightly faster than doing E.T @ H
    #     HE = lin.toeplitz(h, np.concatenate(([h[0]], h[-1:0:-1])))[self.G_slice, :]

    #     R = HE.T @ HE + self.eye
    #     p = HE[0, :].T
    #     # g = lin.inv(R) @ p
    #     if self.sym_N == 0:
    #         G = np.zeros((self.N, 0))
    #     else:
    #         g = np.linalg.solve(R, p) if self.sym_N > 0 else np.zeros(self.N)
    #         # Only take every L columns of toepltiz matrix
    #         # Slightly faster than doing G@E
    #         G = lin.toeplitz(g, np.concatenate(([g[0]], g[-1:0:-1])))[:, self.G_slice]
    #         # Potentially better G, fullfills G.T*Q=0 and HE * G = I
    #         # G = lin.inv(R) @ HE[self.G_slice,:].T

    #     # ev, V = lin.eig(R)
    #     ev, V = np.linalg.eigh(R)  # R is symmetric, eigh is optimized for symmetric
    #     # Only use eigenvectors associated with "zero" eigenvalue
    #     #  get indices of the smallest eigenvalues to find "zero" EV
    #     sort_indices = np.argsort(ev)
    #     # Sometimes get a very small imaginary value, ignore it
    #     Q = np.real(V[:, sort_indices[0 : self.noise_N]])

    #     return G, Q

    def _gen_batch_GQ_sample(self, h):

        # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
        # A=E.T @ H
        # Slightly faster than doing E.T @ H
        # HE = lin.toeplitz(h, np.concatenate(([h[0]], h[-1:0:-1])))[self.G_slice, :]
        n_samp, N = h.shape

        def batch_toeplitz(h):
            row = np.concatenate([h[:, [0]], h[:, -1:0:-1]], axis=1)
            col_idx = np.arange(N).reshape(N, 1)
            row_idx = np.arange(N).reshape(1, N)
            idx = col_idx - row_idx
            # idx = idx[self.G_slice, :]
            idx = np.broadcast_to(idx, (h.shape[0], *idx.shape))

            return np.where(
                idx >= 0,
                np.take_along_axis(h[:, None, :], idx, axis=2),
                np.take_along_axis(row[:, None, :], -idx, axis=2),
            )

        HE = batch_toeplitz(h)[:, self.G_slice, :]

        R = np.matmul(HE.transpose(0, 2, 1), HE) + self.eye
        # R = HE.T @ HE + self.eye
        p = HE[:, 0, :]
        # g = lin.inv(R) @ p
        if self.sym_N == 0:
            G = np.zeros((n_samp, N, 0))
        else:
            g = np.linalg.solve(R, p) if self.sym_N > 0 else np.zeros((n_samp, N))
            # Only take every L columns of toepltiz matrix
            # Slightly faster than doing G@E
            G = batch_toeplitz(g)[:, :, self.G_slice]
            # G = lin.toeplitz(g, np.concatenate(([g[0]], g[-1:0:-1])))[:, self.G_slice]
            # Potentially better G, fullfills G.T*Q=0 and HE * G = I
            # G = lin.inv(R) @ HE[self.G_slice,:].T

        # ev, V = lin.eig(R)
        ev, V = np.linalg.eigh(R)  # R is symmetric, eigh is optimized for symmetric
        # Only use eigenvectors associated with "zero" eigenvalue
        #  get indices of the smallest eigenvalues to find "zero" EV
        sort_indices = np.argsort(ev, axis=1)[:, : self.noise_N]
        batch_idx = np.arange(n_samp)[:, np.newaxis]

        # Sometimes get a very small imaginary value, ignore it
        Q = V[batch_idx, :, sort_indices].transpose(
            0, 2, 1
        )  # shape (samples,noise_N,N) -> (samples,N,noise_N)
        Q = np.real(Q) if Q.dtype == dtype else Q

        return G, Q

    def _gen_tt_GQ_func(self):
        # # Define symbolic input
        # h_batch = tt.matrix("h_batch")

        # def toeplitz(col):
        #     N = col.shape[0]
        #     row = tt.concatenate([col[:1], col[-1:0:-1]])
        #     col_idx = tt.arange(N).reshape((N, 1))
        #     row_idx = tt.arange(N).reshape((1, N))
        #     indices = col_idx - row_idx
        #     return tt.switch(indices >= 0, col[indices], row[-indices])

        # def process_GQ(h):
        #     HE = toeplitz(h)[self.tt_G_slice, :]  # shape (len(G_slice), N)
        #     R = tt.dot(HE.T, HE) + self.tt_eye  # shape (N, N)
        #     p = HE[0, :]

        #     # g = solve(R, p)
        #     g = tt.slinalg.solve(R, p)  # shape (N,)

        #     G = toeplitz(g).T[self.tt_G_slice, :].T  # shape (N, len(G_slice))

        #     # eigendecomposition
        #     ev, V = tt.nlinalg.eigh(R)

        #     # sort eigenvalues
        #     sorted_idx = tt.argsort(ev)
        #     Q = V[:, sorted_idx[: self.noise_N]]  # smallest `noise_N` eigenvectors

        #     return G, Q

        # [G_batch, Q_batch], _ = theano.scan(fn=process_GQ, sequences=[h_batch])

        # return theano.function(
        #     inputs=[h_batch], outputs=[G_batch, Q_batch], allow_input_downcast=True
        # )

        h = tt.matrix("h")
        n_samp, N = h.shape

        # Create toeplitz matrix and decimate across its rows
        def gen_HE(col):
            """Generate a toeplitz matrix and slice it such that toeplitz[G_slice,:]"""
            N = col.shape[0]
            # first_col = col
            # first_row = tt.concatenate([col[:1], col[-1:0:-1]])

            row = col[::-1]  # reverse column vector
            # Construct indices for the Toeplitz matrix
            col_idx = self.tt_G_slice.dimshuffle(0, "x")  # shape: (len(G_slice),1)
            row_idx = tt.arange(N).dimshuffle("x", 0)  # shape (1, N)
            # col_idx = tt.arange(N).reshape((N,1))
            # row_idx = tt.arange(N).reshape((1,N))
            indices = col_idx - row_idx

            # Use T.switch to fill values based on indices
            # toeplitz_matrix = tt.switch(indices >= 0, first_col[indices], first_row[-indices])
            toeplitz_matrix = tt.switch(indices >= 0, col[indices], row[-indices - 1])

            return toeplitz_matrix

        HE = theano.scan(fn=gen_HE, sequences=[h], non_sequences=[])[0]
        # return theano.function(inputs=[h], outputs=[HE], allow_input_downcast=True)

        # Compute R, p, and g
        R = tt.batched_dot(HE.transpose(0, 2, 1), HE) + self.tt_eye
        # return theano.function(inputs=[h], outputs=[R], allow_input_downcast=True)
        p = HE[:, 0, :]

        # Solve g = inv(R) * p
        # Make toeplitz matrix and decimate across its columns
        def make_G(R, p):
            if self.sym_N == 0:  # return 0 vector if there are 0 symbols
                return tt.zeros((N, 0))
            col = tt.slinalg.solve(R, p)  # g vector R*g = p

            """Generate a toeplitz matrix HE and slice it such that HE[:,G_slice]"""
            row = col[::-1]  # reverse column vector
            # Construct indices for the Toeplitz matrix
            col_idx = tt.arange(col.shape[0]).dimshuffle(0, "x")  # shape (1, len(row))
            row_idx = self.tt_G_slice.dimshuffle("x", 0)  # shape: (len(G_slice),1)
            indices = col_idx - row_idx

            # Use T.switch to fill values based on indices
            toeplitz_matrix = tt.switch(indices >= 0, col[indices], row[-indices - 1])

            return toeplitz_matrix

        G = theano.scan(fn=make_G, sequences=[R, p], non_sequences=[])[0]

        # return theano.function(inputs=[h], outputs=G)

        # Q = theano.scan(fn=make_Q, sequences=[R])[0]
        # return theano.function(inputs=[h], outputs=[G, Q], allow_input_downcast=True)

        # Compute eigenvalues and eigenvectors of R
        def compute_ev(r):
            ev, V = tt.nlinalg.eigh(r)
            return ev, V

        ev_b, V_b = theano.scan(fn=compute_ev, sequences=[R])[0]

        # obtain vectors associated with smallest ev.
        # Some weird indexing here due to shape of sort_indices
        sort_indices = tt.argsort(ev_b, axis=1)[:, : self.noise_N]  # shape (samples, noise_N)
        batch_idx = tt.arange(sort_indices.shape[0]).dimshuffle(0, "x")  # shape (samples,1)
        Q = V_b[batch_idx, :, sort_indices].dimshuffle(
            0, 2, 1
        )  # shape (samples,noise_N,N) -> (samples,N,noise_N)

        # Define Theano function
        # return theano.function(inputs=[h], outputs=[G, Q, ev_b, V_b], allow_input_downcast=True)
        return theano.function(inputs=[h], outputs=[G, Q], allow_input_downcast=True)

    def chan_entropy(self):
        N = self.sym_N + self.noise_N
        return N / 2 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(
            np.linalg.det(np.diag(self.fading))
        )
        return 0.5 * np.log(np.linalg.det(2 * math.pi * np.exp(1) * np.diag(self.fading)))

    def use_chan_in_sim(self, h_flag=True):
        """Enables sim to append G in addition to X

        Args:
            g_flag (bool, optional): Enable G output. Defaults to True.
        """
        self._use_chan = h_flag
        if h_flag:
            self.input_dim = self.N * self.T + self.N
        else:
            self.input_dim = self.N * self.T

    def only_sim_chan(self, sim_h=True):
        """Set flag to only return G when using sim()

        Args:
            sim_g (bool): True if you want sim() to return only G
        """
        self._sim_chan_only = sim_h
        if sim_h:
            self.input_dim = self.N
        elif self._use_chan:
            self.input_dim = self.N * self.T + self.N
        else:
            self.input_dim = self.N * self.T
        self.input_dim = self.x_dim

    def get_base_X_h(self, n_samples=1000, reuse_GQ=False):
        """Generate values for G and X

        Args:
            n_samples (int, optional): Number of input samples. Defaults to 1000.

        Returns:
            numpy: X, X_T, X_cond, G
        """
        # self.use_chan_in_sim(h_flag=True)
        self._use_chan = True
        vals = self._sim(n_samples, reuse_GQ)
        X = vals[:, 0 : self.N * self.T]
        X_T = X[:, -self.N :]
        X_cond = X[:, 0 : -self.N]
        return X, X_T, X_cond, self.h

    def set_dim_hxc(self):
        self._use_chan = True
        self.input_dim = self.N * (self.T - 1) + self.N

    def set_dim_xxc(self):
        self._use_chan = False
        self.input_dim = self.N * self.T

    def set_dim_cond(self):
        self._use_chan = False
        self.input_dim = self.N * (self.T - 1)

    def set_dim_joint(self):
        self._use_chan = True
        self.input_dim = self.N * self.T + self.N

    def set_T(self, T):
        ### TODO: update class to change how many transmissions are outputted when calling sim().
        # Potentially look at re-using samples from sim
        self.T = T

        self.input_dim = self.T * self.N
        self.sim_S = mvn(rho=0.0, dim_x=self.sym_N * self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.noise_N * self.T)


class CPDSSS_Cond(CPDSSS):
    def __init__(self, num_tx, N, L=None, d0=None, d1=None, use_fading=True):
        super().__init__(num_tx, N, L, d0, d1, use_fading)

    def sim(self, n_samples=1000, reuse_GQ=True):
        assert self.input_dim[1] != -1, "Input_dim[1] has not been set, run set_Xcond or set_XHcond"
        # if not reuse_GQ or self._X is None:

        vals = self._sim(n_samples, reuse_GQ)
        X = vals[:, 0 : self.N * self.T]
        XT = X[:, -self.N :]
        Xcond = X[:, 0 : -self.N]

        if self._sym_type == "XT|Xold":
            return [XT, Xcond]
        elif self._sym_type == "XT|Xold,h":
            cond = (
                np.concatenate((Xcond, self.h[:n_samples]), axis=1)
                if self.input_dim[0] > 0
                else np.zeros((n_samples, 1))
            )
            return [XT, cond]
        elif self._sym_type == "h|XT,Xold":
            return [self.h[:n_samples], X]
        else:
            raise ValueError("Invalid sym type, use model.set**() to set sym type")

        # if self.input_dim[1] == self.N * self.T:  # X,H are conditionals
        #     cond = np.concatenate((Xcond, self.h[:n_samples]), axis=1)
        # elif self.input_dim[1] == 0:
        #     cond = np.zeros((n_samples, 1))
        # else:  # X is the conditional
        #     cond = Xcond
        # return [XT, cond]

    def set_Xcond(self):
        self.input_dim[1] = self.N * (self.T - 1)
        self.update_x_dim()
        self._sym_type = "XT|Xold"

    def set_XHcond(self):
        self.input_dim[1] = self.N * self.T
        self.update_x_dim()
        self._sym_type = "XT|Xold,h"

    def set_H_given_X(self):
        self.input_dim = [self.N, self.N * self.T]
        self._sym_type = "h|XT,Xold"

    def set_T(self, T):
        ### TODO: update class to change how many transmissions are outputted when calling sim().
        # Potentially look at re-using samples from sim
        self.T = T

        # self.x_dim = self.N * self.T + self.N
        self.input_dim = [self.N, -1]
        self.sim_S = mvn(rho=0.0, dim_x=self.sym_N * self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.noise_N * self.T)


class CPDSSS_Cond_ZC(CPDSSS_Cond):
    """Modification of CPDSSS to include zadoff chu sequence for spreading.
    Thus the output is complex. For simulation, it may be thought of as dimension = N*2
    x=GZ*s+Qv  where GZ = G*Z*E and E is a decimator matrix [:,range(0,N,N/L)]

    Args:
        CPDSSS_Cond (_type_): _description_
    """

    def __init__(self, num_tx, N, L=None, d0=None, d1=None, use_fading=True):
        super().__init__(num_tx, N, L, d0, d1, use_fading)
        self.N = N * 2  # double N due to complex

        self.sim_S = mvn_complex(0, self.sym_N * self.T)
        self.sim_V = mvn_complex(0, self.noise_N * self.T)
        self.sim_H = mvn_complex(0, N)

        # self.sim_H.sim(1000)

        # get first integer that is prime with respect to N, ie gcd(N,u)=1
        # try not not have u=1
        u = np.argmax(np.gcd(self.N, np.arange(2, self.N)) == 1) + 2 if self.N > 2 else 1
        z = zadoff_chu(N, u).astype(np.complex64)
        self.Z = lin.toeplitz(z, np.concatenate(([z[0]], z[-1:0:-1])))

    def _gen_batch_GQ_sample(self, h):
        G, Q = super()._gen_batch_GQ_sample(h)
        # If circular, then matrix multiplicaiton is communitive
        # GZ = G*Z*E = Z*G*E   where E is a decimating matrix, ie [:,g_slice]
        return self.Z @ G, Q

    def sim(self, n_samples=1000, reuse_GQ=True):
        samples = super().sim(n_samples, reuse_GQ)

        samples = [util.misc.split_complex_to_real(x) for x in samples]
        return samples

    def chan_entropy(self):
        return np.log(np.linalg.det(np.pi * np.exp(1) * np.diag(self.fading)))

    def set_T(self, T):
        self.T = T

        # self.x_dim = self.N * self.T + self.N
        self.input_dim = [self.N, -1]
        self.sim_S = mvn_complex(rho=0.0, dim_x=self.sym_N * self.T)
        self.sim_V = mvn_complex(rho=0.0, dim_x=self.noise_N * self.T)


class CPDSSS_Gram_Schmidt(CPDSSS_Cond):
    def __init__(self, num_tx, N, L=None, d0=None, d1=None):
        super().__init__(num_tx, N, L, d0, d1)

    def sim(self, n_samples=1000):
        assert self.input_dim[1] != -1, "Input_dim[1] has not been set, run set_Xcond or set_XHcond"
        s = (
            self.sim_S.sim(n_samples=n_samples)
            .astype(dtype)
            .reshape((n_samples, self.sym_N, self.T))
        )

        v = (
            self.sim_V.sim(n_samples=n_samples)
            .astype(dtype)
            .reshape((n_samples, self.noise_N, self.T))
        )
        self.H = np.random.standard_normal((n_samples, self.N, self.N))
        self.h = np.reshape(self.H, (n_samples, self.N * self.N), order="F")
        self.sim_GQ()

        if self.sym_N == self.N:
            X = np.matmul(self.G[:n_samples, :, :], s)
        else:
            X = np.matmul(self.G[:n_samples, :, :], s) + np.matmul(self.Q[:n_samples, :, :], v)
        del s, v
        gc.collect()

        # separate X into T-th transmission and previous transmissions
        previous_X_T = X[:, :, : self.T - 1].reshape((n_samples, self.N * (self.T - 1)), order="F")
        X_T = X[:, :, -1]
        gc.collect()

        if self._sym_type == "XT|Xold,h":  # X,H are conditionals
            cond = np.concatenate((previous_X_T, self.h[:n_samples]), axis=1)
            return [X_T, cond]
        elif self.input_dim[1] == 0:
            cond = np.zeros((n_samples, 1))
            return [X_T, cond]
        elif self._sym_type == "XT|Xold":  # X is the conditional
            cond = previous_X_T
            return [X_T, cond]
        elif self._sym_type == "h|X":
            return [
                self.h[:n_samples],
                X[:, :, : self.T].reshape((n_samples, self.N * self.T), order="F"),
            ]
        return [X_T, cond]

    def sim_GQ(self):
        # generate G and Q using Gram-Schmidt process
        # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

        def project(v, u):
            return np.dot(u, v) / np.dot(u, u) * u

        def gram_schmidt(vectors, normalize=True):
            orthogonal_vectors = []
            for v in vectors:
                for u in orthogonal_vectors:
                    v -= project(v, u)
                orthogonal_vectors.append(v)

            orthogonal_vectors = np.array(orthogonal_vectors)
            if normalize:
                return orthogonal_vectors / np.linalg.norm(orthogonal_vectors, axis=1)[:, None]
            else:
                return orthogonal_vectors

        n_samples = self.H.shape[0]

        self.G = np.empty((n_samples, self.N, self.sym_N), dtype=dtype)
        self.Q = np.empty((n_samples, self.N, self.noise_N), dtype=dtype)

        start = time.time()
        for i in range(n_samples):
            if i % 10000 == 0:
                util.misc.printProgressBar(i, n_samples, "G,Q generation")
            vect = gram_schmidt(self.H[i])
            self.G[i] = vect[:, : self.sym_N]
            self.Q[i] = (
                vect[:, -self.noise_N :] if self.noise_N > 0 else np.empty((self.N, 0), dtype=dtype)
            )
        util.misc.printProgressBar(n_samples, n_samples, "G,Q generation")
        print(f"G,Q time {str(timedelta(seconds=int(time.time() - start)))}")

    def set_Xcond(self):
        self.input_dim[1] = self.N * (self.T - 1)
        self.update_x_dim()
        self._sym_type = "XT|Xold"

    def set_XHcond(self):
        self.input_dim[1] = self.N * (self.T - 1) + self.N * self.N
        self.update_x_dim()
        self._sym_type = "XT|Xold,h"

    def set_H_given_X(self):
        self.input_dim = [self.N * self.N, self.N * self.T]
        self._sym_type = "h|X"

    def chan_entropy(self):
        return self.N * self.N / 2 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(1)


class CPDSSS_Hinv(CPDSSS):
    def __init__(self, num_tx, N, L=None, d0=None, d1=None):
        super().__init__(num_tx, N, L, d0, d1)

    def _gen_GQ_sample(self, h):

        # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
        # A=E.T @ H
        # Slightly faster than doing E.T @ H
        H = lin.toeplitz(h, np.concatenate(([h[0]], h[-1:0:-1])))
        Hinv = np.linalg.inv(H + self.eye)

        G = Hinv[:, : self.sym_N]
        Q = Hinv[:, -self.noise_N :]

        return G, Q


class MIMO_Gaussian(_distribution):
    def __init__(self, N=1, sigma_A=1, sigma_x=1, sigma_n=1, rng=np.random.default_rng()):
        super().__init__(x_dim=N)
        self.N = N
        self.sigma_A = sigma_A
        self.sigma_x = sigma_x
        self.sigma_n = sigma_n
        self.rng = rng
        self.T = -1
        self._sym_type = "n/a"

    def sim(self, n_samples=1000):
        assert self._sym_type != "n/a", "Set the _sym_type before calling sim() using set_input*()"
        assert self.T != -1, "Set the T before calling sim() using set_T()"

        self.A = self.rng.normal(0, np.sqrt(self.sigma_A), (n_samples, self.N, self.N)).astype(
            dtype
        )
        if self._sym_type == "A":
            return self.A.reshape(n_samples, self.N * self.N, order="F")

        self.X = self.rng.normal(0, np.sqrt(self.sigma_x), (n_samples, self.N, self.T)).astype(
            dtype
        )
        n = self.rng.normal(0, np.sqrt(self.sigma_n), (n_samples, self.N, self.T)).astype(dtype)
        Y = np.matmul(self.A, self.X) + n
        Y = Y.reshape(n_samples, self.N * self.T, order="F")
        X = self.X.reshape(n_samples, self.N * self.T, order="F")
        A = self.A.reshape(n_samples, self.N * self.N, order="F")

        if self._sym_type == "Y":
            return Y
        elif self._sym_type == "YX":
            return np.concatenate((Y, X.reshape(n_samples, self.N**2, order="F")), axis=1)
        elif self._sym_type == "X":
            return X
        elif self._sym_type == "A":
            return A
        elif self._sym_type == "YA":
            return np.concatenate((Y, A), axis=1)
        elif self._sym_type == "Y|A":
            return [Y, A]
        elif self._sym_type == "Y|X":
            return [Y, X]
        else:
            raise ValueError("Invalid sym type, use model.set**() to set sym type")

    def entropy(self):
        if self._sym_type == "X":
            return self.N * self.T / 2 * np.log(2 * np.pi * np.exp(1)) + self.T / 2 * np.log(
                self.sigma_x
            )
        if self._sym_type == "A":
            return self.N * self.N / 2 * np.log(2 * np.pi * np.exp(1)) + self.N / 2 * np.log(
                self.sigma_A
            )
        if self._sym_type == "Y|A":
            dets = np.sum(
                np.log(
                    np.linalg.eigvalsh(
                        self.sigma_n * np.eye(self.N)
                        + self.sigma_x * np.matmul(self.A, self.A.transpose(0, 2, 1))
                    )
                ),
                axis=1,
            )
            return self.N * self.T / 2 * np.log(2 * np.pi * np.exp(1)) + self.T / 2 * np.mean(dets)
        if self._sym_type == "Y|X":
            dets = np.sum(
                np.log(
                    np.linalg.eigvalsh(
                        self.sigma_n * np.eye(self.T)
                        + self.sigma_A * np.matmul(self.X.transpose(0, 2, 1), self.X)
                    )
                ),
                axis=1,
            )
            return self.N * self.T / 2 * np.log(2 * np.pi * np.exp(1)) + self.N / 2 * np.mean(dets)

        return np.nan

    def set_T(self, T):
        self.T = T

    def set_input_X(self):
        self.input_dim = self.N * self.T
        self._sym_type = "X"

    def set_input_YX(self):
        self.input_dim = self.N * self.T + self.N * self.N
        self._sym_type = "YX"

    def set_input_A(self):
        self.input_dim = self.N * self.N
        self._sym_type = "A"

    def set_input_YA(self):
        self.input_dim = self.N * self.T + self.N * self.N
        self._sym_type = "YA"

    def set_input_Y(self):
        self.input_dim = self.N * self.T
        self._sym_type = "Y"

    def set_input_Y_given_A(self):
        self.input_dim = [self.N * self.T, self.N * self.N]
        self._sym_type = "Y|A"

    def set_input_Y_given_X(self):
        self.input_dim = [self.N * self.T, self.N * self.T]
        self._sym_type = "Y|X"


class CPDSSS_XS(CPDSSS):
    """
    Generate the output samples X for CPDSSS
    """

    def __init__(self, N, L):
        CPDSSS.__init__(self, num_tx=1, N=N, L=L, use_gaussian_approx=False)
        # self.base_model=CPDSSS(num_tx=num_tx,N=N,L=L,use_gaussian_approx=False)

    def sim_use_X(self):
        self.use_X = True
        self.use_S = False
        self.x_dim = self.N

    def sim_use_S(self):
        self.use_X = True
        self.use_S = False
        self.x_dim = self.sym_N

    def sim_use_XS(self):
        self.use_X = True
        self.use_S = True
        self.x_dim = self.N + self.sym_N

    def sim(self, n_samples=1000):
        X = super().sim(n_samples=n_samples)
        ret_val = np.empty((n_samples, 0))
        if self.use_X:
            ret_val = np.concatenate((ret_val, X), axis=1)
        # if self.use_S:
        #     ret_val = np.concatenate((ret_val, self.s[:, :, 0]), axis=1)

        return ret_val


class CPDSSS_XG(CPDSSS):
    """
    Used to generate the joint samples between X and G for CPDSSS
    """

    def __init__(self, num_tx, N, L):
        super().__init__(num_tx, N, L)
        # self.base_model = CPDSSS(num_tx=num_tx, N=N, L=L)

    def sim(self, n_samples=1000):
        self.gen_XG(n_samples=n_samples)
        # order 'F' needed to make arrays stack instead of interlaced
        joint_X = self.X[:, :, 0 : self.T].reshape((n_samples, self.N * self.T), order="F")
        g_term = self.G[:, :, 0]
        return np.concatenate((joint_X, g_term), axis=1)


class Gaussian(_distribution):
    def __init__(self, mu, sigma2, N=1):
        if not np.isscalar(mu):
            N = mu.shape[0]
        super().__init__(x_dim=N)
        # make mu,sigma vectors if input is a scalar
        if np.isscalar(mu):
            mu = [mu] * N
            sigma2 = sigma2 * np.eye(N)
        self.mu = np.asarray(mu, dtype=dtype)
        self.sigma = np.asarray(sigma2, dtype=dtype)

    def sim(self, n_samples=1000):
        return self.rng.multivariate_normal(mean=self.mu, cov=self.sigma, size=n_samples).astype(
            dtype
        )

    def entropy(self):
        return 0.5 * np.log(lin.det(self.sigma)) + self.x_dim / 2 * (1 + np.log(2 * np.pi))


class Limited_Gaussian(Gaussian):
    def __init__(self, mu, sigma2, limit, N=1):
        super().__init__(mu, sigma2, N)
        self.limit = limit

    def sim(self, n_samples=1000):
        samples = super().sim(n_samples)
        mask = np.linalg.norm(samples, axis=1) <= self.limit
        return samples[mask, :]

    def entropy(self):
        # found through estimation. Unsure if this scales linearly with dimensions
        if self.x_dim == 3:
            return 4.110


class Exponential(_distribution):
    """Class to simulate an exponential distribution
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    def __init__(self, lamb):
        super.__init__(x_dim=1)
        self.lamb = lamb

    def sim(self, n_samples):
        return (
            np.random.default_rng()
            .exponential(scale=1 / self.lamb, size=(n_samples, 1))
            .astype(dtype)
        )

    def entropy(self):
        return 1 - np.log(self.lamb)


class Exponential_sum(_distribution):
    """Class to simulate the sum of two independent exponential variables
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    def __init__(self, lambda1, lambda2):
        super().__init__(x_dim=1)
        assert lambda1 > lambda2, "lambda1 > lambda2 for this entropy test"
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.expo1 = Exponential(lambda1)
        self.expo2 = Exponential(lambda2)

    def sim(self, n_samples):
        return self.expo1.sim(n_samples) + self.expo2.sim(n_samples)

    def entropy(self):
        return (
            1
            + np.euler_gamma
            + np.log((self.lambda1 - self.lambda2) / (self.lambda1 * self.lambda2))
            + spec.digamma(self.lambda1 / (self.lambda1 - self.lambda2))
        )


class Laplace(_distribution):
    """Simulate the laplace distribution
    https://en.wikipedia.org/wiki/Laplace_distribution"""

    def __init__(self, mu, b, N=1):
        super().__init__(x_dim=N)
        self.mu = np.asarray(mu, dtype=dtype)
        self.b = np.asarray(b, dtype=dtype)

    def sim(self, n_samples):
        return (
            np.random.default_rng().laplace(self.mu, self.b, (n_samples, self.x_dim)).astype(dtype)
        )

    def entropy(self):
        return (1 + np.log(2 * self.b)) * self.x_dim

    def pdf(self, x):
        return 1 / (2 * self.b) * np.exp(-np.abs(x - self.mu) / self.b)

    def logpdf(self, x):
        return -self.x_dim * np.log(2 * self.b) - tt.sum(tt.abs_(x - self.mu) / self.b, axis=1)


class Cauchy(_distribution):
    def __init__(self, mu, gamma, N=1):
        super().__init__(x_dim=N)
        self.mu = mu
        self.gamma = gamma

    def sim(self, n_samples):
        return stats.cauchy.rvs(loc=self.mu, scale=self.gamma, size=(n_samples, self.x_dim)).astype(
            dtype
        )

    def entropy(self):
        return np.log(4 * np.pi * self.gamma) * self.x_dim


class Logistic(_distribution):
    def __init__(self, mu, s, N=1):
        super().__init__(x_dim=N)
        self.s = s
        self.mu = mu

    def sim(self, n_samples):
        return stats.logistic.rvs(loc=self.mu, scale=self.s, size=(n_samples, self.x_dim)).astype(
            dtype
        )

    def entropy(self):
        return (np.log(self.s) + 2) * self.x_dim
