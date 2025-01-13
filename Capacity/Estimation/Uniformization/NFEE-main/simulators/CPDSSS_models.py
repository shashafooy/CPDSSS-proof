import abc
import numpy as np
import scipy.linalg as lin
import scipy.special as spec
import scipy.stats as stats
from simulators.complex import mvn
import math
from joblib import Parallel, delayed
import theano.tensor as tt
import theano

dtype = theano.config.floatX
USE_GPU = True


# from memory_profiler import profile


class _distribution:
    __metaclass__ = abc.ABCMeta

    def __init__(self, x_dim=0, rng=np.random):
        self.x_dim = x_dim
        self.rng = rng.default_rng()

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
    def __init__(self, num_tx, N, L=None, d0=None, d1=None):
        super().__init__(x_dim=N * num_tx + N)

        self.T = num_tx
        self.N = N
        self.L = L
        self.input_dim = self.T * self.N
        if L is not None:
            assert N / L % 1 == 0, "N/L must be an integer"
            self.sym_N = int(N / L)
            self.noise_N = N - self.sym_N
            self.G_slice = range(0, N, L)
            if USE_GPU:
                self.tt_G_slice = theano.shared(np.arange(0, N, L, dtype=np.int32))
        elif d0 is not None and d1 is not None:
            assert d0 + d1 == N, "d0+d1 must be equal to N"
            self.sym_N = d0
            self.noise_N = d1
            if N / d0 % 1 == 0:  # same as using L
                self.G_slice = range(0, N, int(N / d0))
                if USE_GPU:
                    self.tt_G_slice = theano.shared(np.arange(0, N, int(N / d0), dtype=np.int32))
            else:
                self.G_slice = range(0, d0)
                if USE_GPU:
                    self.tt_G_slice = theano.shared(np.arange(0, d0, dtype=np.int32))
        else:
            raise ValueError(
                "Invalid input. Require either L or d0,d1 as inputs. N/L must be an integer, or d0+d1=N"
            )

        self.sim_S = mvn(rho=0.0, dim_x=self.sym_N * self.T)
        self.sim_V = mvn(rho=0.0, dim_x=self.noise_N * self.T)
        self.sim_H = mvn(rho=0.0, dim_x=self.N)

        self.use_chan_in_sim()
        self._sim_chan_only = False

        self.fading = np.exp(-np.arange(self.N) / 3).astype(dtype)

        # For larger N, the regularization factor may need to increase so H^T*H is not singular
        if self.N > 10:
            self.eye = 0.0005 * np.eye(self.N).astype(dtype)
            self.tt_eye = 0.0005 * tt.eye(self.N, dtype=dtype)
        else:
            self.eye = 0.0001 * np.eye(self.N).astype(dtype)
            self.tt_eye = 0.0001 * tt.eye(self.N, dtype=dtype)

        self.tt_GQ_func = None

    def sim(self, n_samples=1000):
        """wrapper allowing inherited class to reuse while _sim() retains the core functionality

        Args:
            n_samples (int, optional): number of samples to generate. Defaults to 1000.
        """
        return self._sim(n_samples)

    def _sim(self, n_samples=1000):
        """Generate samples X and (optional) G for CPDSSS

        Args:
            n_samples (int, optional): number of input samples. Defaults to 1000.

        Returns:
            numpy: (n_samples, dim_x) array of generated values
        """

        self.s = (
            self.sim_S.sim(n_samples=n_samples)
            .astype(dtype)
            .reshape((n_samples, self.sym_N, self.T))
        )

        v = (
            self.sim_V.sim(n_samples=n_samples)
            .astype(dtype)
            .reshape((n_samples, self.noise_N, self.T))
        )
        self.h = (self.sim_H.sim(n_samples=n_samples) * np.sqrt(self.fading)).astype(dtype)

        if self._sim_chan_only:  # return early if we only need channel
            self.samples = self.h
            return self.h

        self.G, Q = self.sim_GQ()
        # import timeit
        # timeit.timeit(lambda: self.sim_GQ(n_samples=200000),number=1)
        if self.sym_N == self.N:
            self.X = np.matmul(self.G, self.s)
        else:
            self.X = np.matmul(self.G, self.s) + np.matmul(Q, v)
        joint_X = self.X[:, :, : self.T].reshape(
            (n_samples, self.N * self.T), order="F"
        )  # order 'F' needed to make arrays stack instead of interlaced

        if self._use_chan:
            self.samples = np.concatenate((joint_X, self.h), axis=1)
        else:
            self.samples = joint_X

        return self.samples
        # g_term = G[:,:,0]
        # xT_term = X[:,:,self.T-1]
        # xCond_term = X[:,:,0:self.T-1].reshape((n_samples,self.N*(self.T-1)),order='F')#order 'F' needed to make arrays stack instead of interlaced

    # @profile
    def sim_GQ(self):
        import multiprocessing as mp
        import time

        n_samples = self.h.shape[0]
        workers = min(8, mp.cpu_count() - 1)  # 6 seems to be optimal due to threading overhead

        if USE_GPU:
            if self.tt_GQ_func is None:
                self.tt_GQ_func = self._gen_tt_GQ_func()

            start = time.time()

            # G1, Q1 = self._gen_GQ_sample(self.h[100, :])

            G, Q = self.tt_GQ_func(self.h)
            print(f"one shot GPU {time.time() - start:.4f}")

            return G, Q
        else:

            start = time.time()
            with mp.Pool(workers) as pool:
                G_results, Q_results = zip(
                    *pool.map(self._gen_GQ_sample, (self.h[i, :] for i in range(n_samples)))
                )

            # results = Parallel(n_jobs=-1, backend="threading")(delayed(self._gen_GQ_sample)(self.h[i,:]) for i in range(n_samples))

            # print(f"pool time: {time.time() - start_time}")

            G = np.array(G_results)
            Q = np.array(Q_results)
            print(f"CPU thread workers = {workers}, time = {time.time() - start:.4f}")

            return G, Q

    def _gen_GQ_sample(self, h):

        # H = lin.toeplitz(h[i,:],np.concatenate(([h[i,0]],h[i,-1:0:-1])))
        # A=E.T @ H
        # Slightly faster than doing E.T @ H
        HE = lin.toeplitz(h, np.concatenate(([h[0]], h[-1:0:-1])))[self.G_slice, :]
        R = HE.T @ HE + self.eye
        p = HE[0, :].T
        # g = lin.inv(R) @ p
        g = np.linalg.solve(R, p)  # faster than inverse
        # Only take every L columns of toepltiz matrix
        # Slightly faster than doing G@E
        G = lin.toeplitz(g, np.concatenate(([g[0]], g[-1:0:-1])))[:, self.G_slice]
        # Potentially better G, fullfills G'*Q=0 and HE * G = I
        # G = lin.inv(R) @ HE[self.G_slice,:].T

        # ev, V = lin.eig(R)
        ev, V = np.linalg.eigh(R)  # R is symmetric, eigh is optimized for symmetric
        # Only use eigenvectors associated with "zero" eigenvalue
        #  get indices of the smallest eigenvalues to find "zero" EV
        sort_indices = np.argsort(ev)
        # Sometimes get a very small imaginary value, ignore it
        Q = np.real(V[:, sort_indices[0 : self.noise_N]])

        return G, Q

    def _gen_tt_GQ_func(self):
        # Define symbolic input
        h = tt.matrix("h")

        # Create toeplitz matrix and decimate across its rows
        HE = theano.scan(fn=self.tt_toeplitz_row_slice, sequences=[h], non_sequences=[])[0]

        # Compute R, p, and g
        R = tt.batched_dot(HE.transpose(0, 2, 1), HE) + self.tt_eye
        p = HE[:, 0, :]

        # Solve g = inv(R) * p
        # Make toeplitz matrix and decimate across its columns
        def make_g(R, p):
            g = tt.slinalg.solve(R, p)
            # g = tt.dot(tt.nlinalg.matrix_inverse(R), p)
            return self.tt_toeplitz_col_slice(g)

        G = theano.scan(fn=make_g, sequences=[R, p], non_sequences=[])[0]

        # return theano.function(inputs=[h], outputs=G)

        # Compute eigenvalues and eigenvectors of R
        def compute_ev(r):
            ev, V = tt.nlinalg.eigh(r)
            # ev, V = tt.nlinalg.eig(r)
            return ev, V

        ev_b, V_b = theano.scan(fn=compute_ev, sequences=[R])[0]

        # Get the eigenvectors associated with the smallest eigenvalues
        def make_Q(ev, V):
            sort_indices = tt.argsort(ev)
            sort_indices = sort_indices[: self.noise_N]
            return V[:, sort_indices]

        Q = theano.scan(fn=make_Q, sequences=[ev_b, V_b])[0]

        # Define Theano function
        return theano.function(inputs=[h], outputs=[G, Q], allow_input_downcast=True)

    def tt_toeplitz_col_slice(self, col, row=None):
        """Generate a toeplitz matrix and slice it such that toeplitz[:,G_slice]"""

        row = col[::-1]  # reverse column vector
        # Construct indices for the Toeplitz matrix
        col_idx = tt.arange(col.shape[0]).dimshuffle(0, "x")  # shape (1, len(row))
        row_idx = self.tt_G_slice.dimshuffle("x", 0)  # shape: (len(G_slice),1)
        indices = col_idx - row_idx
        # indices = tt.arange(col.shape[0])[:, None] - tt.arange(row.shape[0])[None, :]

        # Use T.switch to fill values based on indices
        toeplitz_matrix = tt.switch(indices >= 0, col[indices], row[-indices - 1])

        return toeplitz_matrix

    def tt_toeplitz_row_slice(self, col):
        """Generate a toeplitz matrix and slice it such that toeplitz[G_slice,:]"""
        row = col[::-1]  # reverse column vector
        # Construct indices for the Toeplitz matrix
        col_idx = self.tt_G_slice.dimshuffle(0, "x")  # shape: (len(G_slice),1)
        row_idx = tt.arange(row.shape[0]).dimshuffle("x", 0)  # shape (1, len(row))
        indices = col_idx - row_idx
        # indices = tt.arange(col.shape[0])[:, None] - tt.arange(row.shape[0])[None, :]

        # Use T.switch to fill values based on indices
        toeplitz_matrix = tt.switch(indices >= 0, col[indices], row[-indices - 1])

        return toeplitz_matrix

    def chan_entropy(self):
        return 0.5 * np.log(np.linalg.det(2 * math.pi * np.exp(1) * np.diag(self.fading)))

    def use_chan_in_sim(self, h_flag=True):
        """Enables sim to append G in addition to X

        Args:
            g_flag (bool, optional): Enable G output. Defaults to True.
        """
        self._use_chan = h_flag
        if h_flag:
            self.x_dim = self.N * self.T + self.N
        else:
            self.x_dim = self.N * self.T
        self.input_dim = self.x_dim

    def only_sim_chan(self, sim_h=True):
        """Set flag to only return G when using sim()

        Args:
            sim_g (bool): True if you want sim() to return only G
        """
        self._sim_chan_only = sim_h
        if sim_h:
            self.x_dim = self.N
        elif self._use_chan:
            self.x_dim = self.N * self.T + self.N
        else:
            self.x_dim = self.N * self.T
        self.input_dim = self.x_dim

    def get_base_X_h(self, n_samples=1000):
        """Generate values for G and X

        Args:
            n_samples (int, optional): Number of input samples. Defaults to 1000.

        Returns:
            numpy: X, X_T, X_cond, G
        """
        # self.use_chan_in_sim(h_flag=True)
        self._use_chan = True
        vals = self._sim(n_samples=n_samples)
        X = vals[:, 0 : self.N * self.T]
        X_T = X[:, -self.N :]
        X_cond = X[:, 0 : -self.N]
        return X, X_T, X_cond, self.h

    def set_dim_hxc(self):
        self._use_chan = True
        self.x_dim = self.N * (self.T - 1) + self.N
        self.input_dim = self.x_dim

    def set_dim_xxc(self):
        self._use_chan = False
        self.x_dim = self.N * self.T
        self.input_dim = self.x_dim

    def set_dim_cond(self):
        self._use_chan = False
        self.x_dim = self.N * (self.T - 1)
        self.input_dim = self.x_dim

    def set_dim_joint(self):
        self._use_chan = True
        self.x_dim = self.N * self.T + self.N
        self.input_dim = self.x_dim

    def set_T(self,T):



class CPDSSS_Cond(CPDSSS):
    def __init__(self, num_tx, N, L=None, d0=None, d1=None):
        super().__init__(num_tx, N, L, d0, d1)
        self.x_dim = N
        self.input_dim = [N, None]
        self.sim_val = None
        self._X = None

    def sim(self, n_samples=1000, reuse=False):
        assert self.input_dim[1] is not None
        if not reuse or self._X is None:
            self._X, self._XT, self._Xcond, self._h = super().get_base_X_h(n_samples)
        if self.input_dim[1] == self.N * self.T:  # X,H are conditionals
            cond = np.concatenate((self._Xcond, self._h), axis=1)
        else:  # X is the conditional
            cond = self._Xcond
        return [self._XT, cond]

    def set_Xcond(self):
        self.input_dim[1] = self.N * (self.T - 1)
        self.x_dim = self.input_dim[1] + self.N

    def set_XHcond(self):
        self.input_dim[1] = self.N * self.T
        self.x_dim = self.input_dim[1] + self.N


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
        if self.use_S:
            ret_val = np.concatenate((ret_val, self.s[:, :, 0]), axis=1)

        return ret_val


class CPDSSS_XG:
    """
    Used to generate the joint samples between X and G for CPDSSS
    """

    def __init__(self, num_tx, N, L):
        self.base_model = CPDSSS(num_tx=num_tx, N=N, L=L)

    def sim(self, n_samples=1000):
        self.base_model.gen_XG(n_samples=n_samples)
        T = self.base_model.T
        N = self.base_model.N
        # order 'F' needed to make arrays stack instead of interlaced
        joint_X = self.base_model.X[:, :, 0:T].reshape((n_samples, N * T), order="F")
        g_term = self.base_model.G[:, :, 0]
        return np.concatenate((joint_X, g_term), axis=1)


class Gaussian(_distribution):
    def __init__(self, mu, sigma2, N=1):
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
