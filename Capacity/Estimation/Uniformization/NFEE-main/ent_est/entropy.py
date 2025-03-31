import numpy as np

# from sklearn.neighbors import NearestNeighbors as cpuNN
from scipy import stats
import scipy.special as spl

import sys
import configparser

import ml.trainers as trainers
import ml.step_strategies as ss

import misc_CPDSSS.util as utils

from theano import config

dtype = config.floatX


config = configparser.ConfigParser()
config.read("CPDSSS.ini")
n_jobs = config["GLOBAL"].getint("knn_cores", 1)
KNN_GPU = config["GLOBAL"].getboolean("knn_GPU", False)


if KNN_GPU:
    from cuml.neighbors import NearestNeighbors  # type: ignore

    algorithm = "auto"
else:
    from sklearn.neighbors import NearestNeighbors

    algorithm = "kd_tree"


def tkl_tksg(y, n=None, k=1, max_k=None, shuffle=True, rng=np.random):
    """Evaluate truncated knn KL and KSG algorithms
    Ziqiao Ao and Jinglai Li. “Entropy estimation via uniformization”

    Args:
        y (_type_): data points to find the distances of
        n (_type_, optional): number of samples. Defaults to None.
        k (int,list, optional): number of neighbors to find. Entropy for multiple values may be found if k is a list. Defaults to 1.
        max_k (int, optional): maximum k value if K is a range. Defaults to None.
        shuffle (bool, optional): True to shuffle the data samples. Defaults to True.
        rng (_type_, optional): type of rng generator. Defaults to np.random.

    Returns:
        _type_: entropy estimate, return list if k is a list
    """

    y = np.asarray(y, dtype)
    N, dim = y.shape
    if isinstance(k, list):
        max_k = k[-1]
        k_range = range(1, max_k + 1)
    else:
        max_k = k
        k_range = range(k, k + 1)
    # k=k if max_k is None else max_k

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # knn search
    nbrs = NearestNeighbors(
        n_neighbors=max_k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)
    dist, idx = nbrs.kneighbors(y)

    # k_range = range(1,max_k+1) if max_k is not None else range(k,k+1)

    h_kl = np.zeros(len(k_range))
    h_ksg = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        # truncated KL
        N_split = np.ceil(y.size / 10e6)
        hh_mean = []
        n_size = []
        for section in np.array_split(range(N), N_split):
            zeros_mask = dist[section, k] != 0
            yy = y[section]
            r = dist[section, k]  # radius
            r = np.tile(r[:, np.newaxis], (1, dim))
            lb = (yy - r >= 0) * (yy - r) + (yy - r < 0) * 0
            ub = (yy + r <= 1) * (yy + r) + (yy + r > 1) * 1

            zeta = (ub - lb)[zeros_mask]  # remove zeros, duplicate points result in 0 distance
            # N = zeta.shape[0]
            hh = np.sum(np.log(zeta), axis=1)
            n_size.append(hh.shape[0])
            hh_mean.append(np.mean(hh))

        h_kl[i] = -spl.digamma(k) + spl.digamma(sum(n_size)) + utils.combine_means(hh_mean, n_size)

        # truncated KSG
        hh_mean = []
        n_size = []
        for section in np.array_split(range(N), N_split):
            y_dup = np.tile(
                y[section, np.newaxis, :], (1, k, 1)
            )  # duplicate last axis to add k dimension
            epsilons = np.max(np.abs(y_dup - y[idx[section, 1 : k + 1]]), axis=1)
            zeta2 = np.minimum(y[section] + epsilons, 1) - np.maximum(y[section] - epsilons, 0)
            # remove zeros, invalid data. Zeros occur if points along a dimension are exactly the same
            zeros_mask = ~np.any(zeta2 == 0, axis=1)
            zeta2 = zeta2[zeros_mask]
            hh2 = np.sum(np.log(zeta2), axis=1)
            # N = zeta2.shape[0]
            n_size.append(hh2.shape[0])
            hh_mean.append(np.mean(hh2))

        h_ksg[i] = (
            -spl.digamma(k)
            + spl.digamma(sum(n_size))
            + (dim - 1) / k
            + utils.combine_means(hh_mean, n_size)
        )

    # If we only used 1 k value, return that value, not an array
    (h_kl, h_ksg) = (h_kl, h_ksg) if len(k_range) > 1 else (h_kl[0], h_ksg[0])

    return h_kl, h_ksg


def kl_ksg(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    """Evaluate knn KL and KSG algorithms

    Args:
        y (_type_): data points to find the distances of
        n (_type_, optional): number of samples. Defaults to None.
        k (int,list, optional): number of neighbors to find. May also be a list to find multiple neighborsDefaults to 1.
        shuffle (bool, optional): True to shuffle the data samples. Defaults to True.
        rng (_type_, optional): type of rng generator. Defaults to np.random.

    Returns:
        _type_: entropy estimate, returns list of estimates in K is a list
    """
    y = np.asarray(y, dtype)

    if isinstance(k, list):
        max_k = k[-1]
        k_range = range(1, max_k + 1)
    else:
        max_k = k
        k_range = range(k, k + 1)

    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y / y_std

    N, dim = y.shape

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # knn search
    # print("starting distance search")
    nbrs = NearestNeighbors(
        n_neighbors=max_k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)
    dist, idx = nbrs.kneighbors(y)

    h_kl = np.empty(len(k_range))
    h_ksg = np.empty(len(k_range))
    for i, k in enumerate(k_range):
        # KL
        zeros_mask = dist[:, k] != 0
        dist = dist[zeros_mask, :]

        N_split = np.ceil(y.size / 10e6)
        hh_mean = []
        n_size = []
        for section in np.array_split(range(N), N_split):
            if standardize == True:
                hh = dim * np.log(2 * dist[section, k]) + np.sum(np.log(y_std))
            else:
                hh = dim * np.log(2 * dist[section, k])
            hh_mean.append(np.mean(hh))
            n_size.append(len(hh))
        # N = hh.shape[0]
        h_kl[i] = -spl.digamma(k) + spl.digamma(sum(n_size)) + utils.combine_means(hh_mean, n_size)

        # KSG
        # epsilons=np.abs(y-y[idx[:,k]])
        hh_mean = []
        n_size = []
        for section in np.array_split(range(N), N_split):
            y_dup = np.tile(
                y[section, np.newaxis, :], (1, k, 1)
            )  # duplicate last axis to add k dimension
            epsilons = np.max(np.abs(y_dup - y[idx[section, 1 : k + 1]]), axis=1)
            zeros_mask = ~np.any(epsilons == 0, axis=1)
            epsilons = epsilons[zeros_mask]
            if standardize == True:
                hh = np.sum(np.log(2 * epsilons * y_std), axis=1)
            else:
                hh = np.sum(np.log(2 * epsilons), axis=1)
            hh_mean.append(np.mean(hh))
            n_size.append(len(hh))
        # N = hh.shape[0]
        h_ksg[i] = (
            -spl.digamma(k)
            + spl.digamma(sum(n_size))
            + (dim - 1) / k
            + utils.combine_means(hh_mean, n_size)
        )

    h_kl, h_ksg = (h_kl, h_ksg) if len(k_range) > 1 else (h_kl[0], h_ksg[0])

    # print("finished knn")
    return h_kl, h_ksg


def kl(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    """Evaluate knn KL algorithms

    Args:
        y (_type_): data points to find the distances of
        n (_type_, optional): number of samples. Defaults to None.
        k (int, optional): number of neighbors to find. Defaults to 1.
        shuffle (bool, optional): True to shuffle the data samples. Defaults to True.
        rng (_type_, optional): type of rng generator. Defaults to np.random.

    Returns:
        _type_: entropy estimate
    """

    y = np.asarray(y, dtype)

    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y / y_std

    N, dim = y.shape

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # knn search
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)
    dist, idx = nbrs.kneighbors(y)
    zeros_mask = dist[:, k] != 0
    dist = dist[zeros_mask, :]
    N = dist.shape[0]

    N_split = np.ceil(y.size / 10e6)
    hh_mean = []
    n_size = []
    for section in np.array_split(range(N), N_split):
        if standardize == True:
            hh = dim * np.log(2 * dist[section, k]) + np.sum(np.log(y_std))
        else:
            hh = dim * np.log(2 * dist[section, k])
        hh_mean.append(np.mean(hh))
        n_size.append(len(hh))
    # N = hh.shape[0]
    h = -spl.digamma(k) + spl.digamma(sum(n_size)) + utils.combine_means(hh_mean, n_size)

    # if standardize == True:
    #     hh = dim * np.log(2 * dist[:, k]) + np.sum(np.log(y_std))
    # else:
    #     hh = dim * np.log(2 * dist[:, k])

    # h = -spl.digamma(k) + spl.digamma(N) + np.mean(hh)

    return h


def tkl(y, n=None, k=1, shuffle=True, rng=np.random):
    """Evaluate truncated knn KL and KSG algorithms
    Ziqiao Ao and Jinglai Li. “Entropy estimation via uniformization”

    Args:
        y (_type_): data points to find the knn distances of
        n (_type_, optional): number of samples. Defaults to None.
        k (int, optional): number of neighbors to find. Defaults to 1.
        shuffle (bool, optional): True to shuffle the data samples. Defaults to True.
        rng (_type_, optional): type of rng generator. Defaults to np.random.

    Returns:
        _type_: entropy estimate
    """

    y = np.asarray(y, dtype)
    N, dim = y.shape

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # Auto algorithm switches to brute after dim=16. For truncated, better to swap at dim=30
    # algorithm = "brute" if dim > 30 else "kd_tree"
    # algorithm = (
    #     "kd_tree"  # should almost always be faster for the uniform truncated case (range [0 1])
    # )

    # knn search
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)

    dist, idx = nbrs.kneighbors(y)

    N_split = np.ceil(y.size / 10e6)
    hh_mean = []
    n_size = []
    for section in np.array_split(range(N), N_split):
        zeros_mask = dist[section, k] != 0
        yy = y[section]
        r = dist[section, k]  # radius
        r = np.tile(r[:, np.newaxis], (1, dim))
        lb = (yy - r >= 0) * (yy - r) + (yy - r < 0) * 0
        ub = (yy + r <= 1) * (yy + r) + (yy + r > 1) * 1

        zeta = (ub - lb)[zeros_mask]  # remove zeros, duplicate points result in 0 distance
        # N = zeta.shape[0]
        hh = np.sum(np.log(zeta), axis=1)
        n_size.append(hh.shape[0])
        hh_mean.append(np.mean(hh))

    h = -spl.digamma(k) + spl.digamma(sum(n_size)) + utils.combine_means(hh_mean, n_size)

    return h


def ksg(y, n=None, k=1, shuffle=True, standardize=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004
    """

    y = np.asarray(y, dtype)

    if standardize == True:
        y_std = np.std(y, axis=0)
        y = y / y_std

    N, dim = y.shape

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # knn search
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)

    dist, idx = nbrs.kneighbors(y)

    N_split = np.ceil(y.size / 10e6)
    hh_mean = []
    n_size = []
    for section in np.array_split(range(N), N_split):
        y_dup = np.tile(
            y[section, np.newaxis, :], (1, k, 1)
        )  # duplicate last axis to add k dimension
        epsilons = np.max(np.abs(y_dup - y[idx[section, 1 : k + 1]]), axis=1)
        zeros_mask = ~np.any(epsilons == 0, axis=1)
        epsilons = epsilons[zeros_mask]
        if standardize == True:
            hh = np.sum(np.log(2 * epsilons * y_std), axis=1)
        else:
            hh = np.sum(np.log(2 * epsilons), axis=1)
        hh_mean.append(np.mean(hh))
        n_size.append(len(hh))
    # N = hh.shape[0]
    h_ksg[i] = (
        -spl.digamma(k)
        + spl.digamma(sum(n_size))
        + (dim - 1) / k
        + utils.combine_means(hh_mean, n_size)
    )

    return h


def tksg(y, n=None, k=1, shuffle=True, rng=np.random):
    """
    Implements the KSG entropy estimation in m-dimensional case, as discribed by:
    Alexander Kraskov, Harald Stogbauer, and Peter Grassberger, "Estimating Mutual Information", Physical review E, 2004
    """
    y = np.asarray(y, dtype)
    N, dim = y.shape

    if n is None:
        n = N
    else:
        n = min(n, N)

    # permute y
    if shuffle is True:
        rng.shuffle(y)

    # Auto algorithm switches to brute after dim=16. For truncated, better to swap at dim=30
    # algorithm = 'brute' if dim>30 else 'kd_tree'
    #    algorithm = 'kd_tree' #should almost always be faster for the uniform truncated case (range [0 1])

    # knn search
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm=algorithm, metric="chebyshev", n_jobs=n_jobs
    ).fit(y)
    dist, idx = nbrs.kneighbors(y)
    N_split = np.ceil(y.size / 10e6)

    hh_mean = []
    n_size = []
    for section in np.array_split(range(N), N_split):
        y_dup = np.tile(
            y[section, np.newaxis, :], (1, k, 1)
        )  # duplicate last axis to add k dimension
        epsilons = np.max(np.abs(y_dup - y[idx[section, 1 : k + 1]]), axis=1)
        zeta2 = np.minimum(y[section] + epsilons, 1) - np.maximum(y[section] - epsilons, 0)
        # remove zeros, invalid data. Zeros occur if points along a dimension are exactly the same
        zeros_mask = ~np.any(zeta2 == 0, axis=1)
        zeta2 = zeta2[zeros_mask]
        hh2 = np.sum(np.log(zeta2), axis=1)
        # N = zeta2.shape[0]
        n_size.append(hh2.shape[0])
        hh_mean.append(np.mean(hh2))

    h = (
        -spl.digamma(k)
        + spl.digamma(sum(n_size))
        + (dim - 1) / k
        + utils.combine_means(hh_mean, n_size)
    )

    return h


def learn_density(
    model,
    xs,
    regularizer=None,
    val_frac=0.05,
    step=ss.Adam(a=1.0e-3, bm=0.99),
    minibatch=100,
    patience=20,
    monitor_every=1,
    logger=sys.stdout,
    rng=np.random,
    val_tol=None,
    target=None,
    show_progress=False,
    coarse_fine_tune=False,
):
    """Train model to learn the density p(x).


    Args:
        model (_type_): model to train
        xs (list): list of samples to train on
        ws (_type_, optional): weights. Defaults to None.
        regularizer (_type_, optional): _description_. Defaults to None.
        val_frac (float, optional): _description_. Defaults to 0.05.
        step (_type_, optional): _description_. Defaults to ss.Adam(a=1.e-4).
        minibatch (int, optional): _description_. Defaults to 100.
        patience (int, optional): Epochs to try after best validation case. Defaults to 20.
        monitor_every (int, optional): _description_. Defaults to 1.
        logger (_type_, optional): _description_. Defaults to sys.stdout.
        rng (_type_, optional): _description_. Defaults to np.random.
        val_tol (_type_, optional): Tolerance if validation loss has improved. Defaults to None.
        target (_type_,optional): Target optimal validation value. Defaults to None

    Returns:
        _type_: Trained model
    """
    xs = xs if isinstance(xs, list) else [xs]
    xs = [np.asarray(_xs, dtype) for _xs in xs]

    n_data = xs[0].shape[0]

    # shuffle data, so that training and validation sets come from the same distribution
    idx = rng.permutation(n_data)
    xs = [_xs[idx] for _xs in xs]

    # split data into training and validation sets
    n_trn = int(n_data - val_frac * n_data)
    xs_trn = [_xs[:n_trn] for _xs in xs]
    xs_val = [_xs[n_trn:] for _xs in xs]

    trainer = trainers.SGD(
        model=model,
        trn_data=xs_trn,
        trn_loss=model.trn_loss if regularizer is None else model.trn_loss + regularizer,
        val_data=xs_val,
        val_loss=model.trn_loss,
        step=step,
        val_target=target,
    )
    trainer.train(
        minibatch=minibatch,
        patience=patience,
        monitor_every=monitor_every,
        logger=logger,
        val_Tol=val_tol,
        show_progress=show_progress,
        coarse_fine_tune=coarse_fine_tune,
    )

    return model


class UMestimator:

    def __init__(self, sim_model, model, samples=[None]):
        """Estimator class to hold the given neural net model and associated functions to train and evaluate entropy

        Args:
            sim_model (_type_): class that generates points from target distribution
            model (_type_): model to be learned
            samples (_type,optional): Initial samples used for training and entropy
        """

        self.sim_model = sim_model
        self.model = model
        self.samples = samples if isinstance(samples, list) else [samples]
        self.n_samples = None
        self.target = sim_model.entropy()
        self.checkpointer = trainers.ModelCheckpointer(model)

        self.checkpointer.checkpoint()

    def __del__(self):
        del self.samples
        del self.checkpointer

    def learn_transformation(
        self,
        n_samples,
        logger=sys.stdout,
        rng=np.random,
        patience=5,
        val_tol=None,
        show_progress=False,
        minibatch=256,
        coarse_fine_tune=True,
        step=ss.Adam(),
    ):
        """Learn the transformation to push a gaussian towards target distribution

        Args:
            n_samples (int): number of samples
            logger (_type_, optional): output log type. Defaults to sys.stdout.
            rng (_type_, optional): Defaults to np.random.
            patience (int, optional): How many epochs to try after finding the best validation. Defaults to 10.
            val_tol (int, optional): Tolerance of validation loss to decide when the model improved. Defaults to None.
            show_progress (bool, optional): True to display plot showing error over time after training completes. Defaults to False.
            minibatch (int, optional): minibatch size used in training. Defaults to 128.
            fine_tune (bool, optional): True to run training a second time with smaller step size. Defaults to False.
            step (step_strategies, optional): step type class. Defaults to ss.Adam().
        """

        step.reset_shared()  # shared variables can retain values between method calls

        if self.samples[0] is None:
            xs = self.sim_model.sim(n_samples)
            self.samples = xs

        self.n_samples = self.samples[0].shape[0]

        # Scale so validation occurs at most every 10**5 / minibatch during training
        # if fine_tune:
        #     monitor_every = min(1e5 / float(self.n_samples), 1.0)
        # else:
        #     monitor_every = min(1e6 / float(self.n_samples), 1.0)
        monitor_every = min(5e5 / float(self.n_samples), 1.0)
        logger.write("training model...\n")
        learn_density(
            self.model,
            self.samples,
            monitor_every=monitor_every,
            logger=logger,
            rng=rng,
            patience=patience,
            val_tol=val_tol,
            minibatch=minibatch,
            show_progress=show_progress,
            coarse_fine_tune=coarse_fine_tune,
            step=step,
        )
        logger.write("training done\n")

    def calc_ent(self, k=1, samples=None, method="umtksg", SHOW_PDF_PLOTS=False):
        """After training, evaluate the correction terms and knn entropy. Does not utilize paralell processing

        Args:
            k (int, optional): k neighbors for knn. Defaults to 1.
            reuse_samples (bool, optional): True to reuse samples stored in estimator. Defaults to True.
            method (str, optional): knn method ('kl','ksg','kl_ksg','umtkl','umtksg','both'). Defaults to 'umtksg'.
            SHOW_PDF_PLOTS (bool, optional): true to display plots of original data and transformed gaussian and uniform points. Defaults to False.

        Returns:
            _type_: entropy estimate. Return tuple (H KL, H KSG) if method='both'
        """
        if samples is None:
            samples = self.sim_model.sim(self.n_samples)

        correction = 0
        if method == "kl":
            H = kl(samples, k=k)
        elif method == "ksg":
            H = ksg(samples, k=k)
        elif method == "kl_ksg":
            H = kl_ksg(samples, k=k)
        else:
            print("evaluating uniformizing corrections")
            samples = [samples] if not isinstance(samples, list) else samples
            uniform, correction = self.uniform_correction(samples, SHOW_PDF_PLOTS)
            print("evaluating uniform knn")
            if method == "umtkl":
                H = tkl(uniform, k=k)
            elif method == "umtksg":
                H = tksg(uniform, k=k)
            elif method == "both":
                H = tkl_tksg(uniform, k=k)
            else:
                raise Exception("invalid method type")

        # If method is 'both', then H is a tuple with (H_KL,H_KSG)
        return np.asarray(H) + correction

    def knn_ent(self, k=1, reuse_samples=True, method="kl"):
        """
        DEPRECATED: Use calc_ent using the flags method={'kl','ksg','kl_ksg'}
        Evaluate knn directly without uniformizing or using truncated knn

        Args:
            k (int, optional): k value for knn. Defaults to 1.
            reuse_samples (bool, optional): True to reused stored samples. Defaults to True.
            method (str, optional): knn method ('kl','ksg','both'). Defaults to 'kl'.

        Returns:
            _type_: _description_
        """

        if reuse_samples:
            samples = self.samples[0]
        else:
            samples = self.sim_model.sim(self.n_samples)

        if method == "kl":
            return kl(samples, k=k)
        elif method == "ksg":
            return ksg(samples, k=k)
        elif method == "both":
            return kl_ksg(samples, k=k)

    def uniform_correction(self, samples=None, SHOW_PDF_PLOTS=False):
        """Generate uniformized data and the associated entropy correction terms

        Args:
            samples (numpy, optional): data samples to generate uniform points for. New samples are generated if set to None. Defaults to True.
            reuse_samples (bool, optional): Use data samples stored in estimator. Setting to true will override given samples. Defaults to False.
            SHOW_PDF_PLOTS (bool, optional): Display histogram of given data, gaussian, and uniform transforms. Defaults to False.

        Returns:
            _type_: Uniformized points, Jacobian entropy correction term
        """

        if samples is None:
            samples = self.sim_model.sim(self.n_samples)

        # z = self.model.calc_random_numbers(samples)

        chunk_size = 1e6
        tot_samp = samples[0].shape[0]
        N_split = np.ceil(tot_samp / chunk_size)
        uniform = np.empty_like(samples[0])
        n_size = []
        correction1 = []
        correction2 = []
        start_idx = 0
        for section in np.array_split(range(tot_samp), N_split):
            z_chunk = self.model.calc_random_numbers([x[section] for x in samples])
            idx = np.all(np.abs(z_chunk) < stats.norm.ppf(1.0 - 1e-6), axis=1)
            z_chunk = z_chunk[idx]
            n_size.append(len(z_chunk))
            # something went wrong in training, reload initial value
            if np.any(np.isnan(z_chunk)):
                self.checkpointer.restore()
                z = self.model.calc_random_numbers([x[section] for x in samples])
            uniform[start_idx : start_idx + n_size[-1]] = stats.norm.cdf(z_chunk)
            start_idx = start_idx + n_size[-1]
            correction1.append(-np.mean(np.sum(np.log(stats.norm.pdf(z_chunk)), axis=1)))
            correction2.append(
                -np.mean(self.model.logdet_jacobi_u([x[section][idx] for x in samples]))
            )
        uniform = uniform[:start_idx]  # remove ending empty samples

        # memory efficient mean
        # mean(val) = 1/N * sum(mean(val_i)*N_i)
        N_tot = sum(n_size)
        correction1 = utils.combine_means(correction1, n_size)
        correction2 = utils.combine_means(correction2, n_size)
        del z_chunk

        # remove extreme data that isn't within 99.9999% of the norm dist
        # idx = np.all(np.abs(z) < stats.norm.ppf(1.0 - 1e-6), axis=1)
        # z = z[idx]

        # # Made a bad gaussian estimate
        # if z.shape[0] < 0.01 * self.samples[0].shape[0]:
        #     return None

        if SHOW_PDF_PLOTS == True:
            # Plot histograms of original data, gaussian, and uniform transforms
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 3)
            x = np.linspace(-0.1, 1.1, 100)
            ax[0].plot(x, stats.uniform.pdf(x), lw=5)
            x = np.linspace(stats.norm.ppf(1e-6), stats.norm.ppf(1 - 1e-6), 100)
            ax[1].plot(x, stats.norm.pdf(x), lw=5)
            ax[2].plot(x, stats.norm.pdf(x), lw=5)

            ax[0].hist(stats.norm.cdf(z), bins=40, density=True), ax[0].set_title(
                "Transformed Uniform"
            )
            ax[1].hist(z, bins=40, density=True), ax[1].set_title("Transformed Gaussian")
            ax[2].hist(samples[0], bins=100, density=True), ax[2].set_title("Original Data")

        # Jacobian correction from the CDF
        # uniform = stats.norm.cdf(z)
        # correction1 = -np.mean(np.sum(np.log(stats.norm.pdf(z)), axis=1))

        # logdet = self.model.logdet_jacobi_u(samples)
        # correction2 = -np.mean(logdet[idx])

        return uniform, correction1 + correction2

    def start_knn_thread(self, data, k=1, method="umtksg"):
        """Start the knn algorithm in a thread and return the background thread for the user to handle

        Args:
            data (_type_): Uniformized data points
            k (int, optional): K value for knn. Defaults to 1.
            method (str, optional): type of truncated KNN to use ('umtkl','umtksg','both'). Defaults to 'umtksg'.
        Returns:
            BackgroundThread: thread running the knn algorithm
        """
        if method == "umtkl":
            self.h_thread = BackgroundThread(target=tkl, args=(data, None, k))
            # h = tkl(z, k=k) + correction1
        elif method == "umtksg":
            self.h_thread = BackgroundThread(target=tksg, args=(data, None, k))
            # h = tksg(z, k=k) + correction1
        elif method == "both":
            self.h_thread = BackgroundThread(target=tkl_tksg, args=(data, None, k))
            # h,h2 = tkl_tksg(z,k=k) + correction1
        self.h_thread.start()
        return self.h_thread
