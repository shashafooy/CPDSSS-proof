import numpy as np
import scipy.misc
import itertools


def isposint(n):
    """
    Determines whether number n is a positive integer.
    :param n: number
    :return: bool
    """
    return isinstance(n, int) and n > 0


def isdistribution(p):
    """
    :param p: a vector representing a discrete probability distribution
    :return: True if p is a valid probability distribution
    """
    return np.all(p >= 0.0) and np.isclose(np.sum(p), 1.0)


def logistic(x):
    """
    Elementwise logistic sigmoid.
    :param x: numpy array
    :return: numpy array
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))


def logsoftmax(x):
    """
    Calculates the log softmax of x, or of the rows of x.
    :param x: vector or matrix
    :return: log softmax
    """

    x = np.asarray(x)

    if x.ndim == 1:
        return x - scipy.misc.logsumexp(x)

    elif x.ndim == 2:
        return x - scipy.misc.logsumexp(x, axis=1)[:, np.newaxis]

    else:
        raise ValueError("input must be either vector or matrix")


def softmax(x):
    """
    Calculates the softmax of x, or of the rows of x.
    :param x: vector or matrix
    :return: softmax
    """

    return np.exp(logsoftmax(x))


def discrete_sample(p, n_samples=None, rng=np.random):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples, only 1 if None
    :return: vector of samples
    """

    # check distribution
    # assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    one_sample = n_samples is None

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    r = rng.rand(1 if one_sample else n_samples, 1)
    samples = np.sum((r > c).astype(int), axis=1)

    return samples[0] if one_sample else samples


def importance_sample(target, proposal, n_samples, rng=np.random):
    """
    Importance sampling.
    :param target: target distribution
    :param proposal: proposal distribution
    :param n_samples: number of samples
    :param rng: random generator to use
    :return: samples, normalized log weights
    """

    xs = proposal.gen(n_samples, rng=rng)
    log_ws = target.eval(xs, log=True) - proposal.eval(xs, log=True)
    log_ws -= scipy.misc.logsumexp(log_ws)

    return xs, log_ws


def ess_importance(ws):
    """
    Calculates the effective sample size of a set of weighted independent samples (e.g. as given by importance
    sampling or sequential monte carlo). Takes as input the normalized sample weights.
    """

    ess = 1.0 / np.sum(ws**2)
    return ess


def ess_mcmc(xs):
    """
    Calculates the effective sample size of a correlated sequence of samples, e.g. as given by markov chain monte
    carlo.
    """

    n_samples, n_dims = xs.shape

    mean = np.mean(xs, axis=0)
    xms = xs - mean

    acors = np.zeros_like(xms)
    for i in range(n_dims):
        for lag in range(n_samples):
            acor = np.sum(xms[: n_samples - lag, i] * xms[lag:, i]) / (n_samples - lag)
            if acor <= 0.0:
                break
            acors[lag, i] = acor

    act = 1.0 + 2.0 * np.sum(acors[1:], axis=0) / acors[0]
    ess = n_samples / act

    return np.min(ess)


def calc_whitening_transform(xs):
    """
    Calculates the parameters that whiten a dataset.
    """

    assert xs.ndim == 2, "Data must be a matrix"
    N = xs.shape[0]

    means = np.mean(xs, axis=0)
    ys = xs - means

    cov = np.dot(ys.T, ys) / N
    vars, U = np.linalg.eig(cov)
    istds = np.sqrt(1.0 / vars)

    return means, U, istds


def whiten(xs, params):
    """
    Whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys -= means
    ys = np.dot(ys, U)
    ys *= istds

    return ys


def de_whiten(xs, params):
    """
    De-whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys /= istds
    ys = np.dot(ys, U.T)
    ys += means

    return ys


def median_distance(xs):
    """
    Calculate the median distance of a set of points.
    :param xs: matrix with points as rows
    :return: median distance
    """

    xs = np.asarray(xs)
    diffs = np.array([x1 - x2 for x1, x2 in itertools.combinations(xs, 2)])
    dists = np.sqrt(np.sum(diffs**2, axis=1))

    return np.median(dists)


def zadoff_chu(N, u):
    """Evaluate the length N zadoff_chu sequence with root index u and unit power. Index u must be prime with respect to N

    Args:
        N (_type_): length of sequence
        u (_type_): root index prime to N
    """
    assert np.gcd(N, u) == 1, "u is not prime with respect to N"
    N_range = np.arange(N)
    if N % 2 == 0:  # even
        return np.exp(-1j * np.pi * u * N_range**2 / N) / np.sqrt(N)
    else:  # odd
        return np.exp(-1j * np.pi * u * N_range * (N_range + 1) / N) / np.sqrt(N)


def project(v, u, pre_normalized=False):
    uv = np.sum(u * v, axis=1, keepdims=True)
    # if previous vector u has been normalized, it will always be 1
    uu = 1 if pre_normalized else np.sum(u * u, axis=1, keepdims=True) + 1e-10
    return uv / uu * u

    # return np.dot(u, v) / np.dot(u, u) * u


def gram_schmidt(vectors, normalize=False):
    n_dims = vectors.ndim
    if n_dims == 2:
        vectors = vectors[np.newaxis, :, :]  # batch size of 1
    n_samples, N, M = vectors.shape
    Q = np.zeros_like(vectors)
    for i in range(M):  # iterate over each column
        v = vectors[:, :, i]
        for j in range(i):  # project based on previous columns
            u = Q[:, :, j]
            v -= project(v, u, pre_normalized=normalize)

        if normalize:
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
        Q[:, :, i] = v
    if n_dims == 2:
        Q = Q[0]
    return Q

    # orthogonal_vectors = []
    # for v in vectors.T:  # iterate over each column
    #     # for v in vectors:  # iterate over each column
    #     for u in orthogonal_vectors:
    #         v -= project(v, u)
    #     orthogonal_vectors.append(v)

    # orthogonal_vectors = np.array(orthogonal_vectors).T  # return to original orientation
    # # orthogonal_vectors = np.array(orthogonal_vectors)  # return to original orientation

    # if normalize:
    #     return orthogonal_vectors / np.linalg.norm(orthogonal_vectors, axis=0)
    # else:
    #     return orthogonal_vectors
