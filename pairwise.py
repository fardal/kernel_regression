# -*- coding: utf-8 -*-

#alters code from sklearn/pairwise.py
#goal is to allow rbf_kernel with individual smoothing lengths,
#  also to prevent NaN resulting from premature exponentiation

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances, \
    _parallel_pairwise, _pairwise_callable
from sklearn.utils.fixes import partial

def log_rbf_kernel(X, Y=None, gamma=None, norm=False):
    """
    Compute the rbf (gaussian) kernel between X and Y::

        K(x, y) = exp(-gamma ||x-y||^2)

    for each pair of rows x in X and y in Y.  Return the log of this.
    Allows for individual smoothing lengths (gamma values) in X.
    Note gamma = 1/(2 sigma^2) where sigma is the std or smoothing length.

    Read more in the :ref:`User Guide <rbf_kernel>`.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float, or array of shape n_samples_X

    norm : whether to include the kernel normalizing factor 1/(2 pi sigma^2)^(d/2)

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    #K now has shape n_samples_X, n_samples_Y
    if np.isscalar(gamma):
        K *= -gamma
        if (norm):
            dim = X.shape[1]
            K += (dim/2.) * np.log(gamma/np.pi)
    else:
        assert gamma.size == X.shape[0]
        K *= -gamma[:,np.newaxis]  #need this new axis to broadcast correctly
        if (norm):
            dim = X.shape[1]
            K += (dim/2.) * np.log(gamma[:,np.newaxis]/np.pi)
    # np.exp(K, K)    # exponentiate K in-place
    return K


# Helper functions - distance
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'rbf': log_rbf_kernel }


def pairwise_kernels(X, Y=None, metric="linear", filter_params=False,
                     n_jobs=1, **kwds):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are::
        ['rbf']
    These should be included later::
        ['sigmoid', 'polynomial', 'poly', 'linear', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise kernels between samples, or a feature array.

    Y : array [n_samples_b, n_features]
        A second feature array only if X has shape [n_samples_a, n_features].

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    filter_params: boolean
        Whether to filter invalid parameters or not.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    """
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = dict((k, kwds[k]) for k in kwds
                        if k in KERNEL_PARAMS[metric])
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)



