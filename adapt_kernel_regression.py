"""The :mod:`sklearn.kernel_regressor` module implements the Kernel Regressor.
"""
# Author: Jan Hendrik Metzen <janmetzen@mailbox.de>
#
# License: BSD 3 clause

import numpy as np

# from sklearn.metrics.pairwise import pairwise_kernels
from pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors


class KernelRegression(BaseEstimator, RegressorMixin):
    """Nadaraya-Watson kernel regression with automatic bandwidth selection.

    This implements Nadaraya-Watson kernel regression with (optional) automatic
    bandwith selection of the kernel via leave-one-out cross-validation. Kernel
    regression is a simple non-parametric kernelized technique for learning
    a non-linear relationship between input variable(s) and a target variable.

    ALTERING FOR ADAPTIVE SMOOTHING

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF ("bandwidth"), polynomial,
        exponential chi2 and sigmoid kernels. Interpretation of the default
        value is left to the kernel; see the documentation for
        sklearn.metrics.pairwise. Ignored by other kernels. If a sequence of
        values is given, one of these values is selected which minimizes
        the mean-squared-error of leave-one-out cross-validation.
        These are scaled by power of k-nearest-neighbor distance.
    n_neighbors : int
        number of neighbors to use in computing individual point smoothing lengths

    See also
    --------

    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """

    def __init__(self, kernel="rbf", gamma=None, n_neighbors=16):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X, y, indX=None, pary=None, plotcv=False):
        """Fit the model

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values

        pary (optional): if provided, should be independent sample of y values
          evaluated at X, or else at X[indX,:], to use parallel-validation
          instead of cross-validation.

        indX (optional): if provided, pary should be independent sample
          evaluated at X[indX,:]

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.y = y

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        dneighb, ineighb = nbrs.kneighbors(X)
        dkn = dneighb[:,-1]
        # self.ptscale = 1.0 / dkn**2  #constant neighbor number
        self.ptscale = 1.0 / dkn  # from silverman 1-d discussion, unsure about n-d though

        if hasattr(self.gamma, "__iter__"):
            gamma_in = self.gamma
            if (pary is not None):
                if (indX is not None):
                    parX = X[indX,:]
                else:
                    parX = X
                self.gamma = self._optimize_gamma_pv(self.gamma, pary, parX,
                                                     plot=plotcv)
            else:
                self.gamma = self._optimize_gamma(self.gamma, plot=plotcv)
            if (self.gamma == gamma_in.min()) or (self.gamma == gamma_in.max()):
                print 'Chosen gamma at extreme of input choices:'
                print self.gamma
                print gamma_in.min(), gamma_in.max()
        return self

    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """
        K = pairwise_kernels(self.X, X,
                             metric=self.kernel,
                             gamma=self.gamma * self.ptscale,
                             norm=True)
        K = K - K.max(axis=0)[np.newaxis, :]  # only relative values along 0 axis matter
        np.exp(K, K)
        return ysmooth(self.y, K)

    def _optimize_gamma(self, gamma_values, plot=False):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in leave-one-out cross validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, self.X, metric=self.kernel,
                                 gamma=gamma * self.ptscale,
                                 norm=True)
            np.fill_diagonal(K, -np.inf)  # leave-one-out - exponentiates to zero
            K = K - K.max(axis=0)[np.newaxis, :]  # only relative values along 0 axis matter
            np.exp(K, K)
            y_pred = ysmooth(self.y, K)
            mse[i] = ((y_pred - self.y) ** 2).mean()
        self.mse = np.nanmin(mse)
        if (plot):
            import matplotlib.pyplot as plt
            plt.semilogx(gamma_values, mse, 'bd-')
            plt.plot(gamma_values, gamma_values*0, 'kd')
            plt.plot(gamma_values[np.nanargmin(mse)], mse[np.nanargmin(mse)], 'rs')
            plt.show()

        return gamma_values[np.nanargmin(mse)]

    def _optimize_gamma_pv(self, gamma_values, pary, parX, plot=False):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in parallel validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, parX, metric=self.kernel,
                                 gamma=gamma * self.ptscale,
                                 norm=True)
            K = K - K.max(axis=0)[np.newaxis, :]  # only relative values along 0 axis matter
            np.exp(K, K)
            y_pred = ysmooth(self.y, K)
            mse[i] = ((y_pred - pary) ** 2).mean()

        if (plot):
            import matplotlib.pyplot as plt
            plt.semilogx(gamma_values, mse, 'bd-')
            plt.plot(gamma_values, gamma_values*0, 'kd')
            plt.plot(gamma_values[np.nanargmin(mse)], mse[np.nanargmin(mse)], 'rs')
            plt.show()
        self.mse = np.nanmin(mse)

        return gamma_values[np.nanargmin(mse)]


def ysmooth(y, K):
    """Smooth y measured at old points onto new points as specified by kernel K[old,new]
    using Nadaraya-Watson"""
    return (K * y[:, np.newaxis]).sum(axis=0) / K.sum(axis=0)
