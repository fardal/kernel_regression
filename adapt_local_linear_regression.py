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
    """Locally linear kernel regression with automatic bandwidth selection.

    This implements locally linear kernel regression with (optional) automatic
    bandwith selection of the kernel via leave-one-out cross-validation. Locally
    linear regression is a simple non-parametric kernelized technique for learning
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

    def __init__(self, kernel="rbf", gamma=None, n_neighbors=16, onedmethod=False):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        if onedmethod:
            self.ysmooth = ysmooth_1d
        else:
            self.ysmooth = ysmooth

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
        self.ptscale = 1.0 / dkn   #from silverman 1-d discussion, unsure about n-d though

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
        K = K - K.max(axis=0)[np.newaxis, :]  #only relative values along 0 axis matter
        np.exp(K, K)
        # this is the Nadaraya-Watson estimate
        # return ysmooth(self.y, K)
        # this is the locally linear estimate
        return self.ysmooth(self.y, self.X, X, K)

    def _optimize_gamma(self, gamma_values, plot=False):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in leave-one-out cross validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, self.X, metric=self.kernel,
                                 gamma=gamma * self.ptscale,
                                 norm=True)
            np.fill_diagonal(K, -np.inf)  # leave-one-out - exponentiates to zero
            K = K - K.max(axis=0)[np.newaxis, :]  #only relative values along 0 axis matter
            np.exp(K, K)
            y_pred = self.ysmooth(self.y, self.X, self.X, K)
            mse[i] = ((y_pred - self.y) ** 2).mean()
            
        if (plot):
            import matplotlib.pyplot as plt
            plt.semilogx(gamma_values, mse, 'bd-')
            plt.plot(gamma_values, gamma_values*0, 'kd')
            plt.plot(gamma_values[np.nanargmin(mse)], mse[np.nanargmin(mse)], 'rs')
            plt.show()
        self.mse = np.nanmin(mse)
            
        return gamma_values[np.nanargmin(mse)]

    def _optimize_gamma_pv(self, gamma_values, pary, parX, plot=False):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in parallel validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, parX, metric=self.kernel,
                                 gamma=gamma * self.ptscale,
                                 norm=True)
            K = K - K.max(axis=0)[np.newaxis, :]  #only relative values along 0 axis matter
            np.exp(K, K)
            try:
                y_pred = self.ysmooth(self.y, self.X, parX, K)
                mse[i] = ((y_pred - pary) ** 2).mean()
            except np.linalg.LinAlgError:
                mse[i] = np.nan
            
        if (plot):
            import matplotlib.pyplot as plt
            plt.semilogx(gamma_values, mse, 'bd-')
            plt.plot(gamma_values, gamma_values*0, 'kd')
            plt.plot(gamma_values[np.nanargmin(mse)], mse[np.nanargmin(mse)], 'rs')
            plt.show()
        self.mse = np.nanmin(mse)
            
        return gamma_values[np.nanargmin(mse)]

# def ysmooth(y, K):
#     """Smooth y measured at old points onto new points as specified by kernel K[old,new]
#     using Nadaraya-Watson"""
#     return (K * y[:, np.newaxis]).sum(axis=0) / K.sum(axis=0)


def ysmooth(y, Xold, Xnew, K):
    """Smooth y measured at old points Xold onto new points Xnew
    as specified by kernel K[old,new] using locally linear regression. Only works in 1-d"""
    # return (K * y[:, np.newaxis]).sum(axis=0) / K.sum(axis=0)  #Nadaraya-Watson
    # remember self.X, X are n_samples x n_dim
    n = Xold.shape[0]
    m = Xnew.shape[0]
    d = Xold.shape[1]
    p = d + 1
    assert (y.shape==(n,))
    assert (Xnew.shape[1]==d)
    assert (K.shape==(n,m))
    assert (y.shape[0]==Xold.shape[0])
    # F matrix is design matrix (Wasserman and others confusingly call this X_x)
    # three dimensions because it differs between every point of Xnew
    deltax = Xold[np.newaxis,:,:] - Xnew[:,np.newaxis,:]  #x_i - X
    F = np.zeros((m, n, p))
    F[:,:,0] = 1.  #intercept term
    F[:,:,1:] = deltax
    #
    WF = K.T[:, :, np.newaxis] * F
    D = np.einsum('irj,irk->ijk', F, WF)  #this is the (X_x^T W_x X_x) matrix of Wasserman
    invmat = np.linalg.inv(D)  #should work even with leading index
    firstrow = invmat[:,0,:]
    B = np.einsum('il,ijl->ij', firstrow, WF)
    #B is now an m x n array
    yfit = (B * y[np.newaxis,:]).sum(axis=1) / B.sum(axis=1)
    return yfit

def ysmooth_1d(y, Xold, Xnew, K):
    """Smooth y measured at old points Xold onto new points Xnew
    as specified by kernel K[old,new] using locally linear regression. Only works in 1-d"""
    # return (K * y[:, np.newaxis]).sum(axis=0) / K.sum(axis=0)  #Nadaraya-Watson
    # remember self.X, X are n_samples x n_dim
    # this version for 1-d only
    assert Xold.shape[1]==1
    assert Xnew.shape[1]==1
    deltax = Xold[:,0][:,np.newaxis] - Xnew[:,0][np.newaxis,:]  #x_i - X
    s1 = (K * deltax).sum(axis=0)
    s2 = (K * deltax**2).sum(axis=0)
    B = K * (s2[np.newaxis,:] - deltax * s1[np.newaxis,:])
    yfit = (B * y[:, np.newaxis]).sum(axis=0) / B.sum(axis=0)
    return yfit
