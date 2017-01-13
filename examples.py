
from __future__ import division  #for KernelRidge example

import numpy as np
import matplotlib.pyplot as plt



def twodcompare():
    'parabola in 2-d uniform sampling'
    from kernel_regression import KernelRegression as KernelRegression1
    from local_linear_regression import KernelRegression as KernelRegression2
    # n = 200
    n = 1000
    # n = 10000

    def getsample(n):
        rsq = np.random.random(size=n)
        phi = np.random.random(size=n) * 2.*np.pi
        r = np.sqrt(rsq)
        X = np.empty((n, 2))
        X[:,0] = r * np.cos(phi)
        X[:,1] = r * np.sin(phi)
        y = 0.5 * rsq
        sigy = 0.01
        yn = y + np.random.normal(size=y.shape) * sigy
        return r, X, y, yn
    r0, X0, y0, yn0 = getsample(n)
    r, X, y, yn = getsample(n//20)

    # sigsmooth = 0.30
    sigsmooth = 0.030
    gamma = 1./2./sigsmooth**2 * np.logspace(-1.0, 1.0, 21)
    kr1 = KernelRegression1(kernel="rbf", gamma=gamma)
    y_kr1 = kr1.fit(X0, y0).predict(X)
    gamma = 1./2./sigsmooth**2 * np.logspace(-1.0, 1.0, 21)
    kr2 = KernelRegression2(kernel="rbf", gamma=gamma)
    y_kr2 = kr2.fit(X0, y0).predict(X)
    print kr1.get_params()
    print kr2.get_params()

    # plt.scatter(X[:,0], X[:,1], c=Y, alpha=0.5)
    # plt.show()
    # plt.plot(r, Yn, 'b.')
    # plt.show()

    plt.plot(r0, y0, 'k.', label='training data')
    plt.plot(r, y_kr1, 'r.', label='Locally constant')
    plt.plot(r, y_kr2, 'g.', lw=4, label='Locally linear')
    # plt.plot(r, y, 'b.', label='True value')
    plt.xlabel('radius')
    plt.ylabel('target')
    plt.xlim([0., 1.2])
    plt.legend(loc='upper left')
    plt.show()
    

def krr_example():
    'Example of KRR from sklearn website.  Illustrates odd behavior beyond data region'
    import time
    from sklearn.grid_search import GridSearchCV
    from sklearn.learning_curve import learning_curve
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.RandomState(0)

    # Generate sample data
    X = 5 * rng.rand(10000, 1)  #default
    X = 0.8 * X + 1.2 * (X > 2.5)  #skip middle part
    y = np.sin(X).ravel() + 5.
    
    # Add noise to targets
    y += 0.5 * 0.29 * np.random.normal(size=X.shape[0])   #evenly distributed

    # X_plot = np.linspace(-10, 15, 10000)[:, None]
    X_plot = np.linspace(-5, 10, 10000)[:, None]

    # Fit regression model
    train_size = 300

    kr0 = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.1)
    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                      param_grid={"alpha": [1e-2],
                                  "gamma": np.logspace(-2, 2, 81)})

    kr0.fit(X[:train_size], y[:train_size])
    
    t0 = time.time()
    kr.fit(X[:train_size], y[:train_size])
    p = kr.best_estimator_.get_params()
    print 'gamma: ', p['gamma']
    print 'alpha: ', p['alpha']
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % kr_fit)

    y_kr0 = kr0.predict(X_plot)
    
    t0 = time.time()
    y_kr = kr.predict(X_plot)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], kr_predict))


    #############################################################################
    # look at the results
    # sv_ind = svr.best_estimator_.support_
    # plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors')
    # plt.scatter(X[:100], y[:100], c='k', label='data')
    plt.scatter(X[::100], y[::100], c='k', label='data')
    plt.hold('on')
    # plt.plot(X_plot, y_svr, c='r',
    #          label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.plot(X_plot, y_kr, c='g', lw=2, 
             label='KRR CV')
    plt.plot(X_plot, y_kr0, 'm--', lw=4, 
             label='KRR hardcoded params')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend()

    plt.show()


def onedcompare():
    'Compare methods in 1-d'
    import time
    from kernel_regression import KernelRegression as KernelRegression1
    from local_linear_regression import KernelRegression as KernelRegression2
    
    rng = np.random.RandomState(0)

    # Generate sample data
    X = 5 * rng.rand(300, 1)  #same training number as KRR example
    y = np.sin(X).ravel()

    # Add noise to targets
    y += 0.5 * 0.29 * np.random.normal(size=X.shape[0])   #evenly distributed

    X_plot = np.linspace(0., 5., 1000)[:, np.newaxis]
    y_true = np.sin(X_plot).ravel()

    sigsmooth = 0.30
    gamma = 1./2./sigsmooth**2
    kr1 = KernelRegression1(kernel="rbf", gamma=gamma)
    y_kr1 = kr1.fit(X, y).predict(X_plot)
    kr2 = KernelRegression2(kernel="rbf", gamma=gamma)
    y_kr2 = kr2.fit(X, y).predict(X_plot)
    
    # Visualize models
    plt.plot(X, y, 'k.', label='data')
    plt.plot(X_plot, y_kr1, 'r', label='Locally constant')
    plt.plot(X_plot, y_kr2, 'g--', lw=4, label='Locally linear')
    plt.plot(X_plot, y_true, 'k-', label='True value')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend()
    plt.show()


def check_llmethods():
    'Check locally linear regression multidim vs 1d formalism'
    import time
    from local_linear_regression import KernelRegression
    
    rng = np.random.RandomState(0)

    # Generate sample data
    X = 5 * rng.rand(500, 1)  #same training number as KRR example
    y = np.sin(X).ravel()

    # Add noise to targets
    y += 0.5 * 0.29 * np.random.normal(size=X.shape[0])   #evenly distributed

    X_plot = np.linspace(0., 5., 200)[:, None]

    sigsmooth = 0.30
    kr0 = KernelRegression(kernel="rbf", gamma=1./2./sigsmooth**2, onedmethod=True)
    kr = KernelRegression(kernel="rbf", gamma=1./2./sigsmooth**2)
    t0 = time.time()
    y_kr0 = kr0.fit(X, y).predict(X_plot)
    y_kr = kr.fit(X, y).predict(X_plot)
    y_true = np.sin(X_plot).ravel()
    print kr0.get_params()
    print kr.get_params()
    print 'rms method difference: ', (y_kr0-y_kr).std()
    
    # Visualize models
    plt.plot(X, y, 'k.', label='data')
    plt.plot(X_plot, y_kr0, 'r', label='1-d formalism')
    plt.plot(X_plot, y_kr, 'g--', lw=4, label='multi-d formalism')
    plt.plot(X_plot, y_true, 'k-', label='True value')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend()
    plt.show()


