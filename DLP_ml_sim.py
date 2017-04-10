import numpy as np
import lib.dlp_art_data as dlp_art_data
import lib.dlp as dlp
import lib.ndiff as ndiff
from __future__ import division
np.random.seed(1) # set the seed

"--- Simulate Data ---"
theta0, W, X, Z, ZE, setup = dlp_art_data.get() # the artificial data is set up in dlp_art_data.py

"--- Optimize: ---"
import scipy.optimize as opt
tol = 1e-7
thetahat = opt.fmin_powell(dlp.neglogl, theta0, args = (W, X, Z, ZE, setup), xtol=tol, ftol=tol, maxfun = 1e5, maxiter = 1e5)

H = ndiff.hessian(dlp.neglogl, thetahat, 1e-6, (W, X, Z, ZE, setup))
paramcov = np.linalg.inv(H)
thetasd = np.sqrt(np.diag(paramcov))

"--- Output: ---"
np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
np.set_printoptions(suppress=True)
tdim = np.size(theta0)
print reduce(np.append, [range(tdim),theta0, thetahat, thetasd]).reshape(-1,tdim).T

"--- Starting Value Analysis ---"
def startvaltest(thetahat, startingcov, S, tol):
    tdim = np.size(thetahat)
    Thetas = np.zeros((tdim,S))
    startingcov = paramcov/10
    for s in range(S):
        thetainit = np.random.multivariate_normal(theta0,startingcov)
        Thetas[:,s] = opt.fmin_powell(dlp.neglogl, thetainit, args = (W, X, Z, ZE, setup), xtol=tol, ftol=tol, maxfun = 1e5, maxiter = 1e5)
    return Thetas

Thetas = startvaltest(thetahat, paramcov/10, 10, tol)
