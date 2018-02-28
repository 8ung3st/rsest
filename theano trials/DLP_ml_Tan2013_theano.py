import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/4. Estimation/rsest/")
import numpy as np
import lib.read_dta as read
import lib.dlp_th as dlp
import lib.ndiff as ndiff
from __future__ import division
np.random.seed(1) # set the seed

"--- Load Data ---"
path = "../../3. Data/Tan_3-4/Sample_py.dta"
theta0, W, X, ZA, ZB, ZE, S, setup, tnames, meanZ, stdZ = read.getdata(path) # the artificial data is set up in dlp_art_data.py

#dlp.neglogl(theta0, W, X, ZA, ZB, ZE, S, setup)

import theano
import theano.tensor as t
#f = theano.function()

thetat = t.fvector()
Wt = t.fmatrix()
Xt = t.fvector()
ZAt = t.fmatrix()
ZBt = t.fmatrix()
ZEt = t.fmatrix()
St = t.fvector()
setupt = t.wvector()

nlogl = dlp.neglogl(thetat, Wt, Xt, ZAt, ZBt, ZEt, St, setup)



class linreg(object):
    def __init__(self, beta, y, x):
        self.beta = beta
        self.y = y
        self.x = x

    def mu(self):
        return t.dot(self.beta,self.x.T)

    def rss(self):
        diff = (self.y - self.mu())**2
        rss = t.sum(diff)
        grad = t.grad(rss,self.beta)
        return rss, grad

beta = t.dscalar('beta')
x = t.fvector()
y = t.fvector()
model = linreg(beta, y , x)
#mean = model.mu()
#fn = theano.function(inputs=[],outputs=[mean])
#fn()

RSS, grad = model.rss()
fn2 = theano.function(inputs=[], outputs=[RSS, grad], givens={model.x: X, model.y: W[:,0], model.beta: 2.})
fn2()

W[:,0].dtype





"--- Optimize: ---"
import scipy.optimize as opt
tol = 1e-7
thetahat = opt.fmin_powell(dlp.neglogl, theta0, args = (W, X, ZA, ZB, ZE, S, setup), xtol=tol, ftol=tol, maxfun = 1e6, maxiter = 1e5) # this takes 25 min!

H = ndiff.hessian(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
paramcov = np.linalg.inv(H)
thetasd = np.sqrt(np.diag(paramcov))

np.save("thetahat.npy",thetahat)


"Test"
thetahat = np.load("thetahat.npy")
testgrad = ndiff.gradient(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
print testgrad
testhessian = ndiff.hessian(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
print testhessian
"----"

"--- Output: ---"
np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
np.set_printoptions(suppress=True)
tdim = np.size(theta0)
print reduce(np.append, [range(tdim), theta0, thetahat, thetasd]).reshape(-1,tdim).T

thetatable = reduce(np.append, [theta0, thetahat, thetasd]).reshape(-1,tdim).T
latextable = ""
for entry, line in zip(tnames, thetatable): # just remove the "+ entry" to have no parameter names
    latextable = latextable + entry + " & " + " & ".join(map('{0:.3f}'.format, line)) + " \\\\\n"

with open("../tab/Tan_3-4.tex", "w") as txt_file:
    txt_file.write(latextable)



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
