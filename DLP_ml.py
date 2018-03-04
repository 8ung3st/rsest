import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/4. Estimation/rsest/")
import numpy as np
import lib.read_dta as read
import lib.dlp as dlp
import lib.ndiff as ndiff
from __future__ import division
np.random.seed(1) # set the seed

"--- Load Data ---"
path = "../../3. Data/Tan_3-4/Sample_py.dta"
theta0, W, X, ZA, ZB, ZE, S, I, setup, tnames, meanZ, stdZ = read.getdata(path) # the artificial data is set up in dlp_art_data.py

"--- Optimize: ---"
import scipy.optimize as opt
import time

tol = 5e-7 # 1e-8 will take about three times as long as 1e-6
start_time_th = time.time()
Result = opt.minimize(dlp.neglogl, theta0, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
thetahat = Result.x
print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

"--- Bootstrap: ---"
from joblib import Parallel, delayed

def bootstrap(t):
    np.random.seed(None) # this way each worker will get a new seed (https://github.com/joblib/joblib/issues/36)
    sample = np.random.choice(range(setup[0]),setup[0], replace = True) # sample with replacement
    Ws, Xs, ZAs, ZBs, ZEs, Ss = read.random_sample(W, X, ZA, ZB, ZE, S, I, setup, sample)

    start_time_th = time.time()
    Result = opt.minimize(dlp.neglogl, thetahat, args = (Ws, Xs, ZAs, ZBs, ZEs, Ss, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
    print("--- Step " + str(t) + " took %s seconds ---" % (time.time() - start_time_th))

    return Result.x, sample

T = 500
if __name__ == '__main__': # "protect my main loop" (https://pythonhosted.org/joblib/parallel.html) but also useful on mac (and seems to indeed make it a bit faster)
    BResults = Parallel(n_jobs = 4)(delayed(bootstrap)(t) for t in range(T))
    Thetahat = np.array([item[0] for item in BResults]).T
    thetahatsd = np.std(Thetahat,1)
    np.save("../results/Thetahat_SAP_" + str(T) + ".npy",Thetahat)

"--- Output: ---"
np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
np.set_printoptions(suppress=True)
tdim = np.size(theta0)
print reduce(np.append, [range(tdim), theta0, thetahat, thetahatsd]).reshape(-1,tdim).T

thetatable = reduce(np.append, [theta0, thetahat, thetahatsd]).reshape(-1,tdim).T
latextable = ""
for entry, line in zip(tnames, thetatable): # just remove the "+ entry" to have no parameter names
    latextable = latextable + entry + " & " + " & ".join(map('{0:.3f}'.format, line)) + " \\\\\n"

with open("../tab/Tan_3-4_SAP_boot" + str(T) + ".tex", "w") as txt_file:
    txt_file.write(latextable)


#np.append(range(T),Thetahat[0,:]).reshape(-1,T).T
#print reduce(np.append, [range(tdim), theta0, thetahat, Thetahat[:,-1]]).reshape(-1,tdim).T

" Diagnose dlp"
#dlp.neglogl(thetahat8, W, X, ZA, ZB, ZE, S, setup)
#Mu = dlp.make_mu(thetahat8, X, ZA, ZB, ZE, S, setup)
#print(Mu)
#Rho, Eta, Alpha1, Alpha2, Alpha3, Beta = dlp.make_indexes(thetahat8, ZA, ZB, ZE, setup)
#min(Eta)
#"------------------"

" Numerical Gradients "
#H = ndiff.hessian(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
#paramcov = np.linalg.inv(H)
#thetasd = np.sqrt(np.diag(paramcov))
#
#"Test"
##thetahat = np.load("thetahat.npy")
#testgrad = ndiff.gradient(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
#print testgrad
#testhessian = ndiff.hessian(dlp.neglogl, thetahat, 1e-6, (W, X, ZA, ZB, ZE, S, setup))
#print testhessian
#"----"

"--- Starting Value Analysis ---"
#def startvaltest(thetahat, startingcov, S, tol):
#    tdim = np.size(thetahat)
#    Thetas = np.zeros((tdim,S))
#    startingcov = paramcov/10
#    for s in range(S):
#        thetainit = np.random.multivariate_normal(theta0,startingcov)
#        Thetas[:,s] = opt.fmin_powell(dlp.neglogl, thetainit, args = (W, X, Z, ZE, setup), xtol=tol, ftol=tol, maxfun = 1e5, maxiter = 1e5)
#    return Thetas

#Thetas = startvaltest(thetahat, paramcov/10, 10, tol)

" Trying different tolerances "
#start_time_th = time.time()
#tol = 1e-5
#res5 = opt.minimize(dlp.neglogl, theta0, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = #{'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat5 = res5.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

#start_time_th = time.time()
#tol = 1e-6
#res6 = opt.minimize(dlp.neglogl, thetahat5, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = #{'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat6 = res6.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

#start_time_th = time.time()
#tol = 1e-7
#res7 = opt.minimize(dlp.neglogl, thetahat6, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = #{'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat7 = res7.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

#start_time_th = time.time()
#tol = 1e-8
#res8 = opt.minimize(dlp.neglogl, thetahat7, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = #{'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat8 = res8.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

" Testing convergence in the boostrap "

## Question: will it converge starting in thetahat - yes, just as well as starting from theta0
## will it generally be closer to thetahat than the true min? - inconclusive, looks like it for some entries, not for others

#np.random.seed(None) # this way each worker will get a new seed (https://github.com/joblib/joblib/issues/36)
#sample = np.random.choice(range(setup[0]),setup[0], replace = True) # sample with replacement
#
#Ws = W[sample,:]
#Xs = X[sample]
#ZAs = ZA[sample,:]
#ZBs = ZB[sample,:]
#ZEs = ZE[sample,:]
#Ss = S[sample]
#
#start_time_th = time.time()
#tol = 1e-6
#res6 = opt.minimize(dlp.neglogl, thetahat, args = (Ws, Xs, ZAs, ZBs, ZEs, Ss, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat6 = res6.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))
#
#start_time_th = time.time()
#tol = 1e-7
#res7 = opt.minimize(dlp.neglogl, thetahat, args = (Ws, Xs, ZAs, ZBs, ZEs, Ss, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat7 = res7.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))
#
#start_time_th = time.time()
#tol = 1e-8
#res8 = opt.minimize(dlp.neglogl, thetahat, args = (Ws, Xs, ZAs, ZBs, ZEs, Ss, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
#thetahat8 = res8.x
#print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))
#
#np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
#np.set_printoptions(suppress=True)
#tdim = np.size(theta0)
#print reduce(np.append, [range(tdim), theta0, thetahat, thetahat6, thetahat7, thetahat8]).reshape(-1,tdim).T
