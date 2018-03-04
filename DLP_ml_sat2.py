import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/4. Estimation/rsest/")
import numpy as np
import lib.read_sat2 as read
import lib.dlp_sat2 as dlp
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
Result = opt.minimize(dlp.neglogl, theta0, args = (W, X, ZA, ZB, ZE, S, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True}) # this is very slow, 5-10 minutes per estimation round
thetahat = Result.x
print("--- Optimization took %s seconds ---" % (time.time() - start_time_th))

"--- Bootstrap: ---"
from joblib import Parallel, delayed

def bootstrap(t):
    np.random.seed(None) # this way each worker will get a new seed (https://github.com/joblib/joblib/issues/36)
    sample = np.random.choice(range(setup[0]),setup[0], replace = True) # sample with replacement
    Ws, Xs, ZAs, ZBs, ZEs, Ss = read.random_sample(W, X, ZA, ZB, ZE, S, I, setup, sample)

    dlp.neglogl(thetahat, Ws, Xs, ZAs, ZBs, ZEs, Ss, setup)

    start_time_th = time.time()
    Result = opt.minimize(dlp.neglogl, thetahat, args = (Ws, Xs, ZAs, ZBs, ZEs, Ss, setup), method='Powell', options = {'ftol':tol, 'maxfev':1e6, 'disp':True})
    print("--- Step " + str(t) + " took %s seconds ---" % (time.time() - start_time_th))

    return Result.x, sample

T = 100
if __name__ == '__main__': # "protect my main loop" (https://pythonhosted.org/joblib/parallel.html) but also useful on mac (and seems to indeed make it a bit faster)
    BResults = Parallel(n_jobs = 1)(delayed(bootstrap)(t) for t in range(T))
    Thetahat = np.array([item[0] for item in BResults]).T
    thetahatsd = np.std(Thetahat,1)
    np.save("../results/Thetahat_SAT2_" + str(T) + ".npy",Thetahat)

"--- Output: ---"
np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
np.set_printoptions(suppress=True)
tdim = np.size(theta0)
print reduce(np.append, [range(tdim), theta0, thetahat, thetahatsd]).reshape(-1,tdim).T

thetatable = reduce(np.append, [theta0, thetahat, thetahatsd]).reshape(-1,tdim).T
latextable = ""
for entry, line in zip(tnames, thetatable): # just remove the "+ entry" to have no parameter names
    latextable = latextable + entry + " & " + " & ".join(map('{0:.3f}'.format, line)) + " \\\\\n"

with open("../tab/Tan_3-4_SAT2_" + str(T) + ".tex", "w") as txt_file:
    txt_file.write(latextable)
