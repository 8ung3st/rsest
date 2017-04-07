import pandas as pd
import sample_rep

"--- Bootstrap: ---"
S = 30
Thetahat = np.zeros((theta0.shape[0],S))
for s in range(S):
    Ws, Xs, Zs, ZEs = sample_rep.sample([W, X, Z, ZE])
    Thetahat[:,s] = fmin(dlp.neglogl, theta0, args = (Ws, Xs, Zs, ZEs, setup), maxfun = 100000, maxiter = 100000)
    print "Iteration " + str(s+1) + " terminated"


"--- Output: ---"
np.set_printoptions(precision=4) # so numbers are displayed with this degree of precision
np.set_printoptions(suppress=True)
tdim = np.size(theta0)
print reduce(np.append, [range(tdim),theta0, theta1, np.std(Thetahat,axis=1), thetasd]).reshape(-1,tdim).T
