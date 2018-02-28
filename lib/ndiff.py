# each function below computes numerical derivative with uniform step size delta in directions theta
# input a funciton handle f of a funciton which takes as arguments theta and *args

# IMPORTANT: The hessian will be computed even if theta is a corner (or rather edge) solution. To this end steps in up and down directions are tried for each dimension. This may not be what you want!!!
import numpy as np

def gradient(f, theta, delta, args):
    gradient = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        thetaplus  = np.copy(theta)
        thetaminus = np.copy(theta)
        thetaplus[i]  = theta[i]+delta
        thetaminus[i] = theta[i]-delta
        gradientp = (f(thetaplus, *args)- f(theta, *args))      /delta
        gradientm = (f(theta, *args)-     f(thetaminus, *args)) /delta
        if np.isinf(gradientp):
            gradientp = np.nan
        if np.isinf(gradientm):
            gradientm = np.nan
        gradient[i] = np.nanmean([gradientp, gradientm])
    return gradient

def hessian(f, theta, delta, args):
    H = np.zeros((theta.shape[0],theta.shape[0]))
    for i in range(theta.shape[0]):
        thetaplus  = np.copy(theta)
        thetaminus = np.copy(theta)
        thetaplus[i]  = theta[i]+delta
        thetaminus[i] = theta[i]-delta
        H[:,i] = (gradient(f, thetaplus, delta, args) - gradient(f, theta, delta, args))/delta
    return H
