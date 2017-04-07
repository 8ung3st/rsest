# each function below computes numerical derivative with uniform step size delta in directions theta
# input a funciton handle f of a funciton which takes as arguments theta and *args
import numpy as np

def gradient(f, theta, delta, args):
    gradient = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        thetaplus = np.copy(theta)
        thetaplus[i] = theta[i]+delta
        gradient[i] = (f(thetaplus, *args)-f(theta, *args))/delta
    return gradient

def hessian(f, theta, delta, args):
    H = np.zeros((theta.shape[0],theta.shape[0]))
    for i in range(theta.shape[0]):
        thetaplus = np.copy(theta)
        thetaplus[i] = theta[i]+delta
        H[:,i] = (gradient(f, thetaplus, delta, args) - gradient(f, theta, delta, args))/delta
    return H
