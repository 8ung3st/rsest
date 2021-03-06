# This module contains the funcitons needed to compute the DLP likelihood from parameters and data.
# Flexibility: Covariates in shares and in tastes can be varied independently.
# Limitations: Only does the linear SAP. Covariates in alpha and beta must be the same (also accross individuals)

import numpy as np
import theano.tensor as t
from theano import function

"-----------------------------------------------------------"

def tpack(rho, eta, alpha1, alpha2, alpha3, beta, Sig, setup):
    NI = setup[1]
    return reduce(np.append, [rho, eta, alpha1, alpha2, alpha3, beta, Sig[np.tril_indices(NI)]])

def extract(theta, mark, step):
    return mark+step, theta[mark:mark+step]

def tunpack(theta, setup):
    N,NI,PA,PB,PE = setup
    mark = 0 # tracks where we are in the vector
    mark, rho = extract(theta, mark, 1+PE)
    mark, eta = extract(theta, mark, 1+PE)
    mark, alpha1 = extract(theta, mark, 1+PA)
    mark, alpha2 = extract(theta, mark, 1+PA)
    mark, alpha3 = extract(theta, mark, 1+PA)
    mark, beta = extract(theta, mark, 1+PB)

    Sig = np.zeros((NI,NI))
    mark, Sig[np.tril_indices(NI)] = extract(theta, mark, NI*(NI+1)/2)
    Sig = Sig + Sig.T - t.diag(t.diag(Sig))
    return rho, eta, alpha1, alpha2, alpha3, beta, Sig

"-----------------------------------------------------------"

def make_indexes(theta, ZA, ZB, ZE, setup):
    rho, eta, alpha1, alpha2, alpha3, beta, Sig = tunpack(theta, setup)
    return t.dot(ZE,rho), t.dot(ZE,eta), t.dot(ZA,alpha1), t.dot(ZA,alpha2), t.dot(ZA,alpha3), t.dot(ZB,beta)

def pwp(m1,m2):
    return m1 * m2

def make_mu(theta, X, ZA, ZB, ZE, S, setup):
    N,NI = setup[:2]
    Rho, Eta, Alpha1, Alpha2, Alpha3, Beta = make_indexes(theta, ZA, ZB, ZE, setup)
    mu1 = pwp(pwp(Rho,(1-Eta))       , Alpha1 + reduce(pwp,[Beta, np.log(pwp(Rho,(1-Eta))     ), X]))
    mu2 = pwp(pwp((1-Rho),(1-Eta))   , Alpha2 + reduce(pwp,[Beta, np.log(pwp((1-Rho),(1-Eta)) ), X]))
    mu3 = pwp(Eta                    , Alpha3 + reduce(pwp,[Beta, np.log(np.divide(Eta,S)     ), X]))
    return reduce(np.append,[mu1,mu2,mu3]).reshape(NI,N).T

def add_errors(mu, Sig, setup):
    N,NI = setup[:2]
    epsilon = np.random.multivariate_normal(np.zeros(NI),Sig,N)
    return mu + epsilon

def simulate(theta, X, Z, ZE, setup):
    Sig = tunpack(theta, setup)[-1] # gives the last output
    mu = make_mu(theta, X, Z, ZE, setup)
    return add_errors(mu, Sig, setup)

"-----------------------------------------------------------"

def logl_normal(Y, mu, Sig):
    N = Y.shape[0]
    invSig = t.nlinalg.MatrixInverse(Sig)
    gradmu = t.dot(invSig,(Y-mu).T)
    ldetSig = t.log(t.nlinalg.Det(Sig))
    logl = - N/2 * ldetSig - .5 * np.sum((Y-mu) * gradmu.T)
    return logl

#def sharecheck(theta, ZA, ZB, ZE, setup):
#    Rho, Eta = make_indexes(theta, ZA, ZB, ZE, setup)[:2]
#    return all(0 < R < 1 for R in Rho) and all(0 < E < 1 for E in Eta)

def logl(theta, W, X, ZA, ZB, ZE, S, setup):
    Sig = tunpack(theta, setup)[-1] # gives the last output
    if all(t.nlinalg.Eig(Sig) > 0): #and sharecheck(theta, ZA, ZB, ZE, setup):
        mu = make_mu(theta, X, ZA, ZB, ZE, S, setup)
        logl = logl_normal(W,mu,Sig)
        return logl
    else:
        return -np.inf

def neglogl(theta, W, X, ZA, ZB, ZE, S, setup):
    return -logl(theta, W, X, ZA, ZB, ZE, S, setup)
