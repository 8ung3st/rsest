import pandas as pd
import numpy as np
import dlp

def prepareZ(Z, normmean, normstd):
    N = Z.shape[0]
    Z += -normmean
    Z /= normstd
    return np.concatenate((np.ones((N,1)), Z), axis=1)

def normvectors(Z):
    return np.mean(Z, axis = 0), np.std(Z, axis = 0)

def rep(var,name):
    return np.tile(name,np.asarray(var.shape))

def tentries(theta, setup, names):
    tentries = [] # initialize list
    params = dlp.tunpack(theta, setup)
    for i in range(np.size(params)-1):
        tentries.append(rep(params[i],names[i]))
    tentries.append(rep(params[-1], names[-1]))
    tentries.append(setup) # appending setup, which we need for tpack
    return dlp.tpack(*tentries)

def getdata(path):
    "--- Data: ---"
    Data = pd.read_stata(path)

    W  = np.asarray(Data[['w1', 'w2', 'w3']])
    X  = np.asarray(Data['lexp'])

    Z  = np.asarray(Data[['age_h','age_w','nkids']])
    meanZ, stdZ = normvectors(Z)
    Z = prepareZ(Z, meanZ, stdZ)

    ZE = np.asarray(Data[['age_h','age_w','nkids']])
    meanZE, stdZE = normvectors(ZE)
    ZE = prepareZ(ZE, meanZE, stdZE)

    "--- Setup: ---"
    N, NI  = W.shape # number of hh, number of hh members
    PP  = Z.shape[1]-1 # number of explanatory variables (accross all equations) for preferences
    PE  = ZE.shape[1]-1 # number of explanatory variables for eta
    setup = [N,NI,PP,PE] # model description

    "--- Initial Values: ---"
    names = ["rho   ", "eta   ", "alpha1", "alpha2", "alpha3", "beta  ", "Sig   "]
    rho0 = np.append(.5, np.zeros(PE)) # father's share in parents' share
    eta0 = np.append(.3, np.zeros(PE)) # childrens' share
    alpha10 = np.append(2, np.zeros(PP))
    alpha20 = np.append(2, np.zeros(PP))
    alpha30 = np.append(2, np.zeros(PP))
    beta0 = np.append(.01, np.zeros(PP)) # common slope
    cov0 = (np.identity(NI)+.1*np.ones((NI,NI)))*.001 # covariance matrix
    theta0 = dlp.tpack(rho0, eta0, alpha10, alpha20, alpha30, beta0, cov0, setup) # parameter vector
    tnames = tentries(theta0, setup, names)

    return theta0, W, X, Z, ZE, setup, tnames, meanZ, stdZ
