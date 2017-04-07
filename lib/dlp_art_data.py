# This file simulates data 'W, X, Z, ZE', describes it in 'setup'
# and gives the true values 'theta0' to be used as a starting value

import numpy as np
import dlp

def get():
    "--- Setup: ---"
    N  = 200 # number of hh
    NI = 3 # number of hh members
    PP  = 2 # number of explanatory variables (accross all equations) for preferences
    PE  = 3 # number of explanatory variables for eta
    setup = [N,NI,PP,PE] # model description

    "--- True Values: ---"
    rho0 = np.append(.5,np.random.normal(0,.04,PE)) # father's share in parents' share
    eta0 = np.append(.3,np.random.normal(0,.02,PE)) # childrens' share
    alpha10 = np.append(1,np.random.normal(0,.05,PP))
    alpha20 = np.append(1,np.random.normal(0,.05,PP))
    alpha30 = np.append(1,np.random.normal(0,.05,PP))
    beta0 = np.append(.1,np.random.normal(0,.05,PP)) # common slope
    cov0 = (np.identity(NI)+.1*np.ones((NI,NI)))*.001 # covariance matrix
    theta0 = dlp.tpack(rho0, eta0, alpha10, alpha20, alpha30, beta0, cov0, setup) # parameter vector

    "--- Generate Data ---"
    X  = np.random.normal(3,1,N) # log hh income
    Z = np.concatenate((np.ones((N,1)), np.random.randn(N, PP)), axis=1) # influence husbands
    ZE  = np.concatenate((np.ones((N,1)), np.random.randn(N, PE-PP), Z[:,1:]), axis=1) # influence eta: Here I take the last covariate for each of the members to see what happens
    W = dlp.simulate(theta0, X, Z, ZE, setup)
    return theta0, W, X, Z, ZE, setup
