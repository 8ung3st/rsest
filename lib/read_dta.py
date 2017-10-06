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

def enumerep(var,name):
    pnames = list(rep(var, name))
    for i in range(var.shape[0]):
        if var.ndim == 2:
            row = list(pnames[i])
            for j in range(var.shape[1]):
                row[j] = "$\\" + row[j] + "_{" + str(i) + "," + str(j) + "}$"
            pnames[i] = np.asarray(row)
        else:
            pnames[i] = "$\\" + pnames[i] + "_{" + str(i) + "}$"
    return np.asarray(pnames)

def tentries(theta, setup, names):
    tentries = [] # initialize list
    params = dlp.tunpack(theta, setup)
    for i in range(np.size(params)-1):
        tentries.append(enumerep(params[i],names[i]))
    tentries.append(enumerep(params[-1], names[-1]))
    tentries.append(setup) # appending setup, which we need for tpack
    return dlp.tpack(*tentries)

def getdata(path):
    "--- Data: ---"
    Data = pd.read_stata(path)

    W  = np.asarray(Data[['w_clothm', 'w_clothw', 'w_clothc']])
    X  = np.asarray(Data['ltotR'])
    S  = np.asarray(Data['nkids'])

    ZA  = np.asarray(Data[[ 'kids_2', 'kids_3', 'kids_4', 'k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death', 'res_1', 'res_2']])
    meanZA, stdZA = normvectors(ZA)
    ZA = prepareZ(ZA, meanZA, stdZA)

    ZB  = np.asarray(Data[[ 'kids_2', 'kids_3', 'kids_4', 'k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death']])
    meanZB, stdZB = normvectors(ZB)
    ZB = prepareZ(ZB, meanZB, stdZB)

    ZE = np.asarray(Data[[ 'kids_2', 'kids_3', 'kids_4', 'k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death']])
    meanZE, stdZE = normvectors(ZE)
    ZE = prepareZ(ZE, meanZE, stdZE)

    "--- Setup: ---"
    N, NI  = W.shape # number of hh, number of hh members
    PA  = ZA.shape[1]-1 # number of explanatory variables (accross all equations) for alpha (including control fct)
    PB  = ZB.shape[1]-1 # number of explanatory variables (accross all equations) for beta
    PE  = ZE.shape[1]-1 # number of explanatory variables for eta
    setup = [N,NI,PA,PB,PE] # model description

    "--- Initial Values: ---"
    names = ["rho", "eta", "alpha^1", "alpha^2", "alpha^3", "beta", "Sigma"]
    rho0 = np.append(.5, np.zeros(PE)) # father's share in parents' share
    eta0 = np.append(.3, np.zeros(PE)) # childrens' share
    alpha10 = np.append(2, np.zeros(PA))
    alpha20 = np.append(2, np.zeros(PA))
    alpha30 = np.append(2, np.zeros(PA))
    beta0 = np.append(.01, np.zeros(PB)) # common slope
    cov0 = (np.identity(NI)+.1*np.ones((NI,NI)))*.001 # covariance matrix
    theta0 = dlp.tpack(rho0, eta0, alpha10, alpha20, alpha30, beta0, cov0, setup) # parameter vector
    tnames = tentries(theta0, setup, names)

    return theta0, W, X, ZA, ZB, ZE, S, setup, tnames, meanZE, stdZE
