import pandas as pd
import numpy as np
import dlp_sat2 as dlp

def prepareZ(Z, normmean, normstd):
    N = Z.shape[0]
    Z = (Z-normmean)
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

    W  = np.asarray(Data[['w_clothm', 'w_clothw', 'w_clothc', 'w_FBout_h', 'w_FBout_w', 'w_FBout_c']])
    X  = np.asarray(Data['ltotR'])
    X -= np.mean(X)
    S  = np.asarray(Data['nkids'])
    I  = np.asarray(Data['lassets']) # instrument

    ZA  = np.asarray(Data[['kids_2', 'kids_3', 'kids_4', 'k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death', 'res_1', 'res_2']])
    meanZA, stdZA = normvectors(ZA)
    ZA = prepareZ(ZA, meanZA, stdZA)

    #ZB  = np.asarray(Data[[ 'k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death']])
    ZB  = np.asarray(Data[['urban']])
    meanZB, stdZB = normvectors(ZB)
    ZB = prepareZ(ZB, meanZB, stdZB)

    #ZE = np.asarray(Data[['k_minage', 'k_meanage', 'sgirls', 'edu_h', 'edu_w', 'age_h', 'age_w', 'urban', 'wave4', 'death']])
    ZE = np.asarray(Data[['kids_2', 'kids_3', 'kids_4']])
    meanZE, stdZE = normvectors(ZE)
    ZE = prepareZ(ZE, meanZE, stdZE)

    "--- Setup: ---"
    N, NI  = W.shape # number of hh, number of equations
    PA  = ZA.shape[1]-1 # number of explanatory variables (accross all equations) for alpha (including control fct)
    PB  = ZB.shape[1]-1 # number of explanatory variables (accross all equations) for beta
    PE  = ZE.shape[1]-1 # number of explanatory variables for eta
    setup = [N,NI,PA,PB,PE] # model description

    "--- Initial Values: ---"
    names = ["rho", "eta", "alpha^1", "alpha^2", "alpha^3", "alpha^4", "alpha^5", "alpha^6", "beta^1", "beta^2", "beta^3", "beta^4", "beta^5", "beta^6", "Sigma"]
    rho0 = np.append(.5, np.zeros(PE)) # father's share in parents' share
    eta0 = np.append(.3, np.zeros(PE)) # childrens' share
    alpha10 = np.append(np.mean(W,0)[0], np.zeros(PA)) # initialized at mean budget shares
    alpha20 = np.append(np.mean(W,0)[1], np.zeros(PA))
    alpha30 = np.append(np.mean(W,0)[2], np.zeros(PA))
    alpha40 = np.append(np.mean(W,0)[3], np.zeros(PA))
    alpha50 = np.append(np.mean(W,0)[4], np.zeros(PA))
    alpha60 = np.append(np.mean(W,0)[5], np.zeros(PA))
    beta10 = np.append(.01, np.zeros(PB))
    beta20 = np.append(.01, np.zeros(PB))
    beta30 = np.append(.01, np.zeros(PB))
    beta40 = np.append(.01, np.zeros(PB))
    beta50 = np.append(.01, np.zeros(PB))
    beta60 = np.append(.01, np.zeros(PB))
    cov0 = (np.identity(NI)+.1*np.ones((NI,NI)))*.001 # covariance matrix
    theta0 = dlp.tpack(rho0, eta0, alpha10, alpha20, alpha30, alpha40, alpha50, alpha60, beta10, beta20, beta30, beta40, beta50, beta60, cov0, setup) # parameter vector
    tnames = tentries(theta0, setup, names)

    return theta0, W, X, ZA, ZB, ZE, S, I, setup, tnames, meanZE, stdZE

"--- Helper functions for bootstrap ---"

def select(W, X, ZA, ZB, ZE, S, I, sample):
    s = sample
    return W[s,:], X[s], ZA[s,:], ZB[s,:], ZE[s,:], S[s], I[s]

def standardize(M):
    meanM, stdM = normvectors(M)
    M = prepareZ(M, meanM, stdM)
    return M

# update ZA with residuals, doesn't change anything if used on full data
def first_stage(X,ZA,I,setup):
    N,NI,PA,PB,PE = setup
    I2 = np.stack((I,I**2), axis=1)
    meanI2, stdI2 = normvectors(I2)
    I2_ones = prepareZ(I2, meanI2, stdI2) # this adds ones at the first column
    Xf = np.concatenate((ZA[:,:-2],I2_ones[:,1:]), axis=1)
    beta = np.dot(np.linalg.inv(np.dot(Xf.T, Xf)), np.dot(Xf.T,X)) # X is the dependent variable here
    res = X - np.dot(beta,Xf.T) # same as from stata
    res2 = np.stack((res,res**2), axis=1)
    return np.concatenate((ZA[:,:-2],res2), axis=1)

def random_sample(W, X, ZA, ZB, ZE, S, I, setup, sample):
    Ws, Xs, ZAs, ZBs, ZEs, Ss, Is = select(W, X, ZA, ZB, ZE, S, I, sample) # note: means, std not
    " First Stage: "
    ZAs = first_stage(Xs,ZAs,Is,setup) # update residuals in ZA

    ZAs = standardize(ZAs[:,1:]) # standardize covariates incl residuals
    ZBs = standardize(ZBs[:,1:])
    ZEs = standardize(ZEs[:,1:])
    Xs -= np.mean(Xs)                   # demean X

    Ws = np.float32(Ws)
    Xs = np.float32(Xs)
    ZAs = np.float32(ZAs)
    ZBs = np.float32(ZBs)
    ZEs = np.float32(ZEs)
    Ss = np.float32(Ss)
    return Ws, Xs, ZAs, ZBs, ZEs, Ss
