import pandas as pd
import numpy as np

def prepareZ(Z, normmean, normstd):
    N = Z.shape[0]
    Z += -normmean
    Z /= normstd
    return np.concatenate((np.ones((N,1)), Z), axis=1)

def normvectors(Z):
    return np.mean(Z, axis = 0), np.std(Z, axis = 0)

def getdata(path):
    bsharesC = ['bsfoodpq', 'bsvicesq', 'bshouspq', 'bsmclothq', 'bswclothq', 'bstranspq', 'bshealcareq', 'bsedureanq', 'bsentertpq']
    bshares1 = ['bsfoodpq', 'bsvicesq', 'bshouspq', 'bsmclothq', 'bstranspq', 'bshealcareq', 'bsedureanq', 'bsentertpq']
    bshares2 = ['bsfoodpq', 'bsvicesq', 'bshouspq', 'bswclothq', 'bstranspq', 'bshealcareq', 'bsedureanq', 'bsentertpq']

    "--- Couples' Data ---"
    DataC = pd.read_stata(path+ "couples.dta")

    WC = np.asarray(DataC[bsharesC])# leaves out "other"
    XC = np.asarray(np.log(DataC['totexpq']))
    ZC1 = np.asarray(DataC[['ownhous', 'owncar', 'age_h', 'age_h2', 'educlassm', 'earn_h']])
    ZC2 = np.asarray(DataC[['ownhous', 'owncar', 'age_w', 'age_w2', 'educlassf', 'earn_w']])
    # standardize with respect to the population of married individuals:
    normmean, normstd = normvectors(np.concatenate((ZC1,ZC2),axis=0))
    ZC1 = prepareZ(ZC1, normmean, normstd)
    ZC2 = prepareZ(ZC2, normmean, normstd)

    ZD1 = np.concatenate((ZC1,np.asarray(DataC[['onekid','twokids']])),axis=1)
    ZD2 = np.concatenate((ZC2,np.asarray(DataC[['onekid','twokids']])),axis=1)

    ZE = np.asarray(DataC[['ownhous', 'owncar', 'age_h', 'age_h2', 'educlassm', 'age_w', 'age_w2', 'educlassf', 'incshare']])
    ZEmirror = np.asarray(DataC[['ownhous', 'owncar', 'age_w', 'age_w2', 'educlassf', 'age_h', 'age_h2', 'educlassm', 'incshare']])
    normmeanE, normstdE = normvectors(np.concatenate((ZE,ZEmirror),axis=0))
    ZE = prepareZ(ZE, normmeanE, normstdE)
    ZE = np.concatenate((ZE,np.asarray(DataC[['onekid','twokids']])),axis=1)


    "--- Singles' Data ---"
    DataS = pd.read_stata(path+ "singles.dta")

    Data1 = DataS[(DataS.sex == 1)] # men
    Data2 = DataS[(DataS.sex == 2)] # women

    W1 = np.asarray(Data1[bshares1])# leaves out "other"
    X1 = np.asarray(np.log(Data1['totexpq']))
    Z1 = prepareZ(np.asarray(Data1[['ownhous', 'owncar', 'age', 'age2', 'educlass', 'earn']]), normmean, normstd)

    W2 = np.asarray(Data2[bshares2])# leaves out "other"
    X2 = np.asarray(np.log(Data2['totexpq']))
    Z2 = prepareZ(np.asarray(Data2[['ownhous', 'owncar', 'age', 'age2', 'educlass', 'earn']]), normmean, normstd)

    return theta0, W, X, Z, ZE, setup
