import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/estimation/rsest/")
import numpy as np
import lib.read_dta as read
import lib.dlp as dlp
import lib.ndiff as ndiff
from __future__ import division
np.random.seed(1) # set the seed

"--- Load Data ---"
path = "../../data/Tanzania 2013/SampleDLP.dta"
theta0, W, X, Z, ZE, setup, tnames, meanZ, stdZ = read.getdata(path) # the artificial data is set up in dlp_art_data.py

goods = ["men's clothing", "women's clothing", "children's clothing", "food in", "vices", "food out", "utilities", "hhexpenses", "health", "transport", "communication", "recreation", "education", "other"]

import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.nonparametric.kernel_regression as kreg

def datafit(W,X,name,title):
    ngoods = W.shape[1]
    order = np.argsort(X)
    f, (ax) = plt.subplots(ngoods, sharex=True, sharey=False)
    f.set_size_inches(20.,100.)
    ax[0].set_title(title)
    for i in range(ngoods):
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
             item.set_fontsize(20)
        ax[i].plot( X, W.T[i], ',k', alpha=1)
        model = kreg.KernelReg(endog=W.T[i], exog=X, var_type='c')
        sm_mean, sm_mfx = model.fit()
        ax[i].plot( X[order], sm_mean[order], '-r', alpha=1)
        ax[i].set_ylabel(goods[i])
    f.savefig(name + '.png')

datafit(W,X,"BS_Tan2013","Budget Shares")

def justfit(W,X,name,title):
    ngoods = W.shape[1]
    order = np.argsort(X)
    f, (ax) = plt.subplots(ngoods, sharex=True, sharey=False)
    f.set_size_inches(20.,50.)
    ax[0].set_title(title)
    for i in range(ngoods):
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
             item.set_fontsize(20)
        model = kreg.KernelReg(endog=W.T[i], exog=X, var_type='c')
        sm_mean, sm_mfx = model.fit()
        ax[i].plot( X[order], sm_mean[order], '-r', alpha=1)
        ax[i].set_ylabel(goods[i])
    f.savefig(name + '.png')

justfit(W,X,"BSfit_Tan2013","Budget Shares - Fitted")
