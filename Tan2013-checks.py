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
theta0, W, X, Xhat, Z, ZE, setup, tnames, meanZ, stdZ = read.getdata(path) # the artificial data is set up in dlp_art_data.py

goods = ["men's clothing", "women's clothing", "children's clothing", "food in", "vices", "food out", "utilities", "hhexpenses", "health", "transport", "communication", "recreation", "education", "other"]

import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.nonparametric.kernel_regression as kreg

"replace X with instrumented value Xhat"
X = Xhat

"-----------------"

def datafit(W,X,name,title,Z=np.ones(0)):
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
        if Z.size!=0:
            Ex = [X, Z]
        else:
            Ex = X
        model = kreg.KernelReg(endog=W.T[i], exog = Ex, var_type='c')
        sm_mean, sm_mfx = model.fit()
        ax[i].plot( X[order], sm_mean[order], '-r', alpha=1)
        ax[i].set_ylabel(goods[i])
    f.savefig(name + '.png')

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

def datafit2(W,X,name,title):
    ngoods = W.shape[1]
    f, (ax) = plt.subplots(ngoods, sharex=True, sharey=False)
    f.set_size_inches(20.,100.)
    ax[0].set_title(title)
    for i in range(ngoods):
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
             item.set_fontsize(20)
        w = W.T[i]
        x = X[w>0]
        order = np.argsort(x)
        ax[i].plot( x, w[w>0], ',k', alpha=1)
        model = kreg.KernelReg(endog=w[w>0], exog=x, var_type='c')
        sm_mean, sm_mfx = model.fit()
        ax[i].plot( x[order], sm_mean[order], '-r', alpha=1)
        ax[i].set_ylabel(goods[i])
    f.savefig(name + '.png')

def justfit2(W,X,name,title):
    ngoods = W.shape[1]
    f, (ax) = plt.subplots(ngoods, sharex=True, sharey=False)
    f.set_size_inches(20.,50.)
    ax[0].set_title(title)
    for i in range(ngoods):
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
             item.set_fontsize(20)
        w = W.T[i]
        x = X[w>0]
        order = np.argsort(x)
        model = kreg.KernelReg(endog=w[w>0], exog=x, var_type='c')
        sm_mean, sm_mfx = model.fit()
        ax[i].plot( x[order], sm_mean[order], '-r', alpha=1)
        ax[i].set_ylabel(goods[i])
    f.savefig(name + '.png')

"-----------------"
datafit(W,X,"BS_Tan2013","Budget Shares", Z=Z.T[3])
justfit(W,X,"BSfit_Tan2013","Budget Shares - Fitted")

datafit2(W,X,"BS_Tan2013_positive","Budget Shares")
justfit2(W,X,"BSfit_Tan2013_positive","Budget Shares - Fitted")
