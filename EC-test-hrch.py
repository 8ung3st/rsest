%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import seaborn as sns

import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/4. Estimation/rsest/")

"--- Read Data ---"
path = "../../3. Data/Combined_Sample.dta"
Data = pd.read_stata(path)

Data[['wavecode', 'w_clothm', 'ltotR']] = Data[['wavecode', 'w_clothm', 'ltotR']].round(4)

Data[['wavecode', 'w_clothm', 'ltotR']].head()

wave_idx = Data['wavecode']
clothm_mean = np.squeeze(np.mean(Data[['w_clothm']]))

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a = pm.Normal('mu_alpha', mu=clothm_mean, sd=10)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=2)
    mu_b = pm.Normal('mu_beta', mu=0., sd=10)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=2)

    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(Data.wavecode.unique()))
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(Data.wavecode.unique()))

    # Error standard deviation
    eps = pm.HalfCauchy('eps', beta=2)

    # Expected value
    w_est = a[wave_idx] + b[wave_idx] * Data.ltotR

    # Data likelihood
    w_like = pm.Normal('w_like', mu=w_est, sd=eps, observed=Data.w_clothm)

with hierarchical_model:
    hierarchical_trace = pm.sample(draws=5000, tune=1000, njobs=4)[1000:]

x = pd.Series(hierarchical_trace['sigma_alpha_log'], name='sigma alpha log')
y = pd.Series(hierarchical_trace['eps_log'], name='sigma eps log')

sns.jointplot(x, y);


pm.traceplot(hierarchical_trace)
