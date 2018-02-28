"--- This works ---"

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

Data[['wavecode', 'w_clothm', 'ltotR']].head()

w_data = Data[Data.wavecode == "Mal_4"]

w_clothm = w_data.w_clothm
w_ltotR = w_data.ltotR

with pm.Model() as wave_model:
    # Intercept prior
    a = pm.Normal('alpha', mu=0, sd=10)
    # Slope prior
    b = pm.Normal('beta', mu=0, sd=10)

    # Model error prior
    eps = pm.HalfCauchy('eps', beta=2)

    # Linear model
    w_est = a + b * w_ltotR

    # Data likelihood
    y_like = pm.Normal('y_like', mu=w_est, sd=eps, observed=w_clothm)

with wave_model:
    # Inference button (TM)!
    trace = pm.sample(draws=5000, tune=1000, njobs = 1)

pm.traceplot(trace)
