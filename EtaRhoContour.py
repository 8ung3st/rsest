# this script takes a bootstrap sample as input to build a contour plot of the distribution of the estimates
import os
os.chdir("/Users/alexwolf/Dropbox/Uni/WB RS Estimates/4. Estimation/rsest/")
import numpy as np

T = 100 # sample size
name = "SAT2_" + str(T)
Thetahat = np.load("../results/Thetahat_" + name + ".npy")

print(Thetahat[:,0])

Rho0 = Thetahat[0,:]
Eta0 = Thetahat[4,:]

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

fig = sns.jointplot(x = Thetahat[4,:], y = Thetahat[0,:], ylim=(0, 1), xlim=(0, .5), kind='kde')
fig.set_axis_labels("$eta_3$", "$rho$")
plt.savefig("../img/RhoEta_" + name + ".png", dpi=100)


#sns.kdeplot(Rho0,Eta0, shade=True, cmap="Blues")
