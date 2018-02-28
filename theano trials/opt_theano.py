import numpy as np
import theano.tensor as t
from theano import function

x = t.dvector('x')
def thfun(x):
    return t.dot(x,x)
y = thfun(x)

f = function([x],y)
f([3,2,4,5,6,7,8,9])

def fct(x):
    return f(x)
fct([3,2,4,5,6,7,8,9])

def fct2(x):
    return np.dot(x,x)
fct2([3,2,4,5,6,7,8,9])

import scipy.optimize as opt
import time

start_time_th = time.time()
opt.fmin(fct, [3,2,4,5,6,7,8,9])
print("--- %s seconds ---" % (time.time() - start_time_th))

start_time_np = time.time()
opt.fmin(fct2, [3,2,4,5,6,7,8,9])
print("--- %s seconds ---" % (time.time() - start_time_np))
