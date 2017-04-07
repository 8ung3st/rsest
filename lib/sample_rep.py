# sample takes a list of matrices of equal length (dim1) and samples row-wise from all of them jointly
# sample returns a list of matrices of the same dimensions

import numpy as np
import pandas as pd

def sample(list):
    N = list[0].shape[0]
    K = len(list)
    if any(e.shape[0] != N for e in list):
        raise ValueError('All matrices must have equal length')
    values = np.zeros(K)
    vector = list[0].T.flatten()
    for i in range(K):
        values[i] = int(list[i].shape[1])
        if i>0:
            vector = np.append(vector,list[i].T)
    mat = vector.reshape((int(np.sum(values)),N)).T
    Data = pd.DataFrame(mat)
    Datas = Data.sample(n=N, replace=True) # random sample with replacement
    mats = Datas.as_matrix()
    mark = 0
    lists = []
    for i in range(K):
        lists.append(mats[:,mark:mark+int(values[i])])
        mark = mark + int(values[i])
    return lists
