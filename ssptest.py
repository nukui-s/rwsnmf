#conding: utf-8
import numpy as np
import tensorflow as tf
import scipy.sparse as ssp

if __name__=='__main__':
    indices = [(1,2),(1,3),(2,3),(0,3),(0,1)]
    values = np.random.rand(len(indices))
    spmat = ssp.lil_matrix((5,5))
    ind2 = np.array(indices)
    spmat[ind2[:,0],ind2[:,1]] = values
    ind = np.array([1,2,3])
    sliced = spmat[ind,:][:,ind]
    nnz = sliced.nonzero()
    vals = sliced[nnz].toarray()[0]
    inp_ind = np.array(nnz).T
    print(inp_ind)
    print(vals)

    tfmat = tf.SparseTensor(inp_ind, vals, shape=[3,3])
    sess = tf.InteractiveSession()

