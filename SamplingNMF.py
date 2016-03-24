import os
from itertools import chain
import numpy as np
import tensorflow as tf


class SamplingNMF(object):

    def __init__(self, K=8, batch_size=32):
        """NMF based on stochastic gradient descent
        """
        self.K = K
        self.batch_size = batch_size

    def fit(self, indices, values=None):
        assert isinstance(indices, list)
        n_nodes = [None, None]
        n_nodes[0] = max([i for i, j in indices]) + 1
        n_nodes[1] = max([j for i, j in indices]) + 1
        self.n_nodes
        self.indices
        self.n_elem = len(indices)
        if values:
            values = np.array(values)
            assert len(indices, values.size)
            self.values = values
        else:
            self.values = np.ones(len(indices))
       self.sum_weights = self.values.sum()
       self._build_graph()
   

    def _build_graph(self):
        indices = self.indices
        values = self.values
        nn1, nn2 = self.n_nodes
        K = self.K
        batch_size = self.batch_size
        sum_weights = self.sum_weights
        scale_w = 2 * np.sqrt(sum_weights / (K * nn1 * nn1))
        scale_h = 2 * np.sqrt(sum_weights / (K * nn2 * nn2))
        
        init_w = tf.random_uniform_initializer(0, scale_w)
        init_h = tf.random_uniform_initializer(0, scale_h)

        self.W = W = tf.get_variable("W", shape=[nn1, K],
                                     initializer=init_w)
        self.H = H = tf.get_variable("H", shape=[nn2, K],
                                     initializer=init_h)

        self.queue = queue = tf.train.RandomShuffleQueue(100000000, 1000,
                                                        dtypes=["int","int","float"],
                                                        shapes=[[1],[1],[1]])
        self.input_w_ind = tf.placeholder("int", [batch_size])
        self.input_h_ind = tf.placeholder("int", [batch_size])
        self.input_val = tf.placeholoder("float", [batch_size])

        self.enqueue = queue.enqueue_many(
        w_ind, h_ind, val = queue.dequeue()

        w = W[w_ind, :]
        h = H[h_ind, :]

        dot = tf.reduce_sum(w * h)

        loss = dot - val

        grad_w, grad_h = tf.gradients(loss, [w, h])
        
        clipped_grad_w = tf.maximum(grad_w, w)
        clipped_grad_h = tf.maximum(grad_h, h)

        self.apply_grad_w = tf.scatter_sub(W, [w_ind], [clipped_grad_w])
        self.apply_grad_h = tf.scatter_sub(H, [h_ind], [clipped_grad_h])




