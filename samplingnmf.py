import os
from itertools import chain
import numpy as np
import tensorflow as tf


class SamplingNMF(object):

    def __init__(self, K=8, batch_size=32, learning_rate=0.1):
        """NMF based on stochastic gradient descent
        """
        self.K = K
        self.batch_size = batch_size
        self.lr = learning_rate

    def fit(self, indices, weights=None):
        assert isinstance(indices, list)
        indices.sort(key=lambda x: (x[0],x[1]))
        n_nodes = [None, None]
        n_nodes[0] = max([i for i, j in indices]) + 1
        n_nodes[1] = max([j for i, j in indices]) + 1
        self.n_nodes = n_nodes
        self.indices = indices
        self.n_elem = len(indices)
        if weights:
            weights = np.array(weights)
            assert len(indices) == weights.size
            self.weights = weights.astype(np.float32)
        else:
            self.weights = np.ones(len(indices), dtype="float32")
        self.sum_weights = self.weights.sum()
        self._build_graph()
        self._enqueue_batch(indices, weights)
        self._train()


    def _train(self):
        import pdb; pdb.set_trace()
        loss, _ = self.sess.run([self.loss, self.opt_op])

    def _enqueue_batch(self, indices, weights):
        sess = self.sess
        indices = np.array(indices, dtype=np.int64)
        w_ind = indices[:,0]
        h_ind = indices[:,1]
        sess.run(self.enqueue, feed_dict={self.inp_w_ind: w_ind,
                                          self.inp_h_ind: h_ind,
                                          self.inp_weights: weights})

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            indices = self.indices
            weights = self.weights
            nn1, nn2 = self.n_nodes
            K = self.K
            batch_size = np.int64(self.batch_size)
            sum_weights = self.sum_weights
            scale_w = 2 * np.sqrt(sum_weights / (K * nn1 * nn1))
            scale_h = 2 * np.sqrt(sum_weights / (K * nn2 * nn2))
            
            init_w = tf.random_uniform_initializer(0, scale_w)
            init_h = tf.random_uniform_initializer(0, scale_h)

            self.W = W = tf.get_variable("W", shape=[nn1, K],
                                        initializer=init_w)
            self.H = H = tf.get_variable("H", shape=[nn2, K],
                                        initializer=init_h)

            self.queue = queue = tf.RandomShuffleQueue(capacity=100000000, min_after_dequeue=0,
                                                            dtypes=["int64","int64","float32"],
                                                            shapes=[[],[],[]])
            self.inp_w_ind = inp_w_ind = tf.placeholder("int64", [None])
            self.inp_h_ind = inp_h_ind = tf.placeholder("int64", [None])
            self.inp_weights = inp_weights = tf.placeholder("float32", [None])

            self.enqueue = queue.enqueue_many((inp_w_ind, inp_h_ind, inp_weights)) 
            w_indices, h_indices, weights = queue.dequeue_many(batch_size)

            #gather blocks of W and H
            W_b = tf.gather(W, w_indices, name="W_block")
            H_b = tf.gather(H, h_indices, name="H_block")

            W_abs = tf.abs(W_b)
            H_abs = tf.abs(H_b)

            x_ind = tf.transpose(tf.pack([w_indices, h_indices]))

            import pdb; pdb.set_trace()
            sp_X = tf.SparseTensor(indices=x_ind, values=weights,
                                shape=[batch_size, batch_size])
            X = tf.sparse_tensor_to_dense(sp_X)
            
            Y = tf.matmul(W_abs, H_abs, transpose_b=True)
            self.loss = loss = tf.nn.l2_loss(X - Y)

            grad_w, grad_h = tf.gradients(loss, [W_b, H_b])

            apply_grad_w = tf.scatter_sub(W, w_indices, grad_w)
            apply_grad_h = tf.scatter_sub(H, h_indices, grad_h)

            self.opt_op = tf.group(apply_grad_w, apply_grad_h)

            self.sess = tf.Session()
            init_op = tf.initialize_all_variables()
            self.sess.run(init_op)


if __name__=="__main__":
    nmf = SamplingNMF()
    indices = [(np.random.randint(0,100), np.random.randint(0,100)) for _ in range(1000)]
    weights = [1.0] * 1000
    nmf.fit(indices, weights)





