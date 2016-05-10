#conding: utf-8
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as ssp

class SamplingNetwork(object):

    def __init__(self, A):
        assert(isinstance(A, ssp.lil_matrix))
        assert(A.shape[0] == A.shape[1])
        self.n_nodes = A.shape[0]
        ind1, ind2 = A.nonzero()
        print("The number of nodes: " + str(self.n_nodes))
        print("The number of edges: " + str(ind1.size))
        vals = A[ind1, ind2].data.tolist()[0]
        self.degrees = degrees = A.sum(axis=0).tolist()[0]

        self.node_seq = np.array(degrees).argsort()
        self.if_covered = {n:False for n in range(self.n_nodes)}

        # neighbor nodes to node
        self.neighbors = {n: [] for n in range(self.n_nodes)}
        # thresholds for sampling next route
        #self.thresholds = {n: np.array([]) for n in range(self.n_nodes)}

        for i, j, v in zip(ind1, ind2, vals):
            #norm_v = v / max(degrees[i], 10e-12)
            self.neighbors[i].append(j)
            #t_list = self.thresholds[i]
            #self.thresholds[i] = np.append(self.thresholds[i],
                                           #t_list.sum() + norm_v)

    def get_start_node(self):
        while True:
            try:
                n = self.node_seq.pop(0)
            except:
                return None
            if self.if_covered[n]: continue
            if_covered[n] = True
            return n

    def select_node_randomly(self):
        return np.random.randint(self.n_nodes)

    def walk_to_neighbor(self, n, buf):
        # Removed nodes included in buf for next candidates
        n_set = set(self.neighbors[n]) - set(buf)
        if len(n_set) == 0:
            return self.get_start_node()
        next_ind = np.random.randint(len(n_set))
        n = list(n_set)[next_ind]
        self.if_covered[n] = True
        return n

if __name__=="__main__":
    A = ssp.lil_matrix((5,5))
    A[[1,2,3,0,1,2], [0,1,2,1,2,3]] = 1
    net = SamplingNetwork(A)

