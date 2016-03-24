import os
import numpy as np
import tensorflow as tf

class RWSNMF(object):

    def __init__(self, K=8, L=10):
        self.K = K
        self.L = L


