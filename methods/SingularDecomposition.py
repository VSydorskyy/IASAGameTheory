import numpy as np

from itertools import chain, combinations
from functools import reduce

class SingularDecomposition(object):
    
    __name__ = 'SVD'
    
    def __init__(self):
        self.solution = None
        self.delta = None
        self.error = None
        
    def fit(self, slar_object):
        U, Z, Vt = np.linalg.svd(slar_object.A_matrix)
        W = (Vt.T @ np.linalg.inv(np.diag(Z)[:slar_object.A_matrix.shape[1], :])) @ U.T[:slar_object.A_matrix.shape[1], :]
        self.solution = W @ slar_object.noised_b 
        self.delta = np.abs(W) @ (np.ones(slar_object.A_matrix.shape[0]) * slar_object.noise_epselon * slar_object.A_matrix.shape[0])
        self.delta = np.abs(self.delta)
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)