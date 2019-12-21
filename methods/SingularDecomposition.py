import numpy as np

from itertools import chain, combinations
from functools import reduce

class SingularDecomposition(object):
    
    __name__ = 'SVD'
    
    def __init__(self):
        self.solution = None
        self.delta = None
        self.error = None
        self.max_error_div_by_eps = None
        self.min_error_div_by_eps = None
        
    def fit(self, slar_object):
        U, Z, Vt = np.linalg.svd(slar_object.A_matrix)
        W = (Vt.T @ np.linalg.inv(np.diag(Z)[:slar_object.A_matrix.shape[1], :])) @ U.T[:slar_object.A_matrix.shape[1], :]
        self.solution = W @ slar_object.noised_b 
        self.delta = np.abs(W) @ (np.ones(slar_object.A_matrix.shape[0]) * slar_object.noise_epselon * slar_object.A_matrix.shape[0])
        self.delta = np.abs(self.delta)
        
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)
        abs_scaled_error = np.abs(self.solution - slar_object.x_vector) / slar_object.noise_epselon
        self.min_error_div_by_eps = min(abs_scaled_error)
        self.max_error_div_by_eps = max(abs_scaled_error)