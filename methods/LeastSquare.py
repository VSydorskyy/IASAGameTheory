import numpy as np

from itertools import chain, combinations
from functools import reduce

from methods.utils import get_algebraic_complement

class LeastSquare(object):
    
    __name__ = 'LSTM'
    
    def __init__(self):
        self.solution = None
        self.delta = None
        self.error = None
        self.A_T_mul_A = None
        
    def get_solution(self, slar_object):
        self.A_T_mul_A = slar_object.A_matrix.T @ slar_object.A_matrix
        self.solution = (np.linalg.inv(self.A_T_mul_A) @ slar_object.A_matrix.T) @ slar_object.noised_b
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)
    
    def get_delta(self, slar_object):
        alg_matrix = get_algebraic_complement(self.A_T_mul_A)
        a_t_ksi = slar_object.A_matrix.T @ (np.ones(slar_object.A_matrix.shape[0]) * slar_object.noise_epselon * slar_object.A_matrix.shape[0])     
        self.delta = np.array([ alg_matrix[:,i] @ a_t_ksi for i in range(alg_matrix.shape[1])]) / np.linalg.det(self.A_T_mul_A)
        self.delta = np.abs(self.delta)

    def fit(self, slar_object):
        self.get_solution(slar_object)
        self.get_delta(slar_object)