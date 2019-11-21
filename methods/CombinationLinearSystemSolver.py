import numpy as np

from itertools import chain, combinations
from functools import reduce

from methods.utils import get_algebraic_complement

class CombinationLinearSystemSolver(object):
    def __init__(self, det_eps=1e-7):
        self.solution = None
        self.delta = None
        self.error = None
        self.det_eps = 1e-7
        
    def get_solo_solution(self, matrix, vector):
        algebric_complement = get_algebraic_complement(matrix)
        return vector@algebric_complement / np.linalg.det(matrix)
    
    def get_solo_interval(self, matrix, epselon_vector):
        algebric_complement = get_algebraic_complement(matrix)
        return epselon_vector@np.abs(algebric_complement) / abs(np.linalg.det(matrix))
    
    def fit(self, slar_object):
        index_combinations = combinations(range(slar_object.m), slar_object.n)
        
        left_ends = []
        right_ends = []
        for idx in index_combinations:
            idx = list(idx)
            if np.linalg.det(slar_object.A_matrix[idx,:]) > self.det_eps:
                solution = self.get_solo_solution(slar_object.A_matrix[idx,:], slar_object.noised_b[idx])
                delta = self.get_solo_interval(slar_object.A_matrix[idx,:], slar_object.random_noise[idx])
                left_ends.append(solution - delta)
                right_ends.append(solution + delta)
                
        left_ends, right_ends = np.array(left_ends), np.array(right_ends)
        
        x_min = np.max(left_ends, axis=0)
        x_max = np.min(right_ends, axis=0)
        
        self.solution = (x_min + x_max)/2
        self.delta = (x_max - x_min)/2
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)