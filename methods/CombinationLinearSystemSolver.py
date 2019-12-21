import numpy as np

from itertools import chain, combinations
from functools import reduce

from methods.utils import get_algebraic_complement

class CombinationLinearSystemSolver(object):
    
    __name__='CombinationLinearSystemSolver'
    
    def __init__(self, det_eps=1e-7, combinations_to_take=-1):
        self.solution = None
        self.delta = None
        self.error = None
        self.max_error_div_by_eps = None
        self.min_error_div_by_eps = None
        
        self.det_eps = 1e-7
        self.combinations_to_take = combinations_to_take
        
    def get_solo_solution_delta(self, matrix, vector, epselon_vector):
        algebric_complement = get_algebraic_complement(matrix)
        solution = vector@algebric_complement / np.linalg.det(matrix)
        delta = epselon_vector@np.abs(algebric_complement) / abs(np.linalg.det(matrix))
        return solution, delta
    
    def fit(self, slar_object):
        index_combinations = list(combinations(range(slar_object.m), slar_object.n))
        
        combinations_to_take = min(len(index_combinations), self.combinations_to_take) if self.combinations_to_take>0 else len(index_combinations)
        index_combinations = sorted(index_combinations, key=lambda idx: -np.abs(np.linalg.det(slar_object.A_matrix[idx,:])))
        index_combinations = index_combinations[:combinations_to_take]
        
        left_ends = []
        right_ends = []
        for idx in index_combinations:
            idx = list(idx)
            if np.linalg.det(slar_object.A_matrix[idx,:]) > self.det_eps:
                solution, delta = self.get_solo_solution_delta(slar_object.A_matrix[idx,:], slar_object.noised_b[idx], slar_object.random_noise[idx])
                left_ends.append(solution - delta)
                right_ends.append(solution + delta)
                
        left_ends, right_ends = np.array(left_ends), np.array(right_ends)
        
        x_min = np.max(left_ends, axis=0)
        x_max = np.min(right_ends, axis=0)
        
        self.solution = (x_min + x_max)/2
        self.delta = abs((x_max - x_min)/2)
        
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)
        abs_scaled_error = np.abs(self.solution - slar_object.x_vector) / slar_object.noise_epselon
        self.min_error_div_by_eps = min(abs_scaled_error)
        self.max_error_div_by_eps = max(abs_scaled_error)