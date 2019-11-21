import numpy as np

from itertools import chain, combinations
from functools import reduce

class SystemLinearEquations(object):
    
    def __init__(self, x_vector, m, n, epselon, random_noise, cond=1):
        self.LIMIT = 4
        self.m = m
        self.n = n
        self.x_vector = x_vector
        self.A_matrix = self.genarate_matrix(n,m)*np.expand_dims(np.array([cond] + [1 for i in range(m-1)]),axis=-1)
        self.initial_b = self.A_matrix @ self.x_vector
        self.random_noise = random_noise
        self.noise_epselon = epselon
        self.noised_b = self.generate_noise(self.initial_b)
        
    def genarate_matrix(self, n, m):
        return np.random.uniform(low=-self.LIMIT, high=self.LIMIT, size=(m,n))
    
    def generate_noise(self, vector):
        self.random_noise = (self.random_noise / np.linalg.norm(self.random_noise)) * self.noise_epselon
        return vector + self.random_noise