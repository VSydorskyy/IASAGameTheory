import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from itertools import chain, combinations
from functools import reduce

from methods.utils import get_algebraic_complement

def normalize_coeffs(alpha):
    return alpha * alpha.shape[1] / alpha.sum(axis=1).reshape(alpha.shape[0], -1)

def mutation(parent, dev=0.1):
    child = parent + np.random.normal(loc=0, scale=dev, size=parent.shape)
    child = np.maximum(child, 0)
    child = np.minimum(child, 1)
    child = normalize_coeffs(child)
    return child

def generate_random_population(size, m, dev=0.3):
    return np.random.normal(loc=1, scale=dev, size=(size, m))

def add_random_elements(population, m, dev=0.3):
    return np.concatenate(
        (population, generate_random_population(size=size-population.shape[0], m=m, dev=dev)), axis=0)
                      
def evaluate_fitness(population, fitness_function):
    fitness_values = np.apply_along_axis(func1d=fitness_function, arr=population, axis=1)
    return fitness_values
                      
def get_best_element(population, fitness_function):
    return sort_by_fitness(population, fitness_function)[-1]
                      
def sort_by_fitness(population, fitness_function):
    fitness_order = np.argsort(evaluate_fitness(population, fitness_function))
    population = population[fitness_order]
    return population
                      

def selection(population, fitness_function, parents_coeff = 0.6, dev=0.1):
    amount = population.shape[0]
    parents_amount = int(round(amount * parents_coeff))
    population = sort_by_fitness(population, fitness_function)
    parents = population[:parents_amount]
    children = mutation(parents, dev=dev)
    population = np.concatenate((population, children))
    population = sort_by_fitness(population, fitness_function)[:amount]
    return population
                      
                      
def optimize(m, fitness_function, iteration_amount=10, output_period=10):

    size = 10

    population = generate_random_population(size, m)

    for iter_ind in range(iteration_amount):
        
        population = selection(population, fitness_function, dev = 0.1 * (1 - iter_ind / 2 / iteration_amount) )
        best_element = get_best_element(population, fitness_function)
        best_value = fitness_function(best_element)
                      


class GeneralizedLeastSquare(object):
    
    __name__ = 'GLSTM'
    
    def __init__(self):
        self.solution = None
        self.delta = None
        self.error = None
        self.A_T_mul_A = None
        self.best_s = None
        
    def get_solution(self, slar_object, weights):
        alhpa_diag = np.diag(weights)
        
        A_matrix = slar_object.A_matrix
        b = slar_object.noised_b
        
        A_matrix = alhpa_diag @ A_matrix
        b = alhpa_diag @ b

            
        A_T_mul_A = A_matrix.T @ A_matrix
        self.solution = (np.linalg.inv(A_T_mul_A) @ A_matrix.T) @ b
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)
    
    def get_delta(self, slar_object, weights):
        alhpa_diag = np.diag(weights)
        
        A_matrix = slar_object.A_matrix
        b = slar_object.noised_b
        
        A_matrix = alhpa_diag @ A_matrix
        b = alhpa_diag @ b
        
        A_T_mul_A = A_matrix.T @ A_matrix
        self.delta = slar_object.A_matrix.shape[0] * np.sum(np.abs(A_T_mul_A @ A_matrix.T / np.linalg.det(A_T_mul_A)), axis=-1) * slar_object.noise_epselon

    def fit(self, slar_object):
        self.unconvex_optimization(slar_object) 

    
    def unconvex_optimization(self, slar_object):  
        
        results = {'alpha':[],'error':[], 'solution':[], 'delta': []}
        def objective(x):
            a_s = np.array(x)
            
            try:
                self.get_solution(slar_object, weights=a_s)
                self.get_delta(slar_object, weights=a_s)
                results['alpha'].append(a_s)
                results['error'].append(self.error)
                results['solution'].append(self.solution)
                results['delta'].append(self.delta)
                r = max(self.delta)
            except:
                r = 100
                   
            return r

        best = optimize(slar_object.A_matrix.shape[0], objective)
        results = pd.DataFrame(results)
        
        self.best_s = results.sort_values('error')['alpha'].iloc[0]
        self.error = results.sort_values('error')['error'].iloc[0]
        self.solution = results.sort_values('error')['solution'].iloc[0]
        self.delta = results.sort_values('error')['delta'].iloc[0]   
