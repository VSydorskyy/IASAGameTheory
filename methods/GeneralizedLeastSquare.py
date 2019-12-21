import numpy as np
import pandas as pd
import hyperopt as hp

                      
class GeneralizedLeastSquare(object):
    
    __name__ = 'GLSTM'
    
    def __init__(self, genetic_alg_steps=1000):
        self.genetic_alg_steps = genetic_alg_steps
        
        self.solution = None
        self.delta = None
        self.error = None
        self.best_s = None
        self.max_error_div_by_eps = None
        self.min_error_div_by_eps = None
        
    def get_solution(self, slar_object, weights):        
        A_matrix = slar_object.A_matrix
        b = slar_object.noised_b
        
        A_matrix = A_matrix * np.expand_dims(weights,axis=-1)
        b = b * weights
            
        A_T_mul_A = A_matrix.T @ A_matrix
        self.solution = (np.linalg.inv(A_T_mul_A) @ A_matrix.T) @ b
        
        self.error = np.linalg.norm(self.solution - slar_object.x_vector)
        abs_scaled_error = np.abs(self.solution - slar_object.x_vector) / slar_object.noise_epselon
        self.min_error_div_by_eps = min(abs_scaled_error)
        self.max_error_div_by_eps = max(abs_scaled_error)
    
    def get_delta(self, slar_object, weights):        
        A_matrix = slar_object.A_matrix
        b = slar_object.noised_b
        
        A_matrix = A_matrix * np.expand_dims(weights,axis=-1)
        b = b * weights
        
        A_T_mul_A = A_matrix.T @ A_matrix
        self.delta = (slar_object.A_matrix.shape[0] * 
                      np.sum(np.abs(A_T_mul_A @ A_matrix.T / np.linalg.det(A_T_mul_A)), axis=-1) * 
                      slar_object.noise_epselon)

    def fit(self, slar_object):
        self.unconvex_optimization(slar_object) 

    
    def unconvex_optimization(self, slar_object):  
        
        results = {'alpha':[],'error':[], 'solution':[], 'delta': [], 'max_delta':[]}
        def objective(x):
            a_s = np.array(x)
            
            try:
                self.get_solution(slar_object, weights=a_s)
                self.get_delta(slar_object, weights=a_s)
                results['alpha'].append(a_s)
                results['error'].append(self.error)
                results['solution'].append(self.solution)
                results['delta'].append(self.delta)
                results['max_delta'].append(max(self.delta))
                r = max(self.delta)
            except:
                r = 100
                   
            return r
        
        space = [hp.hp.uniform(label=str(i), low=0, high=1) for i in range(slar_object.A_matrix.shape[0])]

        best = hp.fmin(objective, space, algo=hp.tpe.suggest, max_evals=self.genetic_alg_steps, show_progressbar=False)

        results = pd.DataFrame(results)
        results = results.sort_values('max_delta').iloc[0]
        
        self.best_s = results['alpha']
        self.error = results['error']
        self.solution = results['solution']
        self.delta = results['delta'] 
