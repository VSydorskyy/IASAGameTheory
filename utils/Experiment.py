import numpy as np
import pandas as pd
import itertools

from itertools import chain, combinations
from functools import reduce
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.SystemLinearEquations import SystemLinearEquations

NECESSARY_PARAMS = ['M', 'EPSILONS', 'CONDS', "DISTRIBUTIONS"] 

class Experiment(object):
    def __init__(self, grid_params, n_dim, x_vector, models, matrix_abs_limit=4):
        
        if list(grid_params.keys()) != NECESSARY_PARAMS:
            raise ValueError('Unrecognaized grid param!')
            
        self.grid_params = grid_params
        self.n_dim = n_dim
        self.x_vector = x_vector
        self.models = models
        self.matrix_abs_limit = matrix_abs_limit
        
        self.results = None
        
    def run_experiment(self, monte_karlo_steps=1, verbose=False):
        
        experiment_results = {k:[] for k in self.grid_params.keys()}
        experiment_results['REAL_COND'] = []
        for model in self.models:
            experiment_results[model.__name__ + '_error'] = []
            experiment_results[model.__name__ + '_min_error_div_by_eps'] = []
            experiment_results[model.__name__ + '_max_error_div_by_eps'] = []
            
        number_of_itters = reduce(lambda x,y: x*y, [len(self.grid_params[k]) for k in self.grid_params.keys()])
        for m, eps, dist, c in tqdm(itertools.product(self.grid_params['M'], 
                                                      self.grid_params['EPSILONS'], 
                                                      self.grid_params['DISTRIBUTIONS'].keys(), 
                                                      self.grid_params['CONDS']), total=number_of_itters):
            
            accumulate_dict = {model.__name__:[] for model in self.models}

            for i in range(monte_karlo_steps):

                accumulate_dict = self.one_fit(m, eps, self.grid_params['DISTRIBUTIONS'][dist](m), c, accumulate_dict) 

            for model in self.models:
                accumulate_dict[model.__name__] = np.array(accumulate_dict[model.__name__]).mean(axis=0)

            if verbose:
                print('M: {}, EPSILON: {}, Distribution: {}, Cond: {}'.format(m, eps, dist, accumulate_dict[self.models[0].__name__][3]))
                for model in self.models:
                    print(model.__name__)
                    print(pd.DataFrame({'solution':accumulate_dict[model.__name__][0], 'delta':accumulate_dict[model.__name__][1], 'real_value':self.x_vector}))
                    print('residual: {}'.format(accumulate_dict[model.__name__][2]))
                    print('\n'*3)

            experiment_results['M'].append(m)
            experiment_results['EPSILONS'].append(eps)
            experiment_results['DISTRIBUTIONS'].append(dist)
            experiment_results['REAL_COND'].append(accumulate_dict[self.models[0].__name__][3])
            experiment_results['CONDS'].append(c)
            for model in self.models:
                experiment_results[model.__name__+'_error'].append(accumulate_dict[model.__name__][2])
                experiment_results[model.__name__+'_min_error_div_by_eps'].append(accumulate_dict[model.__name__][4])
                experiment_results[model.__name__+'_max_error_div_by_eps'].append(accumulate_dict[model.__name__][5])
            
        self.results = pd.DataFrame(experiment_results)
        
        return self.results
    
    def visualise_experiment(self, num_samples_to_visualise=3, colors=['yellow','green','red']):
        possible_params = set(self.grid_params.keys())
        for param in self.grid_params.keys():
            print('Plotting experiments with '+param)
            params_to_fix = list(possible_params - {param})
            fixed_params_values = {p:(np.random.choice(self.grid_params[p], size=num_samples_to_visualise) if p!='DISTRIBUTIONS' else 
                                      np.random.choice(list(self.grid_params[p].keys()), size=num_samples_to_visualise))  
                                   for p in params_to_fix}
            
            for i in range(num_samples_to_visualise):
                self.visualise_with_fixation(params_to_fix=list(fixed_params_values.keys()),
                                             fixed_values=[fixed_params_values[k][i] for k in fixed_params_values.keys()],
                                             param_to_visual=param,
                                             colors=colors)
                
            print('\n'*5)
            
                    
    def one_fit(self, m, eps, random_noise, c, result_dict):
        slar = SystemLinearEquations(x_vector=self.x_vector, 
                                     m=m, n=self.n_dim, epselon=eps, 
                                     random_noise=random_noise, cond=c, 
                                     matrix_abs_limit=self.matrix_abs_limit)
        for model in self.models:
            model.fit(slar)
            result_dict[model.__name__].append([model.solution, model.delta, model.error,  
                                                np.linalg.cond(slar.A_matrix),
                                                model.min_error_div_by_eps, model.max_error_div_by_eps])
            
        return result_dict
    
    def visualise_with_fixation(self, params_to_fix, fixed_values, param_to_visual, colors=['yellow','green','red']):
        case = [self.results[x]==y for x,y in zip(params_to_fix,fixed_values)]
        case = reduce(lambda x,y: x & y, case)

        self.visualise_certain_error('error', params_to_fix, fixed_values, param_to_visual, case, colors)
        self.visualise_certain_error('min_error_div_by_eps', params_to_fix, fixed_values, param_to_visual, case, colors)
        self.visualise_certain_error('max_error_div_by_eps', params_to_fix, fixed_values, param_to_visual, case, colors)

        
        
    def visualise_certain_error(self, error_name, params_to_fix, fixed_values, param_to_visual, case, colors):
        plt.figure(figsize=(10,5))
        plt.title('; '.join([str(x) + ' = ' + str(y) for x, y in zip(params_to_fix, fixed_values)]))
        plt.ylabel(error_name)
        plt.xlabel(param_to_visual)
        for model, color in zip(self.models,colors):
            plt.plot(self.results.loc[case, param_to_visual],
                     self.results.loc[case, model.__name__ + '_' + error_name], 
                                      label=model.__name__ + '_' + error_name, color=color)

        plt.legend(loc='upper left')
        plt.show()
        
        