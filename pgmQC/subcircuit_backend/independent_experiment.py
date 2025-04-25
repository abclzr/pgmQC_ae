from pgmQC.subcircuit_backend.corelated_random_variables import RandomVariable
import numpy as np

class IndependentExperiment:
    exp_counter = 0

    def __init__(self, input_state, output_state, prob):
        self.input_state = input_state
        self.output_state = output_state
        self.prob = prob
        IndependentExperiment.exp_counter += 1
        self.exp_id = IndependentExperiment.exp_counter
        self.random_variable_list = []
        self.covariance = {}
    
    def add_random_variable(self, random_variable : RandomVariable):
        self.random_variable_list.append(random_variable)
    
    
    def deal_with_covariance(self, covariance_matrix):
        for i in range(len(self.random_variable_list)):
            rv1 = self.random_variable_list[i]            
            index1 = rv1.identifier_to_index()
            covariance_matrix[index1, index1] = rv1.variance
            for j in range(i+1, len(self.random_variable_list)):
                rv2 = self.random_variable_list[j]
                # Copilot gives the following equation for covariance, and it's correct.
                # It comes from Cov(X, Y) = E[XY] - E[X]E[Y]
                cov = np.sum(rv1.eigenvalues * rv2.eigenvalues * self.prob) - rv1.mean * rv2.mean
                self.covariance[(rv1.identifier, rv2.identifier)] = cov
                
                index2 = rv2.identifier_to_index()
                covariance_matrix[index1, index2] = cov
                covariance_matrix[index2, index1] = cov