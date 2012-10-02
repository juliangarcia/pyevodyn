'''
Created on Sep 28, 2012

@author: garcia
'''
import unittest
from pyevodyn import symbolic, numerical
from sympy.matrices.matrices import Matrix
import numpy as np

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_stationary_distribution(self):
        for _ in xrange(0,10):
            game_matrix_numeric = np.array([[np.random.randint(0,10), np.random.randint(0,10)],[ np.random.randint(0,10), np.random.randint(0,10)]])
            game_matrix_symbolic = Matrix(game_matrix_numeric.tolist()) 
            intensity_of_selection = 0.001
            mutation_probability = 0.001 
            population_size = 100 
            symbolic_result = symbolic.symbolic_matrix_to_array(symbolic.stationary_distribution_weak_selection(game_matrix_symbolic, intensity_of_selection, population_size, mutation_probability).T)
            numerical_result = np.array([numerical.stationary_distribution_weak_selection(game_matrix_numeric, population_size, intensity_of_selection, mutation_probability)])
            #print symbolic_result
            #print numerical_result
            np.testing.assert_allclose(symbolic_result, numerical_result, rtol=0.05, err_msg = "Numerical does not match symbolic", verbose=False)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_stationary_distribution']
    unittest.main()