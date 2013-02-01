'''
Created on Jan 31, 2013

@author: garcia
'''
import unittest
import numpy as np
from pyevodyn import numerical, games


class Test(unittest.TestCase):


    def testFixation(self):
        intensity_of_selection=0.1
        population_size =50
        game_matrix = games.prisoners_dilemma_equal_gains()
        fix =  numerical.fixation_probability(mutant_index=0, resident_index=1, intensity_of_selection=intensity_of_selection, payoff_function=None, population_size=population_size, game_matrix=game_matrix, 
                                       number_of_strategies=None, mapping='EXP')
        self.assertAlmostEqual(fix, 0.0006059821672805591)
    
    def testMatrix(self):
        game_matrix = games.allc_tft_alld()
        population_size=50
        ios = 0.1
        mutation_probability = 0.001
        result=  numerical.monomorphous_transition_matrix(intensity_of_selection=ios, mutation_probability=mutation_probability, population_size=population_size, game_matrix=game_matrix, number_of_strategies=None, mapping='EXP')
        expected = np.array([[  9.99939406e-01,   1.00000000e-05,   5.05937414e-05],[1.00000000e-05,   9.99989700e-01,   3.00468293e-07],[2.79103633e-07,2.21444512e-05,9.99977576e-01]])
        np.testing.assert_allclose(result, expected)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFixation']
    unittest.main()