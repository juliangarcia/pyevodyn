'''
Created on Jan 31, 2013

@author: garcia
'''
import unittest
from pyevodyn import numerical, games


class Test(unittest.TestCase):


    def testFixation(self):
        intensity_of_selection=0.1
        population_size =50
        game_matrix = games.prisoners_dilemma_equal_gains()
        fix =  numerical.fixation_probability(mutant_index=0, resident_index=1, intensity_of_selection=intensity_of_selection, payoff_function=None, population_size=population_size, game_matrix=game_matrix, 
                                       number_of_strategies=None, mapping='EXP')
        self.assertAlmostEqual(fix, 0.0006059821672805591)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFixation']
    unittest.main()