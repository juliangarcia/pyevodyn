'''
Created on Oct 5, 2012

@author: garcia
'''
import unittest
from pyevodyn import games, numerical
from pyevodyn.simulation import MoranProcess

class Test(unittest.TestCase):


    def test_hardcoded_fixations(self):
        game= games.two_times_two(a=3.0, b=0.5, c=0.5, d=2.0)
        intensity_of_selection=1.0
        population_size=5
        #number_of_samples=100000
        number_of_samples=20000
        index_of_the_incumbent=1
        index_of_the_mutant=0
        numerical_value =  numerical.fixation_probability_strategy_a(game, intensity_of_selection, population_size)
        print numerical_value
        mp = MoranProcess(population_size, intensity_of_selection, game_matrix=game, fitness_mapping='exp', mutation_probability=0.1)
        simulated_value = mp.simulate_fixation_probability(index_of_the_incumbent=index_of_the_incumbent, index_of_the_mutant=index_of_the_mutant, number_of_samples=number_of_samples, seed=None)
        print simulated_value
        self.assertAlmostEqual(numerical_value, simulated_value, places=2, msg="", delta=None)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()