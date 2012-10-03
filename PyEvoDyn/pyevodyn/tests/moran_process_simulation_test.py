'''
Created on Oct 3, 2012

@author: garcia
'''
import unittest
import pyevodyn.simulation as sim
from pyevodyn import games
import numpy as np

class Test(unittest.TestCase):

    def setUp(self):
        pass


    def tearDown(self):
        pass

    def mock_up_payoff_function_everybody_gets_one(self, population_array):
        return np.ones(len(population_array))

    
    def test_custom_fitnes(self):
        mp = sim.MoranProcess(population_size=10, intensity_of_selection=0.001, 
                              game_matrix=None, payoff_function=self.mock_up_payoff_function_everybody_gets_one, number_of_strategies=5, 
                              fitness_mapping='lin', mutation_probability=0.001)
        pop = np.array([1,2,3,4,5])
        np.testing.assert_array_equal(mp.payoff_function(pop), np.ones(len(pop)), "Custom function payoff failed")
        
    def test_game_fitness(self):
        mp = sim.MoranProcess(population_size=10, intensity_of_selection=0.001, 
                              game_matrix=games.neutral_game(2), payoff_function=None, number_of_strategies=2, 
                              fitness_mapping='lin', mutation_probability=0.001)
        pop = np.array([5,5])
        np.testing.assert_array_equal(mp.payoff_function(pop), np.ones(len(pop)), "Neutral game test function payoff failed")
        mp = sim.MoranProcess(population_size=10, intensity_of_selection=0.001, 
                              game_matrix=np.zeros(shape=(2,2)), payoff_function=None, number_of_strategies=2, 
                              fitness_mapping='lin', mutation_probability=0.001)
        pop = np.array([5,5])
        np.testing.assert_array_equal(mp.payoff_function(pop), np.zeros(len(pop)), "Neutral game test function payoff failed")
    
    def test_step_invariable_in_population_size(self):
        for _ in xrange(0,10):
            #random game 2x2, random mutation rate, random intensity
            mp = sim.MoranProcess(population_size=10, intensity_of_selection=np.random.rand(), 
                              game_matrix=np.random.rand(2,2), number_of_strategies=2, 
                              fitness_mapping='exp', mutation_probability=np.random.rand())
            str1 = np.random.randint(0, 10)
            pop = [str1, 10-str1]
            for __ in xrange(0,10):
                pop = mp.step(pop, mutation_step=np.random.randint(0,2))[0]
                #print pop
                self.assertEqual(sum(pop), 10, "Pop size should always be ten!")
        

    def test_initialization(self):
        try:
            sim.MoranProcess(population_size=10, intensity_of_selection=0.05, 
                              game_matrix=None, payoff_function=None, 
                              number_of_strategies=None, fitness_mapping='exp', mutation_probability=0.001, mutation_kernel=None)
        except ValueError, err:
            print 'Error ',err, ' tested OK.'
            self.assertTrue(True,'No exception raised!')
            
        try:
            sim.MoranProcess(population_size=10, intensity_of_selection=0.05, 
                              game_matrix=None, payoff_function=self.mock_up_payoff_function_everybody_gets_one, 
                              number_of_strategies=None, fitness_mapping='exp', mutation_probability=None, mutation_kernel=None)
        except ValueError, err:
            print 'Error ',err, ' tested OK.'
            self.assertTrue(True,'No exception raised!')
        try:
            sim.MoranProcess(population_size=10, intensity_of_selection=0.05, 
                              game_matrix=np.ones(5), payoff_function=None, 
                              number_of_strategies=None, fitness_mapping='exp', mutation_probability=0.01, mutation_kernel=None)
        except ValueError, err:
            print 'Error ',err, ' tested OK.'
            self.assertTrue(True,'No exception raised!')
        try:
            sim.MoranProcess(population_size=10, intensity_of_selection=0.05, 
                              game_matrix=None, payoff_function=self.mock_up_payoff_function_everybody_gets_one, 
                              number_of_strategies=None, fitness_mapping='exp', mutation_probability=0.01, mutation_kernel=None)
        except ValueError, err:
            print 'Error ',err, ' tested OK.'
            self.assertTrue(True,'No exception raised!')
        try:
            sim.MoranProcess(population_size=10, intensity_of_selection=0.05, 
                              game_matrix=None, payoff_function=self.mock_up_payoff_function_everybody_gets_one, 
                              number_of_strategies=5, fitness_mapping='hola', mutation_probability=0.01, mutation_kernel=None)
        except ValueError, err:
            print 'Error ',err, ' tested OK.'
            self.assertTrue(True,'No exception raised!')
            return
        self.assertTrue(False, 'No exception raised')
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()