'''
Created on Oct 3, 2012

@author: garcia
'''
import unittest
from pyevodyn import games
import numpy as np
from pyevodyn.simulation import MoranProcess
import pyevodyn.simulation as sim


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
        for _ in xrange(0,3):
            #random game 2x2, random mutation rate, random intensity
            mp = sim.MoranProcess(population_size=10, intensity_of_selection=np.random.rand(), 
                              game_matrix=np.random.rand(2,2), number_of_strategies=2, 
                              fitness_mapping='exp', mutation_probability=np.random.rand())
            str1 = np.random.randint(0, 10)
            pop = [str1, 10-str1]
            for __ in xrange(0,20):
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
        
        
    def test_if_a_type_is_not_there_it_never_shows_up(self):
        #np.random.seed(999)
        for i in xrange(0,5):
            pop = np.random.randint(1,10,5) #random population with 5 strategies
            zero_element = np.random.randint(0,5)
            pop[zero_element] = 0
            pop_size = np.sum(pop)
            mp = MoranProcess(population_size=pop_size, 
                              intensity_of_selection=1.0, game_matrix=np.random.rand(5,5),
                              number_of_strategies=5, fitness_mapping='exp', mutation_probability=0.1)
            for j in xrange(0,1000):
                pop = mp.step(pop, mutation_step=False)[0]
                self.assertEqual(pop[zero_element], 0, "Type " + str(zero_element) +" showed up in population "+ str(pop) +" at iteration "+ str(i)+" "+str(j))
                    
    def test_fixation_of_neutral_mutant(self):
        number_of_strategies_value = 2
        number_of_samples_=10000
        for _ in xrange(0,5):
            pop_size =np.random.randint(2,11)
            mp = MoranProcess(population_size=pop_size, 
                              intensity_of_selection=0.0, game_matrix=np.random.rand(number_of_strategies_value,number_of_strategies_value),
                              number_of_strategies=number_of_strategies_value, fitness_mapping='exp', mutation_probability=0.1)
            fix = mp.simulate_fixation_probability(0, 1, number_of_samples=number_of_samples_, seed=None)
            np.testing.assert_allclose(fix,1.0/pop_size, rtol=0.01, atol=0.01, err_msg="Paila", verbose=True)
            

        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    