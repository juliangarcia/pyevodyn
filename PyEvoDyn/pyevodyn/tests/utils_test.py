'''
Created on Sep 19, 2012

@author: garcia
'''
import unittest
import numpy as np
import pyevodyn.utils as utils

class TestUtils(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass

    def test_submatrix(self):
        mock = np.array([[1,2,3],[4,5,6],[7,8,9]])
        np.testing.assert_equal(utils.submaxtrix(mock,0,1), np.array([[1,2],[4,5]]))
        np.testing.assert_equal(utils.submaxtrix(mock,0,2), np.array([[1,3],[7,9]]))
        np.testing.assert_equal(utils.submaxtrix(mock,1,2), np.array([[5,6],[8,9]]))

    def test_uniform_mutation_kernel(self):
        #check for 10 random kernels that all rows sump up to 1
        for _ in xrange(0,10):
            mutation_kernel = utils.uniform_mutation_kernel(np.random.rand(), np.random.randint(2, 10))
            rows = mutation_kernel.shape[0]
            columns = mutation_kernel.shape[1]
            self.assertEqual(rows, columns, "Rows and columns must have same size")
            sum_rows = np.array([np.sum(mutation_kernel[i,:]) for i in xrange(0,rows)])
            np.testing.assert_allclose(sum_rows, np.ones(columns), rtol=0.001, err_msg="Rows do not sum up to 1. in " + str(mutation_kernel))
            
    def test_normalize_vector(self):
        #check for 10 random vector that all sum up to 1 when normalized
        for _ in xrange(0,10):
            random_vector = np.random.rand(10) 
            np.testing.assert_almost_equal(np.sum(utils.normalize_vector(random_vector)), 1,decimal=10,  err_msg="Normalized vectors should sum up to be one") 
     
     
    def test_kahan_sum(self):
        for _ in xrange(0,10):
            random_vector = np.random.rand(10) 
            np.testing.assert_almost_equal(np.sum(random_vector), np.sum(random_vector),decimal=10,  err_msg="Kahan sum fails") 
    
    def test_kahan_product(self):
        np.testing.assert_almost_equal(utils.kahan_product([2,3]), 2*3,decimal=10,  err_msg="Kahan sum fails") 
        
    def test_binomial_coefficient(self):
        self.assertEqual(utils.binomial_coefficient(0, 1),0.0)
        self.assertEqual(utils.binomial_coefficient(-1, 1),-1.0)
        self.assertEqual(utils.binomial_coefficient(-1, -1),1.0)
        #TODO: check these two cases
        #self.assertEqual(utils.binomial_coefficient(-1, 10),1.0)
        #self.assertEqual(utils.binomial_coefficient(-1, -10),-1.0)
        self.assertEqual(utils.binomial_coefficient(-1, 0),1.0)
        self.assertEqual(utils.binomial_coefficient(-2, -1),0.0)
        self.assertEqual(utils.binomial_coefficient(-2, -2),1.0)
    
    def test_hypergeometric(self):
        #TODO: do it
        self.assertTrue(True, "TODO")
        
    
    def test_simulate_discrite_distribution(self):
        self.assertEqual(utils.simulate_discrete_distribution([1.0, 0.0]), 0, "Simulating singleton distribution fails") 
        self.assertEqual(utils.simulate_discrete_distribution([0.0, 1.0]), 1, "Simulating singleton distribution fails")
        coin = [0.5, 0.5]
        coin_tossing = np.array([utils.simulate_discrete_distribution(coin) for _ in xrange(0,500000)])
        sample = np.mean(coin_tossing)
        self.assertAlmostEqual(0.5, sample, delta=0.01, msg="Coin tossing test fails " + str(sample) +"  is not 0.5" )
        
    def test_random_edge_population(self):
        for _ in xrange(0,10):
            pop_size = np.random.randint(2,50)
            number_of_strategies = np.random.randint(2,10)
            ans = utils.random_edge_population(number_of_strategies, pop_size)
            self.assertEqual(np.sum(ans), pop_size, "Random edge should sum up to pop size")
            self.assertEqual(len(ans), number_of_strategies, "Testing number of srtategies") 
            
    def test_hamming_distance(self):
        self.assertEqual(utils.hamming_distance('00', '00'), 0)
        self.assertEqual(utils.hamming_distance('01', '00'), 1)
        self.assertEqual(utils.hamming_distance('11', '00'), 2)
        
         
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()