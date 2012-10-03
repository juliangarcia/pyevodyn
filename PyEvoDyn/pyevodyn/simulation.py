'''
Created on Aug 8, 2012

@author: garcia

#TODO: implement simulate fixation probability, test first convergence funcitions separately
'''
import numpy as np
import pyevodyn.utils as utils 
from pyevodyn import games

_FITNESS_MAPPING = ['exp', 'lin']


class MoranProcess(object):
    '''
    Simple Moran process class.
    '''

    def __init__(self, game_matrix, population_size, intensity_of_selection, fitness_mapping='exp', mutation_probability=None, mutation_kernel=None, incumbent_index=None, seed=None):
        '''
        Moran Process

        Parameters
        ----------
        game_matrix: np.darray, shape (n,n)
        population_size: int (>0)
        intensity_of_selection: double
        fitness_mapping
        mutation_probability:double between 0 and 1 (Optional)
        mutation_kernel: ndarray (Optional)
        incumbent_index: double (Optional) if not given initialize to a random edge
        seed: long (Optional)
        '''
        
        #TODO: provide a way to compute payoff with an arbitrary function from ndarray to ndarray http://stackoverflow.com/questions/803616/passing-functions-with-arguments-to-another-function-in-python
        
        #CHECKS
        if len(game_matrix.shape) != 2 or (game_matrix.shape[0] != game_matrix.shape[1]):
            raise ValueError('game_matrix must be a (numpy) square matrix')
        if (mutation_kernel is None and mutation_probability is None):
            raise ValueError('Either a mutation kernel, or a mutation probability has to be specified')
        if(fitness_mapping not in _FITNESS_MAPPING):
            raise ValueError(
                'fitness mapping must be one of ' + str(_FITNESS_MAPPING))

        #START INIT
        np.random.seed(seed)
        self.number_of_strategies = len(game_matrix)
        if (mutation_kernel is None and mutation_probability is not None):
            self.mutation_kernel = utils.uniform_mutation_kernel(
                mutation_probability, self.number_of_strategies)
        
        self.timestep = 0
        self.game_matrix = game_matrix
        self.intensity_of_selection = intensity_of_selection
        self.population_size = population_size
        self.population_array = np.zeros(self.number_of_strategies, dtype=int)
        if incumbent_index is None:
            #if no incumbent is given population_array starts at a random edge
            self.population_array = utils.random_edge_population(
                self.number_of_strategies, self.population_size)
        else:
            self.population_array[incumbent_index] = population_size
        if fitness_mapping == 'lin':
            self.__mapping_function = self.__linear_mapping
        if fitness_mapping == 'exp':
            self.__mapping_function = self.__exponential_mapping
        self.average_payoff_last_generation = None
        
    #EXPONENTIAL MAPPING
    def __exponential_mapping(self, value):
        return np.exp(self.intensity_of_selection * value)
    #LINEAR MAPPING

    def __linear_mapping(self, value):
        return 1.0 + self.intensity_of_selection * value

    def payoff_vector(self):
        """
        Computes a vector payoff containing the payoff of each strategy
        when the population plays the game given by the self.game_matrix

        Returns:
        -------
        ndarray

        """
        return (1.0/(self.population_size-1.0))*(np.dot(self.game_matrix,self.population_array)-np.diag(self.game_matrix))

    def step(self):
        """
        Main method, one generation step updates the population array
        returns the payoff of the population (mostly, nothing should be done with what is returned)

        """
        payoff = self.payoff_vector()
        fitness = np.array(map(self.__mapping_function,payoff)) 
        fitness /= fitness.sum()
        #choose one random guy in proportion to fitness
        chosen_one = utils.simulate_discrete_distribution(fitness)
        #mutate this guy
        chosen_one = utils.simulate_discrete_distribution(self.mutation_kernel[chosen_one])
        #a random guy dies
        current_distribution = np.array(self.population_array, dtype=float)
        current_distribution /= float(self.population_size)
        dies = utils.simulate_discrete_distribution(current_distribution)
        self.population_array[dies] = self.population_array[dies] - 1
        #we add a copy of the fit guy
        self.population_array[chosen_one] = self.population_array[chosen_one] + 1
        self.timestep += 1
        #the payoff is returned but almost never used.
        return payoff
    
    def reset(self, incumbent_index = None, seed=None):
        """
        Restarts the time step counter, reseeds and creates a new starting population
        #TODO: Proper docs
        
        """
        np.random.seed(seed)
        self.timestep = 0
        self.population_array = np.zeros(self.number_of_strategies, dtype=int)
        if incumbent_index is None:
            #if no incumbent is given population_array starts at a random edge
            self.population_array = utils.random_edge_population(
                self.number_of_strategies, self.population_size)
        else:
            self.population_array[incumbent_index] = self.population_size
    
    def is_monomorphous(self):
        """
        Determines if the population is monomorphous. 
        
        Returns
        -------
        int: the index of the strategy that in which the population is monomorphous, or None if it is not monomorphous.
        
        """
        max_index = self.population_array.argmax()
        if self.population_array[max_index] == self.population_size:
            return max_index
        return None
    
    
    
        
    
    def simulate_time_series(self, number_of_generations, report_every=1):
        """
        #TODO: Proper docs
        
        the answer is always a np array, where each row is a generation reported, starting with gen 0
        this can be turn nicely into a pandas data frame with data.time_series_matrix_to_pandas_data_frame
        
        """
        ans = np.array(np.concatenate(([self.timestep],self.population_array)))
        self.timestep = 0
        for _ in xrange(0,number_of_generations):
            self.step()
            if (self.timestep%report_every==0):
                fila = np.concatenate(([self.timestep],self.population_array))
                ans = np.vstack((ans, fila))
        return ans
    
    
    def simulate_stationary_distribution(self, burning_time_per_estimate, samples_per_estimate, number_of_estimates=1):
        """
        #TODO:proper docs and tests
        """
        ans = np.zeros(self.number_of_strategies)
        for __ in xrange(0,number_of_estimates):
            single_estimate = np.zeros(self.number_of_strategies)
            self.reset(incumbent_index=None, seed=None)
            #burn time
            for _ in xrange(0,burning_time_per_estimate):
                self.step()
            #sample
            for _ in xrange(0,samples_per_estimate):
                single_estimate = single_estimate + self.population_array
                self.step()
            single_estimate=(1.0/samples_per_estimate)*single_estimate
            ans += single_estimate
        return utils.normalize_vector((1.0/number_of_estimates)*ans)
        
        
    def simulate_average_payoff_across_trajectory(self, burning_time_per_estimate, samples_per_estimate, number_of_estimates=1):
        """
        #TODO:proper docs and tests, pep8 and clean names
        """
        ans = []
        for __ in xrange(0,number_of_estimates):
            single_estimate = 0.0
            self.reset(incumbent_index=None, seed=None)
            #burn time
            for _ in xrange(0,burning_time_per_estimate):
                self.step()
            #sample
            for _ in xrange(0,samples_per_estimate):
                payoff = self.step()
                single_estimate = single_estimate + np.dot(self.population_array, payoff)
            single_estimate=(1.0/samples_per_estimate)*single_estimate
            ans.append(single_estimate)
        return (1.0/number_of_estimates)*np.mean(ans)
    
    
    def simulate_fixation_probability(self, index_of_the_incumbent, index_of_the_mutant, number_of_samples):
        #temporarily set the mutation kernel to I
        positives = 0
        old_mutation_kernel = self.mutation_kernel
        self.mutation_kernel = np.identity(self.number_of_strategies)
        
        poblacion_init = np.zeros(self.number_of_strategies, dtype='int')
        poblacion_init[index_of_the_incumbent] = self.population_size -1
        poblacion_init[index_of_the_mutant] = 1
        
        for _ in xrange(0,number_of_samples):
            #reset the population with the incumbant and seed the mutant
            self.population_array = np.array(poblacion_init)
            converged_to = self.is_monomorphous()
            while(converged_to == None):
                self.step()
                converged_to = self.is_monomorphous()
            if(converged_to == index_of_the_mutant):
                positives+=1
        #give back kernel
        self.mutation_kernel = old_mutation_kernel
        #return
        return positives/float(number_of_samples)
    


def main():
    game = games.prisoners_dilemma_equal_gains()
    mp = MoranProcess(game, population_size=10, intensity_of_selection=0.0, fitness_mapping='exp', mutation_probability=0.001)
    print mp.simulate_fixation_probability(index_of_the_incumbent=1, index_of_the_mutant=0, number_of_samples=10000)
    
    
    
if __name__ == "__main__":
    main()        