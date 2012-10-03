'''
Created on Aug 8, 2012

@author: garcia

'''
import numpy as np
import pyevodyn.utils as utils 
from pyevodyn import games
from collections import namedtuple

_FITNESS_MAPPING = ['exp', 'lin']

StepResult = namedtuple('StepResult', 'new_population payoff_values')
        

class MoranProcess(object):
    '''
    Simple Moran process class. Provides the methods to simulate a moran process (stationary distribution, fixation probability, and average payoff across
    trajectories). It also provides a step method, that given a population array returns a new population and an array of payoff values.
    '''
    
    def __check_init_values(self, game_matrix, payoff_function,number_of_strategies,fitness_mapping, mutation_probability, mutation_kernel):
        if ((mutation_kernel is None and mutation_probability is None) or (mutation_kernel is not None and mutation_probability is not None)):
            raise ValueError('Either a mutation kernel, or a mutation probability has to be specified')
        if ((game_matrix is None and payoff_function is None) or (game_matrix is not None and payoff_function is not None)):
            raise ValueError('Either a game_matrix, or a payoff_function has to be specified')
        if ((game_matrix is not None) and (len(game_matrix.shape) != 2 or (game_matrix.shape[0] != game_matrix.shape[1]))):
            raise ValueError('if a marix_game is provided, it must be a square numpy array')
        if (game_matrix is None and number_of_strategies is None):
            raise ValueError('if a marix_game is not provided, you must provide number_of_strategies')
        if(fitness_mapping not in _FITNESS_MAPPING):
            raise ValueError(
                'fitness mapping must be one of ' + str(_FITNESS_MAPPING))
        

    def __init__(self, population_size, intensity_of_selection, game_matrix=None, payoff_function= None, number_of_strategies=None, fitness_mapping='exp', mutation_probability=None, mutation_kernel=None):
        '''
        Moran Process

        Parameters
        ----------
        population_size: int (>0)
        intensity_of_selection: double
        game_matrix: np.darray, shape (n,n) (Optional)
        fitness_mapping (Optional): function, from ndarray (abundance of each strategy) into ndarray (payoff of each strategy)
        number_of_strategies (Optional): if a game_matrix is not provided you must provide the number of strategies
        mutation_probability:double between 0 and 1 (Optional)
        mutation_kernel: ndarray (Optional)
        '''
        
        #CHECK THAT EVERYTHING IS IN ORDER
        self.__check_init_values(game_matrix, payoff_function, number_of_strategies, fitness_mapping, mutation_probability, mutation_kernel)
        
        #ASSIGN INIT VARIABLES
        
        #NUMBER OF STRATEGIES
        try:
            self.number_of_strategies = len(game_matrix)
        except TypeError:
            self.number_of_strategies = number_of_strategies
        
        #MUTATION KERNEL
        if (mutation_kernel is None and mutation_probability is not None):
            self.mutation_kernel = utils.uniform_mutation_kernel(mutation_probability, self.number_of_strategies)
        #PAYOFF FUNCTION
        self.game_matrix = game_matrix
        if self.game_matrix is not None:
            self.payoff_function = self.__default_payoff_function
        else:
            self.payoff_function = payoff_function
            #test that the payoff function is well specified
            if len(self.payoff_function(np.ones(self.number_of_strategies))) != self.number_of_strategies:
                raise ValueError('The payoff function is mispecified')
            
        #INTENSITY OF SELECTION
        self.intensity_of_selection = intensity_of_selection
        #POP SIZE
        self.population_size = population_size
        #FITNESS MAPPING
        if fitness_mapping == 'lin':
            self.__mapping_function = self.__linear_mapping
        if fitness_mapping == 'exp':
            self.__mapping_function = self.__exponential_mapping
            
        #END OF INIT    
        
    
    
    #EXPONENTIAL MAPPING
    def __exponential_mapping(self, value):
        return np.exp(self.intensity_of_selection * value)
    #LINEAR MAPPING

    def __linear_mapping(self, value):
        return 1.0 + self.intensity_of_selection * value

    #GAME PAYOFF FUNCTION
    def __default_payoff_function(self, population_array):
        """
        Computes a vector payoff containing the payoff of each strategy
        when the population given by population_array plays the game given by self.game_matrix
        This is only used if a game if given, otherwise a function given by the user is used.
        Returns:
        -------
        ndarray

        """
        return (1.0/(self.population_size-1.0))*(np.dot(self.game_matrix,population_array)-np.diag(self.game_matrix))

    
    def step(self, population_array, mutation_step=True):
        """
        One generation step. Updates the given population array, returns a tuple with the new population and the vector of payoffs
        
        Parameters:
        ----------
        population_array = 
        mutation_step = 
        
        """
        #COMPUTE PAYOFF
        payoff = self.payoff_function(population_array)
        
        #COMPUTE FITNESS
        fitness = map(self.__mapping_function,payoff)
        #FITNESS PROPORTIONAL DISTRIBUTION 
        fitness /= np.sum(fitness)
        #choose one random guy in proportion to fitness
        chosen_one = utils.simulate_discrete_distribution(fitness)
        #mutate this guy
        if mutation_step:
            chosen_one = utils.simulate_discrete_distribution(self.mutation_kernel[chosen_one])
        #a random guy dies
        current_distribution = np.array(population_array, dtype=float)
        current_distribution /= float(self.population_size)
        dies = utils.simulate_discrete_distribution(current_distribution)
        
        population_array[dies] = population_array[dies] - 1
        #we add a copy of the fit guy
        population_array[chosen_one] = population_array[chosen_one] + 1
        #the payoff is returned but almost never used.
        return StepResult(population_array, payoff)
    
    
   

def is_population_monomorphous(population_array, population_size):
    """
    Determines if a population is monomorphous. 
    Returns
    -------
    int: the index of the strategy that in which the population is monomorphous, or None if it is not monomorphous.
    """
    max_index = population_array.argmax()
    if population_array[max_index] == population_size:
        return max_index
    return None
    



def main():
    #mp = MoranProcess(game, population_size=10, intensity_of_selection=0.0, fitness_mapping='exp', mutation_probability=0.001)
    pass
    
    
    
if __name__ == "__main__":
    main()        