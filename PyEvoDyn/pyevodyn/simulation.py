'''
Created on Aug 8, 2012

@author: garcia

'''
import numpy as np
import pyevodyn.utils as utils
from collections import namedtuple
from math import e as the_number_e
import pyevodyn

_FITNESS_MAPPING = ['exp', 'lin']

StepResult = namedtuple('StepResult', 'new_population payoff_values')


class MoranProcess(object):
    '''
    Simple Moran process class. Provides the methods to simulate a moran process (stationary distribution, fixation probability, and average payoff across
    trajectories). It also provides a step method, that given a population array returns a new population and an array of payoff values.
    
    As of now this class is too slow to be usable in real projects. I am currently working on a C extension in order to solve this. The idea is to clone the class using cython. 
    This is managed in a separate project. 
    '''

    def __check_init_values(self, game_matrix, payoff_function, number_of_strategies, fitness_mapping,
                            mutation_probability, mutation_kernel):
        if ((mutation_kernel is None and mutation_probability is None) or (
                mutation_kernel is not None and mutation_probability is not None)):
            raise ValueError('Either a mutation kernel, or a mutation probability has to be specified')
        if ((game_matrix is None and payoff_function is None) or (
                game_matrix is not None and payoff_function is not None)):
            raise ValueError('Either a game_matrix, or a payoff_function has to be specified')
        if ((game_matrix is not None) and (
                len(game_matrix.shape) != 2 or (game_matrix.shape[0] != game_matrix.shape[1]))):
            raise ValueError('if a marix_game is provided, it must be a square numpy array')
        if (game_matrix is None and number_of_strategies is None):
            raise ValueError('if a marix_game is not provided, you must provide number_of_strategies')
        if (fitness_mapping not in _FITNESS_MAPPING):
            raise ValueError(
                'fitness mapping must be one of ' + str(_FITNESS_MAPPING))

    def __init__(self, population_size, intensity_of_selection, game_matrix=None, payoff_function=None,
                 number_of_strategies=None, fitness_mapping='exp', mutation_probability=None, mutation_kernel=None):
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

        # CHECK THAT EVERYTHING IS IN ORDER
        self.__check_init_values(game_matrix, payoff_function, number_of_strategies, fitness_mapping,
                                 mutation_probability, mutation_kernel)

        # ASSIGN INIT VARIABLES

        # NUMBER OF STRATEGIES
        try:
            self.number_of_strategies = len(game_matrix)
        except TypeError:
            self.number_of_strategies = number_of_strategies

        # MUTATION KERNEL
        if (mutation_kernel is None and mutation_probability is not None):
            self.mutation_kernel = utils.uniform_mutation_kernel(mutation_probability, self.number_of_strategies)
        # PAYOFF FUNCTION
        self.game_matrix = game_matrix
        if self.game_matrix is not None:
            self.payoff_function = self.__default_payoff_function
            self.__diagonal = np.diagonal(self.game_matrix)
        else:
            self.payoff_function = payoff_function
            # test that the payoff function is well specified
            if len(self.payoff_function(np.ones(self.number_of_strategies))) != self.number_of_strategies:
                raise ValueError('The payoff function is mispecified')

        # INTENSITY OF SELECTION
        self.intensity_of_selection = intensity_of_selection
        # POP SIZE
        self.population_size = population_size
        # FITNESS MAPPING
        self.fitness_mapping = fitness_mapping

        # END OF INIT

    # GAME PAYOFF FUNCTION
    def __default_payoff_function(self, population_array):
        """
        Computes a vector payoff containing the payoff of each strategy
        when the population given by population_array plays the game given by self.game_matrix
        This is only used if a game if given, otherwise a function given by the user is used.
        Returns:
        -------
        ndarray

        """
        return (1.0 / (self.population_size - 1.0)) * (np.dot(self.game_matrix, population_array) - self.__diagonal)

    def step(self, population_array, mutation_step=True):
        """
        One generation step. Updates the given population array, returns a tuple with the new population and the vector of payoffs
        
        Parameters:
        ----------
        population_array = 
        mutation_step = 
        
        """
        return _detached_step(population_array, self.population_size, self.payoff_function, self.intensity_of_selection,
                              self.number_of_strategies, self.mutation_kernel, self.fitness_mapping, mutation_step)

    def simulate_fixation_probability(self, index_of_the_incumbent, index_of_the_mutant, number_of_samples, seed=None):
        """
        #TODO: Simulate
        """
        np.random.seed(seed)
        # variables that do not change
        pop_size = self.population_size
        payoff_func = self.payoff_function
        intensity = self.intensity_of_selection
        nr_strategies = self.number_of_strategies
        kernel = self.mutation_kernel
        mapping = self.fitness_mapping
        # no dots inside loops
        array = np.array
        positives = 0
        initial_population = np.zeros(self.number_of_strategies, dtype='int')
        initial_population[index_of_the_incumbent] = self.population_size - 1
        initial_population[index_of_the_mutant] = 1
        for _ in range(0, number_of_samples):
            population_array = array(initial_population)
            converged_to = is_population_monomorphous(population_array, pop_size)
            while (converged_to == None):
                (population_array, __) = _detached_step(population_array, pop_size, payoff_func, intensity,
                                                        nr_strategies, kernel, mapping, mutation_step=False)
                converged_to = is_population_monomorphous(population_array, pop_size)
            if (converged_to == index_of_the_mutant):
                positives += 1
        return positives / float(number_of_samples)

    def simulate_time_series(self, number_of_generations, population_array, report_every=1, seed=None):
        """
        #TODO: Proper docs
        the answer is always a np array, where each row is a generation reported, starting with gen 0
        this can be turn nicely into a pandas data frame with data.time_series_matrix_to_pandas_data_frame
        """
        np.random.seed(seed)
        pop_size = self.population_size
        payoff_func = self.payoff_function
        intensity = self.intensity_of_selection
        nr_strategies = self.number_of_strategies
        kernel = self.mutation_kernel
        mapping = self.fitness_mapping
        timestep = 0
        ans = np.array(np.concatenate(([timestep], population_array)))
        for _ in range(0, number_of_generations):
            (population_array, __) = _detached_step(population_array, pop_size, payoff_func, intensity, nr_strategies,
                                                    kernel, mapping, mutation_step=True)
            timestep += 1
            if (timestep % report_every == 0):
                fila = np.concatenate(([timestep], population_array))
                ans = np.vstack((ans, fila))
        return ans

    def simulate_stationary_distribution(self, burning_time_per_estimate, samples_per_estimate, number_of_estimates=1,
                                         seed=None):
        """
        #TODO:proper docs and tests
        """
        np.random.seed(seed)
        ans = np.zeros(self.number_of_strategies)
        pop_size = self.population_size
        payoff_func = self.payoff_function
        intensity = self.intensity_of_selection
        nr_strategies = self.number_of_strategies
        kernel = self.mutation_kernel
        mapping = self.fitness_mapping
        # no dots inside loops
        random_edge = utils.random_edge_population
        for __ in range(0, number_of_estimates):
            single_estimate = np.zeros(self.number_of_strategies)
            population_array = random_edge(nr_strategies, pop_size)
            # burn time
            for _ in range(0, burning_time_per_estimate):
                (population_array, ___) = _detached_step(population_array, pop_size, payoff_func, intensity,
                                                         nr_strategies, kernel, mapping, mutation_step=True)
            # sample
            for _ in range(0, samples_per_estimate):
                single_estimate = single_estimate + population_array
                (population_array, ___) = _detached_step(population_array, pop_size, payoff_func, intensity,
                                                         nr_strategies, kernel, mapping, mutation_step=True)
                single_estimate *= (1.0 / samples_per_estimate)
                ans += single_estimate
        return utils.normalize_vector((1.0 / number_of_estimates) * ans)

    def simulate_average_payoff_across_trajectory(self, burning_time_per_estimate, samples_per_estimate,
                                                  number_of_estimates=1, seed=None):
        """
        TODO:proper docs and tests, pep8 and clean names
        """
        np.random.seed(seed)
        pop_size = self.population_size
        payoff_func = self.payoff_function
        intensity = self.intensity_of_selection
        nr_strategies = self.number_of_strategies
        kernel = self.mutation_kernel
        mapping = self.fitness_mapping
        # no dots inside loops
        random_edge = utils.random_edge_population
        dot = np.dot
        ans = []
        for __ in range(0, number_of_estimates):
            single_estimate = 0.0
            population_array = random_edge(nr_strategies, pop_size)
            # burn time
            for _ in range(0, burning_time_per_estimate):
                (population_array, _) = _detached_step(population_array, pop_size, payoff_func, intensity,
                                                       nr_strategies, kernel, mapping, mutation_step=True)
            # sample
            for _ in range(0, samples_per_estimate):
                (population_array, payoff) = _detached_step(population_array, pop_size, payoff_func, intensity,
                                                            nr_strategies, kernel, mapping, mutation_step=True)
                single_estimate = single_estimate + dot(population_array, payoff)
                single_estimate *= (1.0 / samples_per_estimate)
                ans.append(single_estimate)
        return (1.0 / number_of_estimates) * np.mean(ans)


def _detached_step(population_array, population_size, payoff_function, intensity_of_selection, number_of_strategies,
                   mutation_kernel, fitness_mapping='exp', mutation_step=True):
    """
    One generation step. This is a copy of the previous but it is detached so that it can be called more efficiently
    
    Parameters:
    ----------
    population_array = 
    mutation_step = 
    
    """
    current_distribution = (1.0 / population_size) * np.array(population_array, dtype=float)
    # COMPUTE PAYOFF
    payoff = payoff_function(population_array)
    # compute fitness distribution
    fitness_sum = 0.0
    fitness = np.empty_like(payoff)
    if fitness_mapping == 'lin':
        for i in range(0, number_of_strategies):
            val = current_distribution[i] * (1.0 - intensity_of_selection + intensity_of_selection * payoff[i])
            fitness[i] = val
            fitness_sum += val
        fitness /= fitness_sum
    else:
        for i in range(0, number_of_strategies):
            val = current_distribution[i] * (the_number_e ** (intensity_of_selection * payoff[i]))
            fitness[i] = val
            fitness_sum += val
        fitness /= fitness_sum
        # choose one random guy in proportion to fitness
    chosen_one = utils.simulate_discrete_distribution(fitness)
    # mutate this guy
    if mutation_step:
        chosen_one = utils.simulate_discrete_distribution(mutation_kernel[chosen_one])
    # a random guy dies
    dies = utils.simulate_discrete_distribution(current_distribution)

    population_array[dies] -= 1
    # we add a copy of the fit guy
    population_array[chosen_one] += 1
    # the payoff is returned but almost never used.
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
    game = pyevodyn.games.two_times_two(a=3.0, b=0.5, c=0.5, d=2.0)
    intensity_of_selection = 1.0
    population_size = 5
    number_of_samples = 100000
    index_of_the_incumbent = 1
    index_of_the_mutant = 0
    print
    "Here..."
    mp = MoranProcess(population_size, intensity_of_selection, game_matrix=game, fitness_mapping='exp',
                      mutation_probability=0.1)
    print
    mp.simulate_fixation_probability(index_of_the_incumbent=index_of_the_incumbent,
                                     index_of_the_mutant=index_of_the_mutant, number_of_samples=number_of_samples,
                                     seed=None)


if __name__ == "__main__":
    main()
