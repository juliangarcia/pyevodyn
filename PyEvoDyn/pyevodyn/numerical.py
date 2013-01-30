'''
This modules contains functions to numerically compute fixation probabilities of a Moran process where the fitness is given by matrix games.
It heavily relies on numpy arrays. Exponential mapping of payoff to fitness is assumed throughout.
Created on Aug 8, 2012
@author: garcia
'''
import math
import numpy as np
import pyevodyn.utils as utils
from operator import itemgetter


def monomorphous_transition_matrix(game_matrix, population_size, intensity_of_selection, kernel=None, mutation_probability=None):
    """
    Computes the associated markov chain (transition matrix), when mutations are assumed to be small.
    The approximation is accurate when there are no stable mixtures between any pair of strategies.

    Parameters
    ----------
    game_matrix: numpy matrix
    population_size: int
    intensity_of_selection: float
    kernel: ndarray, optional stochastic matrix
    mutation_probability: float, optional uniform kernel

    Returns
    -------
    ans: ndarray, stochastic matrix

    """
    #TODO: Warning if the game contains stable mixtures

    if (kernel is None and mutation_probability is None):
        raise ValueError('Either a mutation kernel, or a mutation probability has to be specified')
    size = len(game_matrix)
    if (kernel is None and mutation_probability is not None):
        kernel = utils.uniform_mutation_kernel(mutation_probability, size)
    ans = np.zeros((size, size))
    for i in xrange(0, size):
        for j in xrange(0, size):
            if i != j:
                matrix_2_x_2 = utils.submaxtrix(game_matrix, i, j)
                mutation_probability = kernel[i, j]
                    #chance that j appears in an i population
                #the mutation probability already comes from the kernel divided by the number of strategies
                ans[i, j] = mutation_probability * fixation_probability_strategy_b(matrix_2_x_2, intensity_of_selection, population_size)
    for i in range(0, size):
        ans[i, i] = 1.0 - math.fsum(ans[i, :])
    return ans


def transition_probability_from_a_to_b(game_matrix_2_x_2, population_size, intensity_of_selection, mutation_probability):
    """
    Computes the transition probability between two homogeneous populations

    Parameters:
    ----------
    game_matrix_2_x_2: ndarray (size 2)
    population_size: int
    intensity_of_selection: double (positive or 0)
    mutation_probability: double between 0 and 1.

    Returns
    -------
    out: double

    """
    return mutation_probability * fixation_probability_strategy_b(game_matrix_2_x_2, intensity_of_selection, population_size)


def fixation_probability_strategy_b(game_matrix_2_x_2, intensity_of_selection, population_size):
    """
    Computes the fixation probability of a mutant B in a pop of A's, where the game between A and B is given by a matrix 2x2 game.
    This corresponds to  equation 20 of Traulsen,Shoresh, Nowak  2008.

    Parameters:
    -----------
    game_matrix_2_x_2: ndarray
    intensity_of_selection: double
    population_size: int

    Returns:
    -------
    double: fixation probability

    """
    a_value = game_matrix_2_x_2[0][0]
    b_value = game_matrix_2_x_2[0][1]
    c_value = game_matrix_2_x_2[1][0]
    d_value = game_matrix_2_x_2[1][1]
    transformed_matrix = [[d_value, c_value], [b_value, a_value]]
    lista = [__auxiliary_function_for_exponential_mapping(
        intensity_of_selection, k,
             transformed_matrix, population_size) for k in xrange(0, population_size)]
    if any(np.isinf(lista)):
        return 0.0
    try:
        suma = math.fsum(lista)
    except OverflowError:
        return 0.0
    if np.isinf(suma):
        return 0.0
    return 1.0 / suma


def fixation_probability_strategy_a(matrix_2_x_2, intensity_of_selection,
                                    population_size):
    """
    Computes the fixation probability of a mutant A in a pop of B's, where the game between A and B is given by a matrix 2x2 game.
    This corresponds to  equation 20 of Traulsen,Shoresh, Nowak  2008.
    See also, fixation_probability_strategy_b for the implementation where the mutant is strategy b.

    Parameters:
    -----------
    game_matrix_2_x_2: ndarray
    intensity_of_selection: double
    population_size: int

    Returns:
    -------
    double: fixation probability

    """
    lista = [__auxiliary_function_for_exponential_mapping(
        intensity_of_selection, k,
             matrix_2_x_2, population_size) for k in xrange(0, population_size)]
    if any(np.isinf(lista)):
        return 0.0
    try:
        suma = math.fsum(lista)
    except OverflowError:
        return 0.0
    if np.isinf(suma):
        return 0.0
    return 1.0 / suma


def __auxiliary_function_for_exponential_mapping(
    intensity_of_selection, k_value, matrix_2_x_2,
        population_size):
    """
    This is an auxiliary function, used to compute fixation probabilities. It should not be called alone.
    It implements the sum term in equation 20 of  Traulsen,Shoresh, Nowak  2008.
    """
    a_value = matrix_2_x_2[0][0]
    b_value = matrix_2_x_2[0][1]
    c_value = matrix_2_x_2[1][0]
    d_value = matrix_2_x_2[1][1]
    part_0 = k_value * (k_value + 1.0) * (intensity_of_selection / 2.0) * (1.0 / (population_size - 1.0)) * (-a_value + b_value + c_value - d_value)
    part_1 = k_value * intensity_of_selection * (1.0 / (population_size - 1.0)) * (a_value - b_value * population_size + d_value * population_size - d_value)
    return math.e ** (part_0 + part_1)


def stationary_distribution(transition_matrix_markov_chain):
    '''
    Computes the stationary_distribution of a markov chain. The matrix is given by rows.

    Parameters
    ----------
    transition_matrix_markov_chain: ndarray (must be a numpy array)

    Returns
    -------
    out: ndarray

    Examples
    -------
    >>>stationary_distribution(np.array([[0.1,0.9],[0.9,0.1]]))
    Out[1]: array([ 0.5,  0.5])
    >>>stationary_distribution(np.array([[0.1,0.0],[0.9,0.1]]))
    Out[1]: array([ 1.,  0.])
    >>>stationary_distribution(np.array([[0.6,0.4],[0.2,0.8]]))
    Out[1]: array([ 0.33333333,  0.66666667])
    '''
    transition_matrix_markov_chain = transition_matrix_markov_chain.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix_markov_chain)
    #builds a dictionary with position, eigenvalue
    #and retrieves from this, the index of the largest eigenvalue
    index = max(
        zip(range(0, len(eigenvalues)), eigenvalues), key=itemgetter(1))[0]
    #returns the normalized vector corresponding to the
    #index of the largest eigenvalue
    # and gets rid of potential complex values
    return utils.normalize_vector(np.real(eigenvectors[:, index]))


#def __auxiliary_l_k(k, number_of_strategies, game_matrix):
#    return (1.0 / number_of_strategies) * np.sum([game_matrix[k, k] + game_matrix[k, i] - game_matrix[i, k] - game_matrix[i, i] for i in xrange(0, number_of_strategies)])


def __auxiliary_l_k(index, game_matrix, size):
    suma = 0
    for i in xrange(0,size):
        suma = suma + (game_matrix[index,index] + game_matrix[index,i] - game_matrix[i,index] - game_matrix[i,i])
    return suma/size


def __auxiliary_h_k(index,game_matrix, size):
    suma = 0
    for i in range(0,size):
        for j in range(0, size):
            suma = suma + (game_matrix[index,i] - game_matrix[i,j])
    return suma/(size**2)

def stationary_distribution_weak_selection(game_matrix, population_size, intensity_of_selection, mutation_probability):
    """
    Computes the abundance in the stationary distribution, in the weak selection limit, using Antal et al.(2009) JTB, Eq 20.
    This approximation is valid for large population_size, weak selection (intensity_of_selection*population_size <<1 )
    and arbitrary mutation rate.
    
    This does not support mutation kernels

    Parameters
    ----------
    game_matrix: numpy matrix
    population_size: int #N
    intensity_of_selection: float
    mutation_probability: float

    Returns
    -------
    out: ndarray
    
    See also:
    --------
    pyevodyn.symblic.stationary_distribution_weak_selection
    This is the numerical approach

    Examples
    --------
    #TODO: Write examples 
    """
    
    size = game_matrix.shape[0]
    return [(1 + ((intensity_of_selection*population_size*(1-mutation_probability))*((__auxiliary_l_k(index,game_matrix, size) + population_size*mutation_probability*__auxiliary_h_k(index,game_matrix, size))/((1+population_size*mutation_probability)*(2+population_size*mutation_probability)))))/size for index in xrange(0, size)]



def replicator_step(game_matrix, x, dt=0.0001):
    """
    Computes a replicator step.
    
    This does not support mutation kernels

    Parameters
    ----------
    game_matrix: numpy matrix
    x: ndarray 
    dt: float
    
    Returns
    -------
    out: ndarray
    
    Examples
    --------
    #TODO: Write examples 
    """
    fitness_vector = np.dot(game_matrix,x)
    average_fitness = np.dot(x,fitness_vector)
    return x + x*(fitness_vector - average_fitness)*dt
    
    
def replicator_trajectory(game_matrix, x_0, maximum_iterations, dt=0.000001):
    """
    Computes a replicator trajectory. If it reaches a rest point it returns observations until there. Otherwise
    it keeps on going until maximum_iterations are reached.
    
        Parameters
    ----------
    game_matrix: numpy matrix
    x_0: ndarray
    maximum_iterations:int 
    dt: float
    
    Returns
    -------
    out: list
    
    Examples
    --------
    #TODO: Write examples 
    """
    orbit = [x_0]
    for _ in xrange(0, maximum_iterations):
        orbit.append(replicator_step(game_matrix, orbit[-1], dt))
        if (np.allclose(orbit[-1],orbit[-2])):
            return orbit
    return orbit

