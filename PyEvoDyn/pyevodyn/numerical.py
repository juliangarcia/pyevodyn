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
from scipy.integrate import odeint


def monomorphous_transition_matrix(intensity_of_selection, payoff_function=None, mutation_kernel=None,
                                   mutation_probability=None, population_size=None, game_matrix=None,
                                   number_of_strategies=None, mapping='EXP', **kwargs):
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

    See also
    ---------
    old signature: (game_matrix, population_size, intensity_of_selection, kernel=None, mutation_probability=None)

    """
    if (mutation_kernel is None and mutation_probability is None):
        raise ValueError(
            'Either a mutation kernel, or a mutation probability has to be specified')
    if game_matrix is not None:
        size = len(game_matrix)
    elif payoff_function is not None and number_of_strategies is not None:
        size = number_of_strategies
    else:
        raise ValueError(
            "if specifying a payoff function the number_of_strategies has to be provided")
    if (mutation_kernel is None and mutation_probability is not None):
        mutation_kernel = utils.uniform_mutation_kernel(
            mutation_probability, size)
    ans = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            if i != j:
                # chance that j appears in an i population
                # the mutation probability already comes from the kernel
                # divided by the number of strategies
                ans[i, j] = transition_probability(i, j, intensity_of_selection, payoff_function, mutation_kernel[
                    i, j], population_size, game_matrix, number_of_strategies, mapping, **kwargs)
    for i in range(0, size):
        ans[i, i] = 1.0 - math.fsum(ans[i, :])
    return ans


def transition_probability(origin_index, destiny_index, intensity_of_selection, payoff_function, mutation_probability,
                           population_size=None, game_matrix=None, number_of_strategies=None, mapping='EXP', **kwargs):
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

    See also:
    ---------
    old  signature: (game_matrix_2_x_2, population_size, intensity_of_selection, mutation_probability)


    """
    fix_probability = fixation_probability(
        destiny_index, origin_index, intensity_of_selection,
        population_size, payoff_function, game_matrix, number_of_strategies, mapping, **kwargs)
    return mutation_probability * fix_probability


def fixation_probability(mutant_index, resident_index, intensity_of_selection, population_size, payoff_function=None,
                         game_matrix=None, number_of_strategies=None, mapping='EXP', **kwargs):
    suma = []
    for k in range(1, population_size):
        mult = []
        for j in range(1, k + 1):
            if (payoff_function is not None and game_matrix is None):
                if (number_of_strategies is None):
                    raise ValueError(
                        'When using a custom payoff_function you must specify number_of_strategies.')
                strategies = np.zeros(number_of_strategies, dtype=int)
                strategies[mutant_index] = j
                strategies[resident_index] = population_size - j
                payoffMutant = payoff_function(
                    mutant_index, population_composition=strategies, **kwargs)
                payoffResident = payoff_function(
                    resident_index, population_composition=strategies, **kwargs)
            elif (game_matrix is not None and payoff_function is None):
                (payoffMutant, payoffResident) = payoff_from_matrix(
                    mutant_index, resident_index, game_matrix, j, population_size)
            else:
                raise ValueError(
                    'No valid payoff structure given, please specify a game_matrix or a payoff_function.')
            if (mapping == 'EXP'):
                fitnessMutant = math.e ** (intensity_of_selection * payoffMutant)
                fitnessResident = math.e ** (intensity_of_selection * payoffResident)
            elif (mapping == 'LIN'):
                fitnessMutant = 1.0 - intensity_of_selection + \
                                intensity_of_selection * payoffMutant
                fitnessResident = 1.0 - intensity_of_selection + \
                                  intensity_of_selection * payoffResident
            else:
                raise ValueError(
                    'No valid mapping given. Use EXP or LIN for exponential and linear respectively.')
            mult.append(fitnessResident / fitnessMutant)
        suma.append(utils.kahan_product(mult))
    if any(np.isinf(suma)):
        return 0.0
    try:
        complex_expression = utils.kahan_sum(suma)
    except OverflowError:
        return 0.0
    if np.isinf(complex_expression):
        return 0.0
    return 1.0 / (1.0 + complex_expression)


def payoff_from_matrix(focal_index, other_index, game_matrix, number_of_focal_individuals, population_size):
    """
    Computes a vector of payoffs from a game_matrix. The first element is the payoff of the strategy with index focal_index, the second element is
    the payoff of the strategy with index other_index. The game is given by game_matrix, and assumes a population cmposed of number_of_focal_individuals of strategy focal_index
    and population_size - number_of_focal_individuals copies of strategy other_index
    """
    sub_matrix = np.array([[game_matrix[focal_index, focal_index], game_matrix[focal_index, other_index]], [
        game_matrix[other_index, focal_index], game_matrix[other_index, other_index]]])
    return (1.0 / (population_size - 1.0)) * (np.dot(sub_matrix, np.array(
        [number_of_focal_individuals, population_size - number_of_focal_individuals])) - np.diagonal(sub_matrix))


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
    # builds a dictionary with position, eigenvalue
    # and retrieves from this, the index of the largest eigenvalue
    index = max(
        zip(range(0, len(eigenvalues)), eigenvalues), key=itemgetter(1))[0]
    # returns the normalized vector corresponding to the
    # index of the largest eigenvalue
    # and gets rid of potential complex values
    return utils.normalize_vector(np.real(eigenvectors[:, index]))


# def __auxiliary_l_k(k, number_of_strategies, game_matrix):
# return (1.0 / number_of_strategies) * np.sum([game_matrix[k, k] +
# game_matrix[k, i] - game_matrix[i, k] - game_matrix[i, i] for i in
# xrange(0, number_of_strategies)])


def __auxiliary_l_k(index, game_matrix, size):
    suma = 0
    for i in range(0, size):
        suma = suma + (game_matrix[index, index] + game_matrix[
            index, i] - game_matrix[i, index] - game_matrix[i, i])
    return suma / size


def __auxiliary_h_k(index, game_matrix, size):
    suma = 0
    for i in range(0, size):
        for j in range(0, size):
            suma = suma + (game_matrix[index, i] - game_matrix[i, j])
    return suma / (size ** 2)


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
    return [(1 + ((intensity_of_selection * population_size * (1 - mutation_probability)) * ((__auxiliary_l_k(index,
                                                                                                              game_matrix,
                                                                                                              size) + population_size * mutation_probability * __auxiliary_h_k(
        index, game_matrix, size)) / ((1 + population_size * mutation_probability) * (
        2 + population_size * mutation_probability))))) / size for index in range(0, size)]


def replicator_equation(x, t, game):
    """
    Parameters
    ----------
    x: ndarray initial state
    game: ndarray square matrix

    Returns:
    out: ndarray next state
    """
    fitness_vector = np.dot(game, x)
    average_fitness = np.dot(x, fitness_vector)
    return x * (fitness_vector - average_fitness)


def replicator_trajectory(game_matrix, x_0, t_vector, **kwargs):
    """
    Computes a replicator trajectory for a given game, starting point and time vector.
    It uses scipy's odeint.

    Parameters
    ----------
    game_matrix: numpy matrix (must be square)
    x_0: ndarray
    t_vector: time array

    Returns
    -------
    out: list

    Examples
    --------
    #TODO: Write examples
    """
    soln = odeint(replicator_equation, x_0,
                  t_vector, args=(game_matrix,), **kwargs)
    return [soln[:, i] for i in range(len(game_matrix))]


def replicator_equation_two_populations(x, t, game1, game2, number__of_strategies_population_1,
                                        number__of_strategies_population_2):
    """
    Parameters
    ----------
    x: ndarray initial state (concatenated from the two populations)
    t: time
    game1: ndarray, game for population 1
    game2: ndarray, game for population 2
    number__of_strategies_population_1: int
    number__of_strategies_population_2: int
    Returns:
    out: ndarray next state (concatenated from the two populations)
    """

    # the first piece of y corresponds to population 1
    x_population_1 = x[0:number__of_strategies_population_1]
    # the second piece of y corresponds to population 2
    x_population_2 = x[number__of_strategies_population_1:
    number__of_strategies_population_1 + number__of_strategies_population_2]
    # First Ay
    fitness_vector_1 = np.dot(game1, x_population_2)
    # and Bx (see equation above)
    fitness_vector_2 = np.dot(game2, x_population_1)
    # Now xAy
    average_fitness_1 = np.dot(x_population_1, fitness_vector_1)
    # And yBx
    average_fitness_2 = np.dot(x_population_2, fitness_vector_2)
    # the next lines correspond to equations 10.5 and 10.6 of Hofbauer and
    # Sigmund (page 116)
    new_population_1 = x_population_1 * (fitness_vector_1 - average_fitness_1)
    new_population_2 = x_population_2 * (fitness_vector_2 - average_fitness_2)
    return np.array(new_population_1.tolist() + new_population_2.tolist())


def replicator_trajectory_two_populations(game_matrix_1, game_matrix_2, x_0, y_0, t_vector, **kwargs):
    """
    Computes a replicator trajectory for two populations, given two games, starting points and time vector.
    It uses scipy's odeint.

    Parameters
    ----------
    game_matrix_1: numpy matrix (for population 1)
    game_matrix_2: numpy matrix (for population 2)
    x_0: ndarray
    y_0: ndarray
    t_vector: time array

    Returns
    -------
    out: list

    Examples
    --------
    #TODO: Write examples
    """
    # join initial populations to fit signature of replicator_equation
    start = np.array(x_0.tolist() + y_0.tolist())
    number__of_strategies_population_1 = len(x_0)
    number__of_strategies_population_2 = len(y_0)
    # solve
    soln = odeint(replicator_equation_two_populations, start, t_vector, args=(
        game_matrix_1, game_matrix_2, number__of_strategies_population_1, number__of_strategies_population_2), **kwargs)
    return [soln[:, i] for i in range(number__of_strategies_population_1 + number__of_strategies_population_2)]
