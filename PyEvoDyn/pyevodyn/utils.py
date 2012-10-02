'''
This module contains useful functions that are used across the different modules.
Created on Aug 8, 2012

@author: garcia
'''
import math
import numpy as np


def uniform_mutation_kernel(mutation_probability, number_of_strategies):
    """
    Returns a uniform mutation kernel, A = [a_ij] = u/ (n-1), where n is the number of strategies
    and u is the mutation probability

    Parameters
    ----------
    mutation_probability: float
    number_of_strategies: int

    Returns
    -------
    a_matrix: ndarray

    Examples
    --------
    >>> uniform_mutation_kernel(0.1,2)
    Out[1]:
    array([[ 0.9,  0.1],
           [ 0.1,  0.9]])
    >>>uniform_mutation_kernel(0.1,3)
    Out[1]:
    array([[ 0.9 ,  0.05,  0.05],
           [ 0.05,  0.9 ,  0.05],
           [ 0.05,  0.05,  0.9 ]])
    """
    a_matrix = np.zeros(shape=(number_of_strategies, number_of_strategies))
    for i in xrange(0, number_of_strategies):
        for j in xrange(0, number_of_strategies):
            if i != j:
                a_matrix[i, j] = mutation_probability / (
                    number_of_strategies - 1.0)
    for i in xrange(0, number_of_strategies):
        a_matrix[i][i] = 1.0 - a_matrix[i].sum()
    return a_matrix


def submaxtrix(matrix, i, j):
    """
    Given matrix A, returns a sub matrix given by indexes i and j, such that [[a_ii, a_ij],[a_ji, a_jj]]

    Parameters
    ----------
    matrix: ndarray
    i: int
    j: int

    Returns
    -------
    out: ndarray

    Examples
    --------
    >>> submaxtrix([[2,3,1],[4,2,3],[2,1,0]],0,2)
    Out[1]:
    array([[2, 1],
           [2, 0]])
    """
    return np.array([[matrix[i][i], matrix[i][j]], [matrix[j][i], matrix[j][j]]])


def normalize_vector(vector):
    """
    Divides each element of vector over the sum of all the elements in it.
    Assumes that at least one element in vector is not zero, and at least one element is a double.

    Parameters
    ----------
    vector: ndarray

    Returns
    -------
    out: ndarray

    Examples
    -------
    >>> normalize_vector([1.0,1.0,1.0])
    Out[1]: array([ 0.33333333,  0.33333333,  0.33333333])

    """
    vector /= np.max(np.sum(vector), axis=0)
    return vector


def kahan_sum(list_of_floating_point_numbers):
    """
    Computes the sum of a list of floating point numbers, correcting for precision loss.

    Parameters:
    ----------
    list_of_floating_point_numbers: ndarray

    Returns:
    -------
    Sum of the elements.
    """
    suma = 0.0
    c = 0.0
    for i in xrange(0, len(list_of_floating_point_numbers)):
        y = list_of_floating_point_numbers[i] - c
        t = suma + y
        c = (t - suma) - y
        suma = t
    return suma


def kahan_product(list_of_floating_point_numbers):
    """
    Computes the product of a list of floating point numbers, correcting for precision loss.

    Parameters:
    ----------
    list_of_floating_point_numbers: ndarray

    Returns:
    -------
    Product of the elements.
    """
    if 0.0 in list_of_floating_point_numbers:
        return 0.0
    log_list = [math.log(x) for x in list_of_floating_point_numbers]
    suma = kahan_sum(log_list)
    return math.e ** suma


def binomial_coefficient(n_, k_):
    """
    Computes the coefficient of the x k-term in the polynomial expansion of the binomial power (1 + x) n
    Appropriately corrects for special cases, unlike the numpy implementation

    Parameters:
    ----------
    n: int
    k:int

    Returns:
    -------
    Binomial Coefficient.
    """
    n = float(n_)
    k = float(k_)
    if (n == -1 and k > 0):
        return -1.0 ** k
    if (n == -1 and k < 0):
        return(-1.0 ** k) * -1.0
    if (n == -1 and k == 0):
        return 1.0
    if (n == k):
        return 1.0
    if n >= k:
        if k > n - k:  # take advantage of symmetry
            k = n - k
        c = 1
        for i in range(int(k)):
            c = c * (n - i)
            c = c // (i + 1)
        return float(c)
    else:
        return 0.0


def hypergeometric(x, i):
    #TODO: Check against different numpy implementations
    """
    Computes the hypergeometric coefficients...

    Parameters:
    ----------
    x: ndarray
    i:ndarray

    Returns:
    -------
    Hypergeomertic coefficient  Coefficient.
    """
    #TODO:Improve documentation and add examples
    if len(x) != len(i):
        raise ValueError('x and i should be arrays of the same size')
    list_of_binomial_coefficients = [binomial_coefficient(
        x[k], i[k]) for k in xrange(0, len(x))]
    nom = kahan_product(list_of_binomial_coefficients)
    if nom == 0.0:
        return 0.0
    den = binomial_coefficient(math.fsum(x), math.fsum(i))
    return nom / den


def simulate_discrete_distribution(distribution):
    """
    Samples the distribution given by distribution, returning the index of the chosen event (a number between 0 and len(distribution))

    THIS FUNCTION WILL BE OBSOLETE, and probably should go somewhere else
    #numpy 1.7 will come with the np.random.choice module

    Parameters:
    ----------
    distribution ndaray, such taht sum(ndarray)=1

    Returns:
    --------
    int: chosen event

    """
    subtotal = 0.0
    r = np.random.rand()
    j = 0
    while (subtotal <= r):
        subtotal = subtotal + distribution[j]
        j += 1
    return j - 1


def random_edge_population(number_of_strategies, population_size):
    """
    Creates a population array that lives on a random edge. The edge and the distribution over the edge are chosen uniformly.

    Parameters:
    ----------
    number_of_strategies: int > 1
    population_size: int > 0

    Returns:
    -------
    ndarray: population composition

    Examples:
    --------
    >>random_edge_population(2,5)
    Out[1]: array([2, 3])
    >>> random_edge_population(3,10)
    Out[1]: array([3, 0, 7])
    """
    if number_of_strategies < 2 or population_size < 1:
        raise ValueError(
            "number_of_strategies must be > 1, population_size >= 0")
    ans = np.zeros(number_of_strategies, dtype=int)
    edge1, edge2 = 0, 0
    while (edge1 == edge2):
        [edge1, edge2] = np.random.random_integers(
            0, number_of_strategies - 1, 2)
    for __ in xrange(0, population_size):
        if np.random.randint(0, 2):
            ans[edge1] += 1
        else:
            ans[edge2] += 1
    return ans


def hamming_distance(str1, str2):
    """
    Computes the number of different characters in the same position for two given strings. Typically the inputs are binary, but it needs not be the case.

    Parameters:
    ----------
    str1: str
    str2: str

    Returns:
    -------
    int: hamming distance

    Examples:
    --------
    >>>hamming_distance('00','11')
    Out[1]: 2
    >>>hamming_distance('00','01')
    Out[1]: 1
    >>>hamming_distance('00','00')
    Out[1]: 0
    >>>hamming_distance('00','0')
    Out[1]: 0
    """
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

