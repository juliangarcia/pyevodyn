'''
Symbolic functions for evolutionary dynamics

Created on Aug 8, 2012
@author: garcia
'''
import sympy
import numpy as np
import pickle


def uniform_mutation_kernel(number_of_strategies,mutation_probability, self_interaction = False):
    """
    Computes a symbolic representation of a uniform mutation kernel
    
    Parameters
    ----------
    number_of_strategies: int
    mutation_probability: variable
    self_interaction: bool = False (optional)
    
    Returns
    -------
    out: Matrix
    
    See also:
    ---------
    Non symbolic equivalent pyevodyn.utils.uniform_mutation_kernel
    
    Examples
    --------
    >>>symbolic_uniform_mutation_kernel(3,mu)
    Out[1]: 
    [-mu + 1.0,      mu/2,      mu/2]
    [     mu/2, -mu + 1.0,      mu/2]
    [     mu/2,      mu/2, -mu + 1.0]
    
    >>>symbolic_uniform_mutation_kernel(3,mu, True)
    Out[1]: 
    [-2*mu/3 + 1.0,          mu/3,          mu/3]
    [         mu/3, -2*mu/3 + 1.0,          mu/3]
    [         mu/3,          mu/3, -2*mu/3 + 1.0]
    """ 
    a_matrix =  sympy.Matrix(number_of_strategies,number_of_strategies, lambda i,j: 0)
    weight = (number_of_strategies - 1)
    if self_interaction:
        weight = number_of_strategies
    for i in xrange(0, number_of_strategies):
        for j in xrange(0, number_of_strategies):
            if i != j:
                a_matrix[i, j] = (mutation_probability/weight)
    for i in xrange(0, number_of_strategies):
        a_matrix[i,i] = 1.0 - sum(a_matrix[i,:])   
    return a_matrix



def submatrix(game, i, j ):
    """
    Returns a 2x2 matrix, made up of indices i, and j of the given game
    
    Parameters
    ----------
    game: sympy.Matrix
    i: int
    j:int
    
    Returns
    -------
    out: Matrix
    
    See also:
    ---------
    Non symbolic equivalent pyevodyn.submatrix
    
    Examples
    --------
    submatrix(Matrix([[1,2,3],[4,5,6],[7,8,9]]), 1,2)
    >>>Out[1]: 
    [5, 6]
    [8, 9]
    """
    return sympy.Matrix([[game[i,i],game[i,j]],[game[j,i],game[j,j]]])


def antal_l_coefficient(index, game_matrix):
    """
    Returns the L_index coefficient, according to Antal et al. (2009), as given by equation 1.
    L_k = \frac{1}{n} \sum_{i=1}^{n} (a_{kk}+a_{ki}-a_{ik}-a_{ii})
    Parameters
    ----------
    index: int
    game_matrix: sympy.Matrix
    
    Returns
    -------
    out: sympy.Expr
    
    
    Examples:
    --------
    >>>a = Symbol('a')
    >>>antal_l_coefficient(0, Matrix([[a,2,3],[4,5,6],[7,8,9]]))
    Out[1]: 2*(a - 10)/3
    
    
    >>>antal_l_coefficient(0, Matrix([[1,2,3],[4,5,6],[7,8,9]]))
    Out[1]: -6
    """
    size = game_matrix.shape[0]
    suma = 0
    for i in range(0,size):
        suma = suma + (game_matrix[index,index] + game_matrix[index,i] - game_matrix[i,index] - game_matrix[i,i])
    return sympy.together(sympy.simplify(suma/size),size)

def antal_h_coefficient(index, game_matrix):
    """
    Returns the H_index coefficient, according to Antal et al. (2009), as given by equation 2.
    H_k = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} (a_{kj}-a_{jj})
    Parameters
    ----------
    index: int
    game_matrix: sympy.Matrix
    
    Returns
    -------
    out: sympy.Expr
    
    
    Examples:
    --------
    >>>a = Symbol('a')
    >>>antal_h_coefficient(0, Matrix([[a,2,3],[4,5,6],[7,8,9]]))
    Out[1]: (2*a - 29)/9
    
    >>>antal_h_coefficient(0, Matrix([[1,2,3],[4,5,6],[7,8,9]]))
    Out[1]: -3
    """
    size = game_matrix.shape[0]
    suma = 0
    for i in range(0,size):
        for j in range(0, size):
            suma = suma + (game_matrix[index,i] - game_matrix[i,j])
    return sympy.together(sympy.simplify(suma/(size**2)),size)

def antal_strategy_abundance(index, game_matrix, intensity_of_selection, population_size, mutation_probability):
    """
    Returns the abundace of strategy index for weak selection, according to Antal et al. (2009), as given by equation 20.
    
    Parameters
    ----------
    index:int
    game_matrix: sympy.Matrix (must be square)
    intensity_of_selection: sympy.Expr
    population_size: sympy.Expr
    mutation_probability: sympy.Expr
    
    Returns
    -------
    sympy.Expr
    
    Examples:
    --------
    >>>b,c,delta,u,N = symbols('b,c,delta,u,N')
    >>>pd = Matrix([[b-c, -c],[b, 0]])
    >>>antal_strategy_abundance(0, pd,delta,N, u)
    Out[1]: N*c*delta*(u - 1)/(4*(N*u + 1)) + 1/2
    """
    size = game_matrix.shape[0]
    return (1 + sympy.simplify((intensity_of_selection*population_size*(1-mutation_probability))*sympy.simplify((antal_l_coefficient(index,game_matrix) + population_size*mutation_probability*antal_h_coefficient(index,game_matrix))/((1+population_size*mutation_probability)*(2+population_size*mutation_probability)))))/size

def stationary_distribution_weak_selection(game_matrix, intensity_of_selection, population_size, mutation_probability):
    """
    Returns the stationary distribution for weak selection, according to Antal et al. (2009), as given by equation 20.
    
    Parameters
    ----------
    game_matrix: sympy.Matrix (must be square)
    intensity_of_selection: sympy.Expr
    population_size: sympy.Expr
    mutation_probability: sympy.Expr
    
    Returns
    -------
    sympy.Matrix
    
    Examples:
    --------
    >>>b,c,delta,u,N = symbols('b,c,delta,u,N')
    >>>pd = Matrix([[b-c, -c],[b, 0]])
    >>>stationary_distribution_weak_selection(pd,delta,100,u)
    Out[1]: 
    [ 25*c*delta*(u - 1)/(100*u + 1) + 1/2]
    [25*c*delta*(-u + 1)/(100*u + 1) + 1/2]
    """
    return sympy.Matrix(game_matrix.shape[0],1, lambda i,j: antal_strategy_abundance(i, game_matrix, intensity_of_selection, population_size, mutation_probability))


def symbolic_matrix_to_array(symbolic_matrix):
    """
    Converts a sympy.Matrix without symbols into a numpy array
    
    Parameters
    ----------
    symbolic_matrix: sympy.Matrix (without symbols)
    
    Returns
    -------
    np.ndarray
    
    Examples:
    --------
    >>>symbolic_matrix_to_array(Matrix([[1,2],[3,4]]))
    Out[1]: 
    array([[ 1.,  2.],
       [ 3.,  4.]])
    """
    shape_of_the_symbolic_matrix=np.shape(symbolic_matrix)
    ans_array=np.zeros(shape_of_the_symbolic_matrix)
    for i in range(0,shape_of_the_symbolic_matrix[0]):
        for j in range(0,shape_of_the_symbolic_matrix[1]):
            ans_array[i,j]=sympy.N(symbolic_matrix[i,j])
    return ans_array


def array_to_symbolic_matrix(array):
    """
    Converts a numpy array into sympy.Matrix
    
    Parameters
    ----------
    np.ndarray
    
    Returns
    -------
    symbolic_matrix: sympy.Matrix (without symbols)
    
    Examples:
    --------
    #TODO: examples, TEST
    """
    return sympy.Matrix(array.shape[0],array.shape[1], lambda i,j: array[i,j])




#TODO: Document from here below

def load_formula(file_name):
    """
    Loads a pickled formula
    """
    try:
        with open(file_name, "rb") as f: 
            ans = pickle.load(f)
            if not isinstance(ans,(sympy.Expr,sympy.Matrix)):
                raise TypeError("The pickled object must be a sympy expresion or a sympy Matrix")
            return ans
    except IOError:
        print 'File ' + str(file_name) +'does not exists'

def save_formula(formula, file_name):
    """
    Saves a formula
    """
    if not isinstance(formula,(sympy.Expr,sympy.Matrix)):
        raise TypeError("The object to pickle must be a sympy expresion or a sympy Matrix")
    else:
        pickle.dump(formula, open(file_name, "wb" ) )


def symbols_involved(expression):
    """
    Lists the symbols that are present in this expression. 
    """
    expression.atoms(sympy.Symbol)

