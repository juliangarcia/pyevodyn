'''
Created on Aug 8, 2012

@author: garcia
'''

import pandas

def time_series_matrix_to_pandas_dataframe(matrix):
    """
    Transforms a numpy array, as returned by MoranProcess.simulate_time_series into a DataFrame indexed by the generation
    
    #TODO: Document
    """
    number_of_strategies = matrix.shape[1] - 2
    d = dict()
    # add each strategy column
    for i in xrange(0, number_of_strategies):
        d[str(i)] = matrix[:,i+1]
    return pandas.DataFrame(d, index= matrix[:,0].tolist())
    
    
    
    