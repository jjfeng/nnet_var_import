import numpy as np

def variableImportanceIC(full = None, reduced = None, y = None, standardized = True):
    """
    Calculate the includence curve for the variable importance parameters
    @param full: fits to the full data (numpy array)
    @param reduced: fits to the reduced data (numpy array)
    @param y: observed response
    @param standardized: standardized or non-standardized parameter
    @return the includence curve
    """

    ## calculate naive estimates
    naive_j = np.mean(np.square(full - reduced))
    naive_var = np.mean(np.square(y - np.mean(y)))
    
    ## now calculate ic
    if(standardized):
        ret = (2*np.multiply(y - full, full - reduced) + np.square(full - reduced))/naive_var - (np.square(y - np.mean(y)))*naive_j/(naive_var ** 2)
    else:
        ret = (2*np.multiply(y - full, full - reduced) + np.square(full - reduced) - naive_j)
    
    return ret
