import variable_importance_ic as ic
import numpy as np

def variableImportance(full = None, reduced = None, y = None, standardized = True):
    """
    PURPOSE: calculate naive and one-step estimators for both variable importance parameters
    relies on variableImportanceIC to calculate the influence curve
    
    @param full: the model fit to the full data
    @param reduced: the model fit to the reduced data
    @param y: the outcome
    @param n: the sample size
    @param standardized: whether or not to compute the standardized estimator
    @return the naive estimate and one-step estimate
    """
    ## calculate naive
    if(standardized):
        naive = np.mean((full - reduced) ** 2)/np.mean((y - np.mean(y)) ** 2)
    else:
        naive = np.mean((full - reduced) ** 2)
    
    ## now add on mean of ic
    var_import = ic.variableImportanceIC(full, reduced, y, standardized)
    onestep = naive + np.mean(var_import)
    
    ## return as an array
    ret = np.array([naive, onestep])
    return ret
