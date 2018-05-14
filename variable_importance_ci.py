import numpy as np
from scipy.stats import norm

def variableImportanceCI(est = None, se = None, level = 0.95):
    """
    Calculate the CI for the one-step

    @param est: the estimate
    @param se: the standard error
    @param n: the sample size
    @param level: the level of the CI
    @return the confidence interval
    """
    ## get alpha
    a = (1 - level)/2
    a = np.array([a, 1 - a])
    ## calculate the quantiles
    fac = norm.ppf(a)
    ## set up the ci array
    ci = np.zeros((est.shape[0], 2))
    ## create it
    ci = est + np.outer((se), fac)
    return ci[0]
