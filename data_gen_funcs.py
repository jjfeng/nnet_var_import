import numpy as np

from functools import reduce

def six_func(xs):
    """
    Corresponds to simulation example in the paper
    """
    return np.multiply(np.multiply(np.sin(xs[:,0] + 2*xs[:,1]), xs[:,0]), np.cos(xs[:,2] + 2*xs[:,3]))

def six_func_reduced(xs, filter_idxs, sample_size=5000, max_x=2, min_x=-2):
    """
    Calculates the reduced conditional mean via sampling

    @param xs: covariates
    @param filter_idxs: the variables to mask
    @param sample_size: number of samples to use to estimate reduced conditional mean
    @param max_x: the max value of the missing covariates
    @param min_x: the min value of the missing covariate
    Assumes the missing covariates are iid, x ~ Unif[min_x, max_x]

    @return reduced conditional mean
    """
    samp_x = np.random.rand(sample_size, 2) * (max_x - min_x) + min_x
    if list(filter_idxs) == [0,1]:
        mean_filtered = np.mean(np.multiply(np.sin(xs[:,0] + 2*xs[:,1]), xs[:,0]))
        return mean_filtered * np.multiply(np.sin(xs[:,0] + 2*xs[:,1]), xs[:,0])
    elif list(filter_idxs) == [2,3]:
        mean_filtered = np.mean(np.cos(samp_x[:,0] + 2*samp_x[:,1]))
        return mean_filtered * np.cos(xs[:,2] + 2*xs[:,3])
    elif list(filter_idxs) == [4,5]:
        return six_func(xs)
    else:
        raise ValueError("HUH? bad input")

def eight_additive(xs):
    """
    Corresponds to simulation example in the paper
    """
    return xs[:,0] + np.square(xs[:,1]) + np.sin(xs[:,2]) + np.cos(xs[:,3]) + np.square(xs[:,4] + 1) - 2 * xs[:,5] + np.maximum(xs[:,6], 0) + xs[:,7]

def eight_additive_reduced(xs, filter_idxs):
    """
    Calculates the reduced conditional mean analytically
    Assume missing xs are independent, x ~ Unif[-2, 2]

    @param xs: covariates
    @param filter_idxs: the variables to mask
    @return reduced conditional mean
    """
    true_reduced_y = np.zeros(xs.shape[0])
    if 0 not in filter_idxs:
        true_reduced_y += xs[:,0]
    
    if 1 in filter_idxs:
        true_reduced_y += 4.0/3
    else:
        true_reduced_y += np.square(xs[:,1])
    
    if 2 not in filter_idxs:
        true_reduced_y += np.sin(xs[:,2])

    if 3 in filter_idxs:
        true_reduced_y += 0.5 * np.sin(2)
    else:
        true_reduced_y += np.cos(xs[:,3])

    if 4 in filter_idxs:
        true_reduced_y += 7.0/3
    else:
        true_reduced_y += np.square(xs[:,4] + 1)

    if 5 not in filter_idxs:
        true_reduced_y -= 2 * xs[:,5]

    if 6 in filter_idxs:
        true_reduced_y += 0.5
    else:
        true_reduced_y += np.maximum(xs[:,6], 0)

    if 7 not in filter_idxs:
        true_reduced_y += xs[:,7]

    return true_reduced_y
