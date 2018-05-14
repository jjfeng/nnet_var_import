import numpy as np

import data_gen_funcs

class Dataset:
    """
    Stores data
    """
    def __init__(self, x_train=None, y_train=None, y_train_true=None, x_test=None, y_test=None, y_test_true=None):
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_true = y_train_true
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_true = y_test_true

class DataGenerator:
    """
    Conditional expectation functions
    """
    def __init__(self, func_name, n_train, n_test, num_p, noise_sd=0, max_x=2):
        self.n_train = n_train
        self.n_test = n_test
        self.num_p = num_p
        self.max_x = max_x
        self.min_x = -max_x
        self.noise_sd = noise_sd
        self.func = getattr(data_gen_funcs, func_name)

    def create_data(self):
        x_train, y_train, y_train_true = self._create_data(self.n_train)
        x_test, y_test, y_test_true = self._create_data(self.n_test)
        return Dataset(
            x_train, y_train, y_train_true,
            x_test, y_test, y_test_true,
        )

    def _create_data(self, size_n):
        if size_n <= 0:
            return None, None, None

        xs = np.random.rand(size_n, self.num_p) * (self.max_x - self.min_x) + self.min_x
	    # regression
        true_ys = self.func(xs)
        true_ys = np.reshape(true_ys, (true_ys.size, 1))
        eps = np.random.randn(size_n, 1) * self.noise_sd
        eps_norm = np.linalg.norm(eps)
        y_norm = np.linalg.norm(true_ys)
        y = true_ys + eps
        return xs, y, true_ys
