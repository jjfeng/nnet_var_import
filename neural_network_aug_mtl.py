import time
import logging as log

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork
from neural_network_basic import NeuralNetworkBasic

class NeuralNetworkAugMTL(NeuralNetworkBasic):
    """
    A neural network that estimates the full and reduced conditional means.
    Augments the original covariates with a missingness-status vector.
    Trains by minimizing multi-task loss in the paper, via AdamOptimizer.
    """
    def __init__(self,
            layer_sizes=None,
            var_import_idxs=[],
            ridge_param=0.01,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            output_act_func=None,
            sgd_sample_size=2000,
            nan_fill_config=None,
            missing_value_fill=None):
        """
        @param layer_sizes: how many nodes per layer (all layers!), the first layer should be double the number of features
                            since the inputs are the covariates and missingness status
        @param var_import_idxs: list of lists for all the groups of variables we want to estimate importance for
        @param ridge_param: penalty parameter for ridge penalty on network weights
        @param num_inits: number of random initializations
        @param max_iters: max number of training iterations
        @param act_func: activation function for the hidden layers, choices: ["relu", "tanh"]
        @param sgd_sample_size: number of samples for batch sgd
        @param nan_fill_config: a dictionary indicating normal values for the nan data --
                            used for nan values that are not missing at random
        @param missing_value_fill: what to fill in for masked values
                                    None means normal distribution,
                                    otherwise this is a float and we fill in masked values with this constant, e.g. 0
        """
        self.layer_sizes = layer_sizes
        self.var_import_idxs = var_import_idxs
        self.ridge_param = ridge_param
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.output_act_func = output_act_func
        self.sgd_sample_size = sgd_sample_size
        self.nan_fill_config = nan_fill_config
        self.missing_value_fill = missing_value_fill
        if self.layer_sizes:
            # You can only initialize the NN if you have layer_sizes...
            self._init_nn()

    def get_params(self, deep=True):
        return {
            "layer_sizes": self.layer_sizes,
            "var_import_idxs": self.var_import_idxs,
            "ridge_param": self.ridge_param,
            "max_iters": self.max_iters,
            "num_inits": self.num_inits,
            "act_func": self.act_func,
            "output_act_func": self.output_act_func,
            "sgd_sample_size": self.sgd_sample_size,
            "nan_fill_config": self.nan_fill_config,
            "missing_value_fill": self.missing_value_fill,
        }

    def set_params(self, **params):
        if "layer_sizes" in params:
            self.layer_sizes = params["layer_sizes"]
        if "var_import_idxs" in params:
            self.var_import_idxs = params["var_import_idxs"]
        if "ridge_param" in params:
            self.ridge_param = params["ridge_param"]
        if "max_iters" in params:
            self.max_iters = params["max_iters"]
        if "num_inits" in params:
            self.num_inits = params["num_inits"]
        if "act_func" in params:
            self.act_func = params["act_func"]
        if "output_act_func" in params:
            self.output_act_func = params["output_act_func"]
        if "sgd_sample_size" in params:
            self.sgd_sample_size = params["sgd_sample_size"]
        if "nan_fill_config" in params:
            self.nan_fill_config = params["nan_fill_config"]
        if "missing_value_fill" in params:
            self.missing_value_fill = params["missing_value_fill"]
        self._init_nn()

    def _init_nn(self):
        assert self.layer_sizes[0] % 2 == 0
        self.num_p = self.layer_sizes[0]/2

        # Input layers
        self.x_ph = tf.placeholder(tf.float32, [None, self.layer_sizes[0]])
        self.y_ph = tf.placeholder(tf.float32, [None, self.layer_sizes[-1]])

        output_act_func = tf.nn.sigmoid if self.output_act_func == "sigmoid" else None
        self.nn = NeuralNetwork.create_full_nnet(
                self.layer_sizes,
                self.x_ph,
                act_func=getattr(tf.nn, self.act_func),
                output_act_func=output_act_func)

        # Create MSE loss function
        self.y_pred = self.nn.layers[-1]
        self.loss = 0.5 * tf.reduce_mean(tf.pow(self.y_ph - self.y_pred, 2))

        # Create ridge loss
        self.ridge_full = tf.add_n([tf.nn.l2_loss(w) for w in self.nn.coefs])
        self.ridge_pen = 0.5 * self.ridge_param * self.ridge_full

        self.grad_optimizer = tf.train.AdamOptimizer()

        self.coef_list = self.nn.coefs
        self.intercept_list = self.nn.intercepts

        # We are training with a multi-task loss but this is a regular loss here.
        # This is fine since the multi-task loss is a result of the observations that we feed in.
        # Each observation will correspond to some term in the MTL
        self.total_loss = tf.add(self.loss, self.ridge_pen)
        self.train_op = self.grad_optimizer.minimize(self.total_loss, var_list=self.coef_list + self.intercept_list)

        self.coef_phs, self.coef_assign_ops = self._create_ph_assign_ops(self.coef_list)
        self.intercept_phs, self.intercept_assign_ops = self._create_ph_assign_ops(self.intercept_list)

        self.scaler = StandardScaler()

    def create_x_filtered(self, x, filter_mask_idx, fill_val=None):
        """
        Create the augmented input vector that contains the original covariates and missing status.
        This will input random values for the missing covariates

        @param x: covariate values
        @param filter_mask_idx: mask these covariates, i.e. treat them as missing
        @param fill_val: how to fill in the missing covariate values -- None means std normal, otherwise it should be a scalar and we use the same constant for all missing values
        @return the augmented inputs
        """
        copy_x = np.copy(x)
        mask_x = np.zeros(x.shape)
        if filter_mask_idx is not None:
            if fill_val is not None:
                copy_x[:, filter_mask_idx] = fill_val
            else:
                copy_x[:, filter_mask_idx] = np.random.randn(x.shape[0], len(filter_mask_idx))
            mask_x[:, filter_mask_idx] = 1

        # Final processing...
        # Fill in nan values -- we do not indicate missingness in this case
        self._partial_process_nan_values(copy_x)
        # Any remaining nan values should be filled in assing missingness at random
        # nan values are also missing
        remain_nan_mask = np.isnan(copy_x)
        mask_x[remain_nan_mask] = 1
        if fill_val is not None:
            copy_x[remain_nan_mask] = fill_val
        else:
            copy_x[remain_nan_mask] = np.random.randn(remain_nan_mask.sum())
        return np.hstack([copy_x, mask_x])

    def create_dataset_for_multitask_learning(self, X, y, sgd=False, fill_val=None):
        """
        @param X: observed covariates
        @param y: observed response
        @param sgd: if doing SGD, then sample the data

        @return a training (sub)set for one iteration of (sgd) gradient descent
        """
        var_import_idxs_aug = [None] + self.var_import_idxs
        n_filters = len(var_import_idxs_aug)
        if sgd:
            # Create a smaller dataset. Randomly choose which observations to sample
            # To do this, we randomly draw indices from `# of var import groups` * `# of observations`.
            # Partitions of the indices represent which observations are drawn for training which var import groups
            sample_size = min(self.sgd_sample_size, y.size)
            rand_choice = np.random.choice(
                    y.size * n_filters,
                    size=sample_size,
                    replace=True)
            samp_choices = [
                rand_choice[(rand_choice >= i * y.size) & (rand_choice < (i + 1) * y.size)] - i * y.size
                for i in range(n_filters)]

            # Based on the sampled indices, now actually create the augmented input vectors
            new_x_train = np.vstack([
                self.create_x_filtered(X[samp_idx, :], filter_idxs, fill_val=fill_val) for samp_idx, filter_idxs in zip(samp_choices, var_import_idxs_aug)])
            new_y_train = np.vstack(
                [y[samp_idx,:] for samp_idx in samp_choices])
            return new_x_train, new_y_train
        else:
            # Not doing SGD. So just augment and return
            new_x_train = np.vstack([
                self.create_x_filtered(X, filter_idxs, fill_val=fill_val) for filter_idxs in var_import_idxs_aug])
            new_y_train = np.tile(y, (n_filters, 1))
            return new_x_train, new_y_train
