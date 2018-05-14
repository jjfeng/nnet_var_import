import time
import logging

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

from neural_network_wrapper import NeuralNetworkParams
from neural_network import NeuralNetwork

class NeuralNetworkBasic(NeuralNetwork):
    """
    The simplest fully-connected neural network for estimating a conditional mean
    Uses AdamOptimizer to minimize mean squared error.
    Uses default values in AdamOptimizer
    """
    def __init__(self,
            layer_sizes=None,
            ridge_param=0.01,
            num_inits=1,
            max_iters=200,
            act_func="tanh",
            output_act_func=None,
            sgd_sample_size=2000,
            nan_fill_config=None,
            missing_value_fill=None):
        """
        @param layer_sizes: how many nodes per layer (all layers!)
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
        self.ridge_param = ridge_param
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.act_func = act_func
        self.output_act_func = output_act_func
        self.sgd_sample_size = sgd_sample_size
        self.nan_fill_config = nan_fill_config
        self.missing_value_fill = missing_value_fill
        if self.layer_sizes:
            self._init_nn()

    def get_params(self, deep=True):
        return {
            "layer_sizes": self.layer_sizes,
            "ridge_param": self.ridge_param,
            "max_iters": self.max_iters,
            "num_inits": self.num_inits,
            "act_func": self.act_func,
            "output_act_func": self.output_act_func,
            "sgd_sample_size": self.sgd_sample_size,
        }

    def set_params(self, **params):
        if "layer_sizes" in params:
            self.layer_sizes = params["layer_sizes"]
        if "ridge_param" in params:
            self.ridge_param = params["ridge_param"]
        if "act_func" in params:
            self.act_func = params["act_func"]
        if "output_act_func" in params:
            self.output_act_func = params["output_act_func"]
        if "max_iters" in params:
            self.max_iters = params["max_iters"]
        if "num_inits" in params:
            self.num_inits = params["num_inits"]
        if "sgd_sample_size" in params:
            self.sgd_sample_size = params["sgd_sample_size"]
        self._init_nn()

    def _init_nn(self):
        self.num_p = self.layer_sizes[0]

        # Input layers; create placeholders in tensorflow
        self.x_ph = tf.placeholder(tf.float32, [None, self.layer_sizes[0]])
        self.y_ph = tf.placeholder(tf.float32, [None, self.layer_sizes[-1]])

        # Construct the basic neural network
        output_act_func = tf.nn.sigmoid if self.output_act_func == "sigmoid" else None
        self.nn = NeuralNetwork.create_full_nnet(
            self.layer_sizes,
            self.x_ph,
            act_func=getattr(tf.nn, self.act_func),
            output_act_func=output_act_func)

        # Create MSE loss function
        self.y_pred = self.nn.layers[-1]
        self.loss = 0.5 * tf.reduce_mean(tf.pow(self.y_ph - self.y_pred, 2))

        # Create ridge loss; adds loss for each coefficient in neural network to tensorflow
        self.ridge_full = tf.add_n([tf.nn.l2_loss(w) for w in self.nn.coefs])
        self.ridge_pen = 0.5 * self.ridge_param * self.ridge_full

        self.grad_optimizer = tf.train.AdamOptimizer()

        self.coef_list = self.nn.coefs
        self.intercept_list = self.nn.intercepts

        self.total_loss = tf.add(self.loss, self.ridge_pen)
        self.train_op = self.grad_optimizer.minimize(self.total_loss, var_list=self.coef_list + self.intercept_list)

        self.coef_phs, self.coef_assign_ops = self._create_ph_assign_ops(self.coef_list)
        self.intercept_phs, self.intercept_assign_ops = self._create_ph_assign_ops(self.intercept_list)

        self.scaler = StandardScaler()

    def _create_ph_assign_ops(self, var_list):
        """
        Create placeholders and assign ops for model parameters
        """
        all_phs = []
        assign_ops = []
        for var in var_list:
            ph = tf.placeholder(
                tf.float32,
                shape=var.shape,
            )
            assign_op = var.assign(ph)
            all_phs.append(ph)
            assign_ops.append(assign_op)
        return all_phs, assign_ops

    def _partial_process_nan_values(self, X):
        """
        Fill in random values within normal ranges for these missing covariates
        Essentially assumed values are not missing at random and are likely
        to take on something within a range

        Assumes nan_fill_config is dict with form:
        {
            <SOME_KEY>: {
                'indices': [list of feature indices in X],
                'range': [min val, max val]
            }
        }

        Then we take all the features corresponding to 'indices' and fill them randomly with
        values drawn from uniform distribution over 'range'. All nan values with those features
        will share the same randomly chosen value within the same observation (i.e. row in matrix X).

        note: modifies X matrix in-place
        """
        if self.nan_fill_config is None:
            return

        nan_mask = np.isnan(X)
        nan_locs = np.where(nan_mask)
        for _, measure_dict in self.nan_fill_config.iteritems():
            in_mask = np.isin(nan_locs[1], measure_dict["indices"])
            rows_to_update = nan_locs[0][in_mask]
            unique, counts = np.unique(rows_to_update, return_counts=True)
            cols_to_update = nan_locs[1][in_mask]
            feat_num_cols = len(measure_dict["indices"])
            assert rows_to_update.size % feat_num_cols == 0
            num_obs = rows_to_update.size / feat_num_cols
            normal_range = measure_dict["range"]
            X[(rows_to_update, cols_to_update)] = np.repeat(
                    np.random.uniform(normal_range[0], normal_range[1], size=num_obs),
                    feat_num_cols)

    def process_remaining_nan_values(self, X, fill_val=None):
        """
        If there are NaNs in the X matrix, fill them in with some values
        """
        if np.isnan(X).sum() == 0:
            return
        self._partial_process_nan_values(X)

        # There might be remaining nan values corresponding to covariates that we don't
        # think are missing not-at-random, i.e. they were not in self.nan_fill_config
        # so we plan on assuming their values were missing at random. Fill in based on `fill_val`
        final_nan_mask = np.isnan(X)
        if fill_val is None:
            X[final_nan_mask] = np.random.randn(final_nan_mask.sum())
        else:
            X[final_nan_mask] = fill_val

    def fit(self, X, y):
        """
        Fitting function with multiple initializations
        """
        st_time = time.time()
        new_X, y_train = self.create_dataset_for_multitask_learning(X, y, fill_val=0)
        self.process_remaining_nan_values(new_X, fill_val=0)
        self.scaler.fit(new_X)

        sess = tf.Session()
        best_loss = None
        self.model_params = None
        with sess.as_default():
            for n_init in range(self.num_inits):
                logging.info("FIT INIT %d", n_init)
                tf.global_variables_initializer().run()
                model_params, train_loss = self._fit_one_init(sess, X, y)
                if self.model_params is None or train_loss < best_loss:
                    self.model_params = model_params
                    logging.info("model params update %s", str(self.model_params))
                    best_loss = train_loss

        logging.info("best_loss %f (train time %f)", best_loss, time.time() - st_time)
        sess.close()

    def _fit_one_init(self, sess, X, y):
        """
        Fitting function for one initialization
        """
        cum_loss = 0
        for i in range(self.max_iters):
            # Get loss values and calculate gradients
            new_X, y_train = self.create_dataset_for_multitask_learning(
                    X,
                    y,
                    sgd=True,
                    fill_val=self.missing_value_fill)
            self.process_remaining_nan_values(new_X, fill_val=self.missing_value_fill)
            x_train = self.scaler.transform(new_X)
            _, unpen_loss, tot_loss, y_preds = sess.run([self.train_op, self.loss, self.total_loss, self.y_pred], feed_dict={
                self.x_ph: x_train,
                self.y_ph: y_train,
            })
            if i > 1000:
                cum_loss += tot_loss
            if i % 100 == 0:
                logging.info("iter %d: y_preds, mean %f var %f", i, y_preds.mean(), np.var(y_preds))
                logging.info("iter %d: unpen-loss %f", i, unpen_loss)
                logging.info("train %d: penalized-loss %f", i, tot_loss)
                if i > 1000:
                    logging.info("train %d: cum avg penalized-loss %f", i, cum_loss/(i - 1000))

        # Save model parameter values. Otherwise they will disappear!
        model_params = NeuralNetworkParams(
            [c.eval() for c in self.coef_list],
            [b.eval() for b in self.intercept_list],
            self.scaler
        )
        return model_params, tot_loss

    def _init_network_variables(self, sess):
        """
        Initialize network variables
        """
        for i, c_val in enumerate(self.model_params.coefs):
            sess.run(self.coef_assign_ops[i], feed_dict={self.coef_phs[i]: c_val})

        for i, intercept_val in enumerate(self.model_params.intercepts):
            sess.run(self.intercept_assign_ops[i], feed_dict={self.intercept_phs[i]: intercept_val})

    def create_x_filtered(self, x, filter_mask_idx=None, fill_val=None):
        """
        Prepares the x to have features masked
        For the basic NN, this does nothing. This should be overridden in subclasses.
        """
        return np.copy(x)

    def create_dataset_for_multitask_learning(self, x, y, sgd=False, fill_val=None):
        """
        Create a dataset for a training step in (sgd) gradient descent
        @param x: covariates
        @param y: response
        @param fill_val: ignored
        NOT SCALED!
        """
        if sgd:
            sample_size = min(self.sgd_sample_size, y.size)
            samp_idx = np.random.choice(
                    y.size,
                    size=sample_size,
                    replace=False)
            new_x_train = x[samp_idx, :]
            new_y_train = y[samp_idx,:]
            return new_x_train, new_y_train
        else:
            return x, y

    def predict(self, x, filter_idx=None):
        """
        @param x: covariates
        @param filter_mask_idx: which features are masked
        @return predictions corresponding to the (potentially masked) x
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)

            # Predict values
            filtered_x = self.create_x_filtered(x, filter_idx)
            self.process_remaining_nan_values(filtered_x)
            x_post_process = self.scaler.transform(filtered_x)
            y_pred = sess.run(self.y_pred, feed_dict={self.x_ph: x_post_process})

        sess.close()
        return y_pred

    def score(self, x, y):
        """
        Function used by cross validation function in scikit
        """
        sess = tf.Session()
        with sess.as_default():
            # Re-initialize neural network params since we are in a new session
            self._init_network_variables(sess)
            x_post_process, y_post_process = self.create_dataset_for_multitask_learning(x, y, fill_val=0)
            self.process_remaining_nan_values(x_post_process, fill_val=0)
            x_post_process = self.scaler.transform(x_post_process)
            loss = sess.run(self.loss, feed_dict={
                self.x_ph: x_post_process,
                self.y_ph: y_post_process,
            })

        sess.close()
        return -loss
