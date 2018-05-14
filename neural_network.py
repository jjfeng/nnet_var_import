import time
import logging as log

import tensorflow as tf
import numpy as np

from sklearn.base import BaseEstimator

class NeuralNetwork(BaseEstimator):
    """
    Super class for neural nets.
    Has functionality for making ordinary nnets.
    """
    def __init__(self, coefs, intercepts, layers):
        """
        initialization for fully connected, standard nnets

        @param layers: list of nodes at each layer
        """
        self.coefs = coefs
        self.intercepts = intercepts
        self.var_list = coefs + intercepts
        self.layers = layers

    @staticmethod
    def get_init_rand_bound_tanh(shape):
        # Used for tanh
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(6. / np.sum(shape))

    @staticmethod
    def get_init_rand_bound_sigmoid(shape):
        # Use the initialization method recommended by Glorot et al.
        return np.sqrt(2. / np.sum(shape))

    @staticmethod
    def create_tf_var(shape):
        bound = NeuralNetwork.get_init_rand_bound_tanh(shape)
        return tf.Variable(tf.random_uniform(shape, minval=-bound, maxval=bound))

    @staticmethod
    def create_full_nnet(layer_sizes, input_layer, act_func=tf.nn.tanh, output_act_func=None):
        """
        @param input_layer: input layer (tensor)
        @param layer_sizes: size of each layer (input to output)
        """
        coefs = []
        intercepts = []
        layers = []
        n_layers = len(layer_sizes)
        for i in range(n_layers - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W_size = [fan_in, fan_out]
            b_size = [fan_out]
            W = NeuralNetwork.create_tf_var(W_size)
            b = NeuralNetwork.create_tf_var(b_size)
            layer = tf.add(tf.matmul(input_layer, W), b)
            if i < n_layers - 2:
                # If not last layer, add activation function
                layer = act_func(layer)
            else:
                # is the layer layer
                if output_act_func is not None:
                    layer = output_act_func(layer)
            input_layer = layer
            coefs.append(W)
            intercepts.append(b)
            layers.append(layer)

        return NeuralNetwork(coefs, intercepts, layers)
