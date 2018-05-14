class NeuralNetworkParams:
    """
    Class for storing neural network parameters, companion for NeuralNetworkBasic (and its subclasses)
    """
    def __init__(self, coefs, intercepts, scaler):
        """
        @param coefs: list of coefficients in the network (ordering determined by coef_list attr in NeuralNetworkBasic)
        @param intercepts: list of intercepts in th network (ordering determined by intercept_list attr in NeuralNetworkBasic)
        @param scaler: a trained StandardScaler from scikitlearn
        """
        self.coefs = coefs
        self.intercepts = intercepts
        self.scaler = scaler

    def __str__(self):
        return "Coefs %s, Intercepts %s" % (self.coefs, self.intercepts)
