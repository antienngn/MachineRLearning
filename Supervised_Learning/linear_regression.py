import math
import numpy as np

class Regression(object):
    """
    Base regression model.
    Parameter:
    n_iterations: float
        The number of training iterations of models
    learning_rate: float
        The step will be used to updating the weights
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
    
    def init_weight(self, n_features):
        "Initialize random weights for model"
        limit = 1 / math.sqrt(n_features)
        self.weight = np.random.uniform(-limit, limit, (n_features,))
    
    def fit(self,X,Y):
        X = np.insert(X, 0, 1, axis=1)
        self.init_weight(n_features=X.shape[1])
        self.training_errors = []
        
        for i in range(self.n_iterations):
            y_predict = X.dot(self.weight)
            

    def predict(self,X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.weight)
        return y_pred

class l1Regularization: