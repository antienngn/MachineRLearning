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
    
    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)
        self.init_weight(n_features=X.shape[1])
        self.training_errors = []
        
        for i in range(self.n_iterations):
            y_predict = X.dot(self.weight)
            mse = np.mean(0.5 * (Y-y_predict)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = -(Y-y_predict).dot(X) + self.regularizatio.grad(self.w)
            self.w = self.w - self.learning_rate * grad_w
            
    def predict(self,X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.weight)
        return y_pred

class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.01):
        super().__init__(n_iterations, learning_rate)
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
    
    def fit(self, X, Y):
        super().fit(X, Y)