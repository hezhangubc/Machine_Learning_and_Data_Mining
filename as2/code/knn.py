"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
       self.X = X # just memorize the trianing data
       self.y = y       
        
    def predict(self, Xtest):
        X = self.X
        y = self.y
        t = Xtest.shape[0]
        k=self.k

        # Squared distances between X and Xtest
        distance = utils.euclidean_dist_squared(X, Xtest)

        # Get yhat
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            indices = np.argsort(distance[:,i])

            # Choose k nearst neighbours
            set=indices[:k]
            # Get yhat from mode
            yhat[i] = stats.mode(y[set])[0][0]

        return yhat