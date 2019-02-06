import numpy as np
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)                  # initialization of cluster, all examples of X are assigned to cluster 1

        means = np.zeros((self.k, D))   # define the means with dimension k*D 
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]            # initialize the means of each group kk by randomly choose a row of X, by using i = np.random.randint(N)

        while True:
            y_old = y

            # Compute euclidean distance to each mean, and determine the cluster
            dist2 = euclidean_dist_squared(X, means)    # dist2 is a N*K matrix, representing the euclidean distance of every example at every means[kk]
            dist2[np.isnan(dist2)] = np.inf             # 
            y = np.argmin(dist2, axis=1)                # y is a N*1 vector, return kk that has the minimum distance at every row

            # Update means
            for kk in range(self.k):
                if np.any(y==kk):                       # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y==kk].mean(axis=0)   # means[kk] is a 1*D vector recording the D dimensional mean for examples in cluster kk

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)    
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)
    
    def error(self, X):
        N, D = X.shape
        means = self.means
        dist2 = euclidean_dist_squared(X, means)    
        dist2[np.isnan(dist2)] = np.inf
        y = self.predict(X)
        dist_error = 0
        for n in range(N):
            dist_error += dist2[n,y[n]]
        return dist_error
        
