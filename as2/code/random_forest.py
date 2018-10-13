from random_tree import RandomTree
import numpy as np
from scipy import stats

class RandomForest:

    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        
        self.trees = []
        num_trees=self.num_trees
        max_depth=self.max_depth
        
        for m in range(num_trees):
            tree = RandomTree(max_depth = max_depth)
            tree.fit(X,y)
            self.trees.append(tree)

    def predict(self, X):
        num_trees=self.num_trees
        trees=self.trees
        t = X.shape[0]
        yhats = np.ones((t,num_trees), dtype=np.uint8)

        # Predict yhat
        for m in range(num_trees):
            yhats[:,m] = trees[m].predict(X)
        return yhats

        # Get mode
        # use axis=1 to capture mode value across columns (models)
        # use  [0] get first array column (mode value)
        # use flatten() option to a collapsed one-dimension array
        mode=stats.mode(yhats, axis=1)[0] 
        return mode.flatten()
