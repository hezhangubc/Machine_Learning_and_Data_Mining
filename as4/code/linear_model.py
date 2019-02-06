import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


# function for L2 regularization
class logRegL2:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100, lammy=1.0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function plus l2 regularization value
        f = np.sum(np.log(1. + np.exp(-yXw))) + 0.5 * self.lammy * np.inner(w,w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


# function for L1 regularization
class logRegL1:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100, L1_lambda=1.0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = L1_lambda
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function plus l2 regularization value
        f = np.sum(np.log(1. + np.exp(-yXw))) 

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                selected_new = selected | {i} # tentatively add feature "i" to the seected set
                w, loss_value = minimize(list(selected_new))
                loss_value += self.L0_lambda * np.count_nonzero(w)

                if loss_value < minLoss:
                    minLoss  = loss_value
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
    
class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, W, X, y):
        n, d = X.shape
        k = self.n_classes
        
        # reshape the vector w to a matrix W
        W = np.reshape(W, (k, d))           # the dimension of W is k*d
        G = np.zeros((k,d))                 # the dimension of the matrix is k*d
        f_1 = np.zeros(k)
        I = np.unique(y)
        
        # calculate each elemant in the gradient matrix
        for i in range(k):
            I_i = np.where(y == I[i])
            f_1[i] = np.sum(X[I_i]@W[i].T)
            p_1 = np.exp(X@W[i].T)/np.sum(np.exp(X@W.T), axis = 1)
            for j in range(d):
                G[i,j] = -np.sum(X[I_i,j]) + p_1.T@X[:,j]
        
        F = -np.sum(f_1) + np.sum(np.log(np.sum(np.exp(X@W.T),axis = 1)))
        G = G.flatten()
        
        return F, G

    
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        
        m = self.n_classes*d
        self.w = np.reshape(self.W, m)
        
        self.w, f = findMin.findMin(self.funObj, self.w, self.maxEvals, X, y)
        self.W = np.reshape(self.w,(self.n_classes,d))


    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)







